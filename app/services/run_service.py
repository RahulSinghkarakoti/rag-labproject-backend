from datetime import datetime, timezone
from typing import List
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.cancellation import cancellation_manager
from app.models.message import Message
from app.models.project import Project
from app.models.run import Run, RunStep
from app.models.thread import Thread


class RunService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_run_with_steps(self, run_id: UUID, user_id: int) -> Run:
        run = await self.db.scalar(
            select(Run)
            .join(Thread, Run.thread_id == Thread.id)
            .join(Project, Thread.project_id == Project.id)
            .options(selectinload(Run.steps))
            .where(Run.id == run_id, Project.user_id == user_id)
        )
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )
        run.steps.sort(key=lambda s: s.sequence)
        return run

    async def get_run_steps(self, run_id: UUID, user_id: int) -> List[RunStep]:
        run = await self.get_run_with_steps(run_id, user_id)
        return run.steps

    async def get_message_run_steps(
        self, message_id: UUID, user_id: int
    ) -> Run:
        message = await self.db.scalar(
            select(Message)
            .join(Thread, Message.thread_id == Thread.id)
            .join(Project, Thread.project_id == Project.id)
            .where(Message.id == message_id, Project.user_id == user_id)
        )
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found",
            )

        run_id_str = (
            message.metadata_.get("run_id") if message.metadata_ else None
        )
        run = None

        if run_id_str:
            try:
                target_uuid = UUID(str(run_id_str))
                run = await self.db.scalar(
                    select(Run)
                    .options(selectinload(Run.steps))
                    .where(Run.id == target_uuid)
                )
            except (ValueError, TypeError):
                run = None

        if not run:
            run = await self.db.scalar(
                select(Run)
                .options(selectinload(Run.steps))
                .where(
                    (Run.output_message_id == message_id)
                    | (Run.input_message_id == message_id)
                )
                .order_by(Run.started_at.desc())
            )

        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No execution steps found for this message",
            )

        run.steps.sort(key=lambda s: s.sequence)
        return run

    async def cancel_run(self, run_id: UUID, user_id: int) -> Run:
        await cancellation_manager.cancel(run_id)
        run = await self.get_run_with_steps(run_id, user_id)
        if run.status in ["completed", "failed", "cancelled"]:
            return run

        now = datetime.now(timezone.utc)
        run.status = "cancelled"
        run.completed_at = now
        run.error = {
            "code": "CANCELLED",
            "message": "User interrupted this response.",
        }

        for step in (run.steps or []):
            if step.status == "running":
                step.status = "cancelled"
                step.completed_at = now
                step.error = {
                    "code": "CANCELLED",
                    "message": "User interrupted this response.",
                }

        if run.output_message_id:
            msg = await self.db.scalar(
                select(Message).where(Message.id == run.output_message_id)
            )
            if msg:
                msg.status = "cancelled"
                msg.metadata_ = dict(msg.metadata_ or {})
                msg.metadata_["interrupted_by_user"] = True
                if not msg.content:
                    msg.content = "[User interrupted this response]"
                elif "[User interrupted this response]" not in msg.content:
                    msg.content = (
                        f"{msg.content}\n\n[User interrupted this response]"
                    )
        elif run.thread_id and run.input_message_id:
            in_msg = await self.db.scalar(
                select(Message).where(Message.id == run.input_message_id)
            )
            seq = ((in_msg.sequence or 0) + 1) if in_msg else 1
            parent_id = in_msg.id if in_msg else None
            new_msg = Message(
                thread_id=run.thread_id,
                parent_message_id=parent_id,
                role="assistant",
                content="[User interrupted this response]",
                sequence=seq,
                status="cancelled",
                metadata_={"run_id": str(run.id), "interrupted_by_user": True},
            )
            self.db.add(new_msg)
            await self.db.flush()
            run.output_message_id = new_msg.id

        await self.db.commit()
        await self.db.refresh(run)
        return run
