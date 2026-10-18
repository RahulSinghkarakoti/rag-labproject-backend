from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models.message import Message
from app.models.project import Project
from app.models.run import Run, RunStep
from app.models.thread import Thread
from app.schemas.common import APIResponse
from app.schemas.run import RunDetailsResponse, RunStepResponse

router = APIRouter(tags=["runs"])


async def _get_run_with_steps(run_id: UUID, user_id: UUID, db: AsyncSession) -> Run:
    run = await db.scalar(
        select(Run)
        .join(Thread, Run.thread_id == Thread.id)
        .join(Project, Thread.project_id == Project.id)
        .options(selectinload(Run.steps))
        .where(Run.id == run_id, Project.user_id == user_id)
    )
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.steps.sort(key=lambda s: s.sequence)
    return run


@router.get("/runs/{run_id}", response_model=APIResponse[RunDetailsResponse])
async def get_run_details(
    run_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_run_with_steps(run_id, current_user.id, db)
    return APIResponse(
        success=True,
        message="Run details retrieved successfully",
        data=RunDetailsResponse.model_validate(run),
    )


@router.get("/runs/{run_id}/steps", response_model=APIResponse[list[RunStepResponse]])
async def get_run_steps(
    run_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_run_with_steps(run_id, current_user.id, db)
    return APIResponse(
        success=True,
        message="Run steps retrieved successfully",
        data=[RunStepResponse.model_validate(step) for step in run.steps],
    )


@router.get("/messages/{message_id}/run-steps", response_model=APIResponse[RunDetailsResponse])
async def get_message_run_steps(
    message_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    message = await db.scalar(
        select(Message)
        .join(Thread, Message.thread_id == Thread.id)
        .join(Project, Thread.project_id == Project.id)
        .where(Message.id == message_id, Project.user_id == current_user.id)
    )
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    run_id_str = message.metadata_.get("run_id") if message.metadata_ else None
    run = None

    if run_id_str:
        try:
            target_uuid = UUID(str(run_id_str))
            run = await db.scalar(
                select(Run)
                .options(selectinload(Run.steps))
                .where(Run.id == target_uuid)
            )
        except (ValueError, TypeError):
            run = None

    if not run:
        run = await db.scalar(
            select(Run)
            .options(selectinload(Run.steps))
            .where((Run.output_message_id == message_id) | (Run.input_message_id == message_id))
            .order_by(Run.started_at.desc())
        )

    if not run:
        raise HTTPException(status_code=404, detail="No execution steps found for this message")

    run.steps.sort(key=lambda s: s.sequence)
    return APIResponse(
        success=True,
        message="Message run steps retrieved successfully",
        data=RunDetailsResponse.model_validate(run),
    )
