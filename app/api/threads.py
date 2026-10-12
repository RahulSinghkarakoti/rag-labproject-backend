import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select , func
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models import Thread , Message , Run
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.schemas.common import APIResponse
from app.schemas.thread import (
    MessageCreate,
    MessageResponse,
    ThreadCreate,
    ThreadResponse,
    ThreadUpdate,
)
from app.schemas.pipeline import PipelineVersionResponse
from app.rag import QueryEngine
from app.core.sse import create_sse_response

router = APIRouter(prefix="/api", tags=["Threads"])


async def _get_thread(thread_id: UUID, user_id: int, db: AsyncSession) -> Thread:
    thread = await db.scalar(select(Thread).where(Thread.id == thread_id, Thread.user_id == user_id))
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


async def _get_kbs(project_id: UUID, user_id: int, ids: list[UUID], db: AsyncSession):
    ids = list(dict.fromkeys(ids))
    kbs = list((await db.scalars(select(KnowledgeBase).where(
        KnowledgeBase.id.in_(ids), KnowledgeBase.project_id == project_id, KnowledgeBase.user_id == user_id,
    ))).all())
    if len(kbs) != len(ids):
        raise HTTPException(status_code=404, detail="One or more knowledge bases not found")
    if not ids or not await db.scalar(select(Document.id).where(Document.knowledge_base_id.in_(ids))):
        raise HTTPException(status_code=400, detail="Select a knowledge base with at least one document")
    return ids, kbs


@router.get("/projects/{project_id}/threads")
async def get_project_threads(project_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    if not await db.scalar(select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)):
        raise HTTPException(status_code=404, detail="Project not found")
    threads = (await db.scalars(select(Thread).where(Thread.project_id == project_id, Thread.user_id == current_user.id).order_by(Thread.updated_at.desc()))).all()
    return APIResponse(success=True, message="Threads retrieved successfully", data=[ThreadResponse.model_validate(t) for t in threads])


@router.post("/projects/{project_id}/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(project_id: UUID, thread_data: ThreadCreate, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    if not await db.scalar(select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)):
        raise HTTPException(status_code=404, detail="Project not found")
    kb_ids, kbs = await _get_kbs(project_id, current_user.id, thread_data.knowledge_base_ids, db)
    config = thread_data.config.model_dump() if thread_data.config else {}
    pipeline = Pipeline(name=thread_data.title, description="Thread pipeline", project_id=project_id, user_id=current_user.id)
    db.add(pipeline)
    await db.flush()
    version = PipelineVersion(version=1, config=config, pipeline_id=pipeline.id)
    db.add(version)
    await db.flush()
    metadata = {**config, "knowledge_base_ids": [str(i) for i in kb_ids] }
    thread = Thread(user_id=current_user.id, project_id=project_id, title=thread_data.title, status="active", 
                    pipeline_id=str(pipeline.id), pipeline_version_id=str(version.id), metadata_=metadata)
    thread.knowledge_bases = kbs
    db.add(thread)
    await db.commit()
    await db.refresh(thread) 
    return APIResponse(success=True, message="Thread created successfully", data=ThreadResponse.model_validate(thread))


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)
    return APIResponse(success=True, message="Thread retrieved successfully", data=ThreadResponse.model_validate(thread))


@router.get("/threads/{thread_id}/pipeline/latest", response_model=APIResponse[PipelineVersionResponse])
async def get_latest_thread_pipeline_version(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_thread(thread_id, current_user.id, db)
    if not thread.pipeline_id:
        raise HTTPException(status_code=404, detail="Thread has no pipeline")

    version = await db.scalar(
        select(PipelineVersion)
        .where(PipelineVersion.pipeline_id == thread.pipeline_id)
        .order_by(PipelineVersion.version.desc())
    )
    if not version:
        raise HTTPException(status_code=404, detail="Pipeline version not found")

    return APIResponse(
        success=True,
        message="Latest pipeline version retrieved successfully",
        data=PipelineVersionResponse.model_validate(version),
    )


@router.patch("/threads/{thread_id}")
async def update_thread(thread_id: UUID, thread_data: ThreadUpdate, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)

    print("Updating thread with data:", thread_data)
    if thread_data.title is not None:
        thread.title = thread_data.title
    if thread_data.knowledge_base_ids is not None:
        kb_ids, kbs = await _get_kbs(thread.project_id, current_user.id, thread_data.knowledge_base_ids, db)
        thread.knowledge_bases = kbs
    # print("Updating thread metadata with config:", thread_data)
    incoming_config = thread_data.config.model_dump() if thread_data.config else None
    if incoming_config is not None:
        current = dict(thread.metadata_ or {})
        latest = await db.scalar(select(PipelineVersion).where(PipelineVersion.pipeline_id == thread.pipeline_id).order_by(PipelineVersion.version.desc()))
        config_changed = incoming_config != (latest.config if latest else {})
        if thread_data.knowledge_base_ids is not None:
            current["knowledge_base_ids"] = [str(i) for i in thread_data.knowledge_base_ids]
        if config_changed:
            # print("Config changed, creating new pipeline version...",current)
            pipeline_id = thread.pipeline_id
            version = PipelineVersion(version=(latest.version + 1 if latest else 1), config=incoming_config, pipeline_id=pipeline_id)
            db.add(version)
            await db.flush()
            thread.pipeline_version_id = version.id
            current.update(incoming_config)
        thread.metadata_ = current
    await db.commit()
    await db.refresh(thread)
    return APIResponse(success=True, message="Thread updated successfully", data=ThreadResponse.model_validate(thread))


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)
    await db.delete(thread)
    await db.commit()
    return APIResponse(success=True, message="Thread deleted successfully", data=None)


@router.get("/threads/{thread_id}/messages", response_model=APIResponse[list[MessageResponse]])
async def get_thread_messages(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_thread(thread_id, current_user.id, db)
    messages = (
        await db.scalars(
            select(Message)
            .where(Message.thread_id == thread_id)
            .order_by(Message.sequence.asc())
        )
    ).all()
    return APIResponse(
        success=True,
        message="Thread messages retrieved successfully",
        data=[MessageResponse.model_validate(message) for message in messages],
    )


@router.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: UUID,
    message_data: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
): 
    
    # 1. Fetch thread
    thread = await db.scalar(
        select(Thread).where(
            Thread.id == thread_id,
            Thread.user_id == current_user.id,
        )
    )

    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # 2. Get current pipeline version
    pipeline_version = await db.scalar(
        select(PipelineVersion).where(
            PipelineVersion.id == thread.pipeline_version_id
        )
    )

    if not pipeline_version:
        raise HTTPException(
            status_code=400,
            detail="Thread has no active pipeline"
        )

    last_sequence = await db.scalar(
    select(func.max(Message.sequence))
    .where(Message.thread_id == thread.id)
    )

    next_sequence = (last_sequence or 0) + 1
    # 3. Create user message
    user_message = Message(thread_id=thread.id,role="human",content=message_data.content,sequence= next_sequence,status="completed",)
    db.add(user_message)
    await db.flush()
    # 4. Create Run
    run = Run(
        thread_id=thread.id,
        input_message_id=user_message.id,
        pipeline_version_id=pipeline_version.id,
        run_type="chat",
        status="running",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # 5. Execute the RAG pipeline inside the SSE stream.
    engine = QueryEngine(db)

    async def event_generator():
        yield {
            "event": "started",
            "data": json.dumps({"run_id": str(run.id), "message_id": str(user_message.id)}),
        }
        async for event in engine.run_stream(
            message=user_message,
            thread=thread,
            pipeline_version=pipeline_version,
            run=run,
        ):
            yield {"event": event["type"], "data": json.dumps(event)}

    return create_sse_response(event_generator())
