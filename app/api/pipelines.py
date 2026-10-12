
from fastapi import APIRouter, Depends, HTTPException, status 
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List
from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models import Project , Pipeline , PipelineVersion
from app.schemas.common import APIResponse
from app.schemas.pipeline import PipelineResponse, PipelineRequest, PipelineVersionResponse, PipelineVersionRequest


router = APIRouter(
    prefix="/api",
    tags=["Pipelines"],
)


async def _verify_project_ownership(
    project_id: UUID, user_id: int, db: AsyncSession
) -> Project:
    stmt = select(Project).where(
        Project.id == project_id, Project.user_id == user_id
    )
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return project


@router.get("/projects/{project_id}/pipelines")
async def get_project_pipelines(
    project_id: UUID,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(Pipeline).where(
        Pipeline.project_id == project_id,
        Pipeline.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    pipeline = result.scalars().all()
    return APIResponse(
        success=True,
        message="Pipelines retrieved successfully",
        data=[PipelineResponse.model_validate(item) for item in pipeline],
    )
 
# POST   /projects/{project_id}/pipelines
@router.post(
    "/projects/{project_id}/pipelines",
    response_model=APIResponse[PipelineResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline(
    project_id: UUID,
    data: PipelineRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_project_ownership(project_id, current_user.id, db)

    new_pipeline = Pipeline(
        name=data.name,
        description=data.description,
        project_id=project_id,
        user_id=current_user.id,
    )
    db.add(new_pipeline)
    await db.commit()
    await db.refresh(new_pipeline)
    return APIResponse(
        success=True,
        message="Pipeline created successfully",
        data=PipelineResponse.model_validate(new_pipeline),
    )

# GET    /pipelines/{pipeline_id}
# PATCH  /pipelines/{pipeline_id}

# GET    /pipelines/{pipeline_id}/versions
@router.get("/pipelines/{pipeline_id}/versions")
async def get_pipeline_versions(
    pipeline_id: UUID,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Pipeline).where(
            Pipeline.id == pipeline_id,
            Pipeline.user_id == current_user.id,
        )
    result = await db.execute(stmt)
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found",
        )
    
    stmt = select(PipelineVersion).where(
        PipelineVersion.pipeline_id == pipeline_id,
    )
    result = await db.execute(stmt)
    versions = result.scalars().all()
    return APIResponse(
        success=True,
        message="Pipeline versions retrieved successfully",
        data=[PipelineVersionResponse.model_validate(v) for v in versions],
    ) 

# POST   /pipelines/{pipeline_id}/versions
@router.post(
    "/pipelines/{pipeline_id}/versions",
    response_model=APIResponse[PipelineVersionResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline_version(
    pipeline_id: UUID,
    data: PipelineVersionRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Pipeline).where(
        Pipeline.id == pipeline_id,
        Pipeline.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline not found",
        )

    latest = await db.scalar(select(PipelineVersion).where(PipelineVersion.pipeline_id == pipeline_id).order_by(PipelineVersion.version.desc()))
    new_version = PipelineVersion(
        version=data.version or (latest.version + 1 if latest else 1),
        config=data.config,
        pipeline_id=pipeline_id,
    )
    db.add(new_version)
    await db.commit()
    await db.refresh(new_version)
    return APIResponse(
        success=True,
        message="Pipeline version created successfully",
        data=PipelineVersionResponse.model_validate(new_version),
    )


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    pipeline = await db.scalar(select(Pipeline).where(Pipeline.id == pipeline_id, Pipeline.user_id == current_user.id))
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return APIResponse(success=True, message="Pipeline retrieved successfully", data=PipelineResponse.model_validate(pipeline))


@router.patch("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: UUID, data: PipelineRequest, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    pipeline = await db.scalar(select(Pipeline).where(Pipeline.id == pipeline_id, Pipeline.user_id == current_user.id))
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    pipeline.name, pipeline.description = data.name, data.description
    await db.commit()
    await db.refresh(pipeline)
    return APIResponse(success=True, message="Pipeline updated successfully", data=PipelineResponse.model_validate(pipeline))


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    pipeline = await db.scalar(select(Pipeline).where(Pipeline.id == pipeline_id, Pipeline.user_id == current_user.id))
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    await db.delete(pipeline)
    await db.commit()
    return APIResponse(success=True, message="Pipeline deleted successfully", data=None)


@router.get("/pipeline-versions/{version_id}")
async def get_pipeline_version(version_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    version = await db.scalar(select(PipelineVersion).join(Pipeline).where(PipelineVersion.id == version_id, Pipeline.user_id == current_user.id))
    if not version:
        raise HTTPException(status_code=404, detail="Pipeline version not found")
    return APIResponse(success=True, message="Pipeline version retrieved successfully", data=PipelineVersionResponse.model_validate(version))


@router.patch("/pipeline-versions/{version_id}")
async def update_pipeline_version(version_id: UUID, data: PipelineVersionRequest, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    version = await db.scalar(select(PipelineVersion).join(Pipeline).where(PipelineVersion.id == version_id, Pipeline.user_id == current_user.id))
    if not version:
        raise HTTPException(status_code=404, detail="Pipeline version not found")
    version.config = data.config
    await db.commit()
    await db.refresh(version)
    return APIResponse(success=True, message="Pipeline version updated successfully", data=PipelineVersionResponse.model_validate(version))


@router.delete("/pipeline-versions/{version_id}")
async def delete_pipeline_version(version_id: UUID, current_user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    version = await db.scalar(select(PipelineVersion).join(Pipeline).where(PipelineVersion.id == version_id, Pipeline.user_id == current_user.id))
    if not version:
        raise HTTPException(status_code=404, detail="Pipeline version not found")
    await db.delete(version)
    await db.commit()
    return APIResponse(success=True, message="Pipeline version deleted successfully", data=None)

# GET    /pipeline-versions/{version_id}

# exmape config:
# {
#   "retrieval": {
#     "strategy": "vector",
#     "top_k": 8,
#     "score_threshold": 0.35,
#     "reranker": null
#   },

#   "query": {
#     "rewrite": true,
#     "multi_query": false
#   },

#   "generation": {
#     "provider": "groq",
#     "model": "...",
#     "temperature": 0.2
#   },

#   "context": {
#     "max_tokens": 6000,
#     "include_citations": true
#   }
# }
