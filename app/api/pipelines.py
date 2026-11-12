from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineVersionRequest,
    PipelineVersionResponse,
)
from app.services.pipeline_service import PipelineService

router = APIRouter(
    tags=["Pipelines"],
)


@router.get(
    "/projects/{project_id}/pipelines",
    response_model=APIResponse[list[PipelineResponse]],
)
async def get_project_pipelines(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    pipelines = await service.get_project_pipelines(
        project_id=project_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipelines retrieved successfully",
        data=[PipelineResponse.model_validate(p) for p in pipelines],
    )


@router.post(
    "/projects/{project_id}/pipelines",
    response_model=APIResponse[PipelineResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline(
    project_id: UUID,
    data: PipelineRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    new_pipeline = await service.create_pipeline(
        project_id=project_id, user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Pipeline created successfully",
        data=PipelineResponse.model_validate(new_pipeline),
    )


@router.get(
    "/pipelines/{pipeline_id}",
    response_model=APIResponse[PipelineResponse],
)
async def get_pipeline(
    pipeline_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    pipeline = await service.get_pipeline(
        pipeline_id=pipeline_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipeline retrieved successfully",
        data=PipelineResponse.model_validate(pipeline),
    )


@router.patch(
    "/pipelines/{pipeline_id}",
    response_model=APIResponse[PipelineResponse],
)
async def update_pipeline(
    pipeline_id: UUID,
    data: PipelineRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    pipeline = await service.update_pipeline(
        pipeline_id=pipeline_id, user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Pipeline updated successfully",
        data=PipelineResponse.model_validate(pipeline),
    )


@router.delete(
    "/pipelines/{pipeline_id}",
    response_model=APIResponse[None],
)
async def delete_pipeline(
    pipeline_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    await service.delete_pipeline(
        pipeline_id=pipeline_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipeline deleted successfully",
        data=None,
    )


@router.get(
    "/pipelines/{pipeline_id}/versions",
    response_model=APIResponse[list[PipelineVersionResponse]],
)
async def get_pipeline_versions(
    pipeline_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    versions = await service.get_pipeline_versions(
        pipeline_id=pipeline_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipeline versions retrieved successfully",
        data=[PipelineVersionResponse.model_validate(v) for v in versions],
    )


@router.post(
    "/pipelines/{pipeline_id}/versions",
    response_model=APIResponse[PipelineVersionResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline_version(
    pipeline_id: UUID,
    data: PipelineVersionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    new_version = await service.create_pipeline_version(
        pipeline_id=pipeline_id, user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Pipeline version created successfully",
        data=PipelineVersionResponse.model_validate(new_version),
    )


@router.get(
    "/pipeline-versions/{version_id}",
    response_model=APIResponse[PipelineVersionResponse],
)
async def get_pipeline_version(
    version_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    version = await service.get_pipeline_version(
        version_id=version_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipeline version retrieved successfully",
        data=PipelineVersionResponse.model_validate(version),
    )


@router.patch(
    "/pipeline-versions/{version_id}",
    response_model=APIResponse[PipelineVersionResponse],
)
async def update_pipeline_version(
    version_id: UUID,
    data: PipelineVersionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    version = await service.update_pipeline_version(
        version_id=version_id, user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Pipeline version updated successfully",
        data=PipelineVersionResponse.model_validate(version),
    )


@router.delete(
    "/pipeline-versions/{version_id}",
    response_model=APIResponse[None],
)
async def delete_pipeline_version(
    version_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = PipelineService(db)
    await service.delete_pipeline_version(
        version_id=version_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Pipeline version deleted successfully",
        data=None,
    )
