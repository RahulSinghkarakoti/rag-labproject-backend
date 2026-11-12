from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.project import ProjectRequest, ProjectResponse
from app.services.project_service import ProjectService

router = APIRouter(
    tags=["Projects"],
)


@router.get("/projects", response_model=APIResponse[list[ProjectResponse]])
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ProjectService(db)
    projects = await service.get_projects(user_id=current_user.id)
    return APIResponse(
        success=True,
        message="Projects retrieved successfully",
        data=[ProjectResponse.model_validate(p) for p in projects],
    )


@router.post(
    "/projects",
    response_model=APIResponse[ProjectResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    data: ProjectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ProjectService(db)
    new_project = await service.create_project(
        user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Project created successfully",
        data=ProjectResponse.model_validate(new_project),
    )


@router.get("/projects/{project_id}", response_model=APIResponse[ProjectResponse])
async def get_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ProjectService(db)
    project = await service.get_project(
        project_id=project_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Project retrieved successfully",
        data=ProjectResponse.model_validate(project),
    )


@router.patch("/projects/{project_id}", response_model=APIResponse[ProjectResponse])
async def update_project(
    project_id: UUID,
    data: ProjectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ProjectService(db)
    project = await service.update_project(
        project_id=project_id, user_id=current_user.id, data=data
    )
    return APIResponse(
        success=True,
        message="Project updated successfully",
        data=ProjectResponse.model_validate(project),
    )


@router.delete("/projects/{project_id}", response_model=APIResponse[None])
async def delete_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ProjectService(db)
    await service.delete_project(
        project_id=project_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Project deleted successfully",
        data=None,
    )
