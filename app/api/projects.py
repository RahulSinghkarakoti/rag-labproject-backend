from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models.project import Project
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.project import ProjectRequest, ProjectResponse

router = APIRouter(
    prefix="/api",
    tags=["Projects"],
)


@router.get("/projects", response_model=APIResponse[List[ProjectResponse]])
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.user_id == current_user.id)
    )
    projects = result.scalars().all()
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
    new_project = Project(
        name=data.name,
        description=data.description,
        user_id=current_user.id,
    )
    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)
    return APIResponse(
        success=True,
        message="Project created successfully",
        data=ProjectResponse.model_validate(new_project),
    )


@router.patch("/projects/{project_id}", response_model=APIResponse[ProjectResponse])
async def update_project(
    project_id: UUID,
    data: ProjectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(
            Project.user_id == current_user.id,
            Project.id == project_id,
        )
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(project, key, value)
    await db.commit()
    await db.refresh(project)
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
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.user_id == current_user.id,
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    await db.delete(project)
    await db.commit()
    return APIResponse(
        success=True,
        message="Project deleted successfully",
        data=None,
    )