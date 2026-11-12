from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.schemas.project import ProjectRequest


class ProjectService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def verify_project_ownership(self, project_id: UUID, user_id: int) -> Project:
        stmt = select(Project).where(
            Project.id == project_id,
            Project.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )
        return project

    async def get_projects(self, user_id: int) -> list[Project]:
        result = await self.db.execute(
            select(Project).where(Project.user_id == user_id)
        )
        return list(result.scalars().all())

    async def create_project(self, user_id: int, data: ProjectRequest) -> Project:
        new_project = Project(
            name=data.name,
            description=data.description,
            user_id=user_id,
        )
        self.db.add(new_project)
        await self.db.commit()
        await self.db.refresh(new_project)
        return new_project

    async def get_project(self, project_id: UUID, user_id: int) -> Project:
        return await self.verify_project_ownership(project_id, user_id)

    async def update_project(
        self, project_id: UUID, user_id: int, data: ProjectRequest
    ) -> Project:
        project = await self.verify_project_ownership(project_id, user_id)
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(project, key, value)
        await self.db.commit()
        await self.db.refresh(project)
        return project

    async def delete_project(self, project_id: UUID, user_id: int) -> None:
        project = await self.verify_project_ownership(project_id, user_id)
        await self.db.delete(project)
        await self.db.commit()
