from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.schemas.pipeline import PipelineRequest, PipelineVersionRequest


class PipelineService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def verify_project_ownership(
        self, project_id: UUID, user_id: int
    ) -> Project:
        stmt = select(Project).where(
            Project.id == project_id, Project.user_id == user_id
        )
        result = await self.db.execute(stmt)
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )
        return project

    async def get_project_pipelines(
        self, project_id: UUID, user_id: int
    ) -> list[Pipeline]:
        await self.verify_project_ownership(project_id, user_id)
        stmt = select(Pipeline).where(
            Pipeline.project_id == project_id,
            Pipeline.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_pipeline(
        self, project_id: UUID, user_id: int, data: PipelineRequest
    ) -> Pipeline:
        await self.verify_project_ownership(project_id, user_id)
        new_pipeline = Pipeline(
            name=data.name,
            description=data.description,
            project_id=project_id,
            user_id=user_id,
        )
        self.db.add(new_pipeline)
        await self.db.commit()
        await self.db.refresh(new_pipeline)
        return new_pipeline

    async def get_pipeline(self, pipeline_id: UUID, user_id: int) -> Pipeline:
        pipeline = await self.db.scalar(
            select(Pipeline).where(
                Pipeline.id == pipeline_id, Pipeline.user_id == user_id
            )
        )
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found",
            )
        return pipeline

    async def update_pipeline(
        self, pipeline_id: UUID, user_id: int, data: PipelineRequest
    ) -> Pipeline:
        pipeline = await self.get_pipeline(pipeline_id, user_id)
        pipeline.name = data.name
        pipeline.description = data.description
        await self.db.commit()
        await self.db.refresh(pipeline)
        return pipeline

    async def delete_pipeline(self, pipeline_id: UUID, user_id: int) -> None:
        pipeline = await self.get_pipeline(pipeline_id, user_id)
        await self.db.delete(pipeline)
        await self.db.commit()

    async def get_pipeline_versions(
        self, pipeline_id: UUID, user_id: int
    ) -> list[PipelineVersion]:
        await self.get_pipeline(pipeline_id, user_id)
        stmt = select(PipelineVersion).where(
            PipelineVersion.pipeline_id == pipeline_id
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_pipeline_version(
        self, pipeline_id: UUID, user_id: int, data: PipelineVersionRequest
    ) -> PipelineVersion:
        await self.get_pipeline(pipeline_id, user_id)
        latest = await self.db.scalar(
            select(PipelineVersion)
            .where(PipelineVersion.pipeline_id == pipeline_id)
            .order_by(PipelineVersion.version.desc())
        )
        new_version = PipelineVersion(
            version=data.version or (latest.version + 1 if latest else 1),
            config=data.config,
            pipeline_id=pipeline_id,
        )
        self.db.add(new_version)
        await self.db.commit()
        await self.db.refresh(new_version)
        return new_version

    async def get_pipeline_version(
        self, version_id: UUID, user_id: int
    ) -> PipelineVersion:
        version = await self.db.scalar(
            select(PipelineVersion)
            .join(Pipeline)
            .where(
                PipelineVersion.id == version_id, Pipeline.user_id == user_id
            )
        )
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline version not found",
            )
        return version

    async def update_pipeline_version(
        self, version_id: UUID, user_id: int, data: PipelineVersionRequest
    ) -> PipelineVersion:
        version = await self.get_pipeline_version(version_id, user_id)
        version.config = data.config
        await self.db.commit()
        await self.db.refresh(version)
        return version

    async def delete_pipeline_version(
        self, version_id: UUID, user_id: int
    ) -> None:
        version = await self.get_pipeline_version(version_id, user_id)
        await self.db.delete(version)
        await self.db.commit()
