from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.models.run import Run
from app.models.thread import Thread
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
        max_attempts = 3 if data.version is None else 1
        for attempt in range(max_attempts):
            latest = await self.db.scalar(
                select(PipelineVersion)
                .where(PipelineVersion.pipeline_id == pipeline_id)
                .order_by(PipelineVersion.version.desc())
            )
            target_version = data.version or (latest.version + 1 if latest else 1)
            new_version = PipelineVersion(
                version=target_version,
                config=data.config,
                pipeline_id=pipeline_id,
            )
            self.db.add(new_version)
            try:
                await self.db.commit()
                await self.db.refresh(new_version)
                return new_version
            except IntegrityError:
                await self.db.rollback()
                if attempt == max_attempts - 1:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Pipeline version {target_version} already exists",
                    )

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

    async def prune_pipeline_versions(
        self, pipeline_id: UUID, user_id: int
    ) -> dict:
        """Prune unreferenced pipeline versions.

        Deletes versions that:
        1. Belong to this pipeline.
        2. Are NOT the latest version of the pipeline.
        3. Are NOT referenced by any active Thread (thread.pipeline_version_id).
        4. Are NOT referenced by any Run (run.pipeline_version_id).
        """
        await self.get_pipeline(pipeline_id, user_id)

        # 1. Identify latest version to ensure it is never deleted
        latest_version_id = await self.db.scalar(
            select(PipelineVersion.id)
            .where(PipelineVersion.pipeline_id == pipeline_id)
            .order_by(PipelineVersion.version.desc())
            .limit(1)
        )

        # Subqueries for referenced version IDs
        referenced_by_threads = select(Thread.pipeline_version_id).where(
            Thread.pipeline_version_id.is_not(None)
        )
        referenced_by_runs = select(Run.pipeline_version_id).where(
            Run.pipeline_version_id.is_not(None)
        )

        stmt = select(PipelineVersion).where(
            PipelineVersion.pipeline_id == pipeline_id,
            PipelineVersion.id.not_in(referenced_by_threads),
            PipelineVersion.id.not_in(referenced_by_runs),
        )

        if latest_version_id is not None:
            stmt = stmt.where(PipelineVersion.id != latest_version_id)

        result = await self.db.scalars(stmt)
        unused_versions = list(result.all())

        deleted_ids: list[UUID] = []
        for version in unused_versions:
            deleted_ids.append(version.id)
            await self.db.delete(version)

        if deleted_ids:
            await self.db.commit()

        return {
            "deleted_count": len(deleted_ids),
            "deleted_version_ids": deleted_ids,
        }

