from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.run import RunDetailsResponse, RunStepResponse
from app.services.run_service import RunService

router = APIRouter(tags=["Runs"])


@router.get("/runs/{run_id}", response_model=APIResponse[RunDetailsResponse])
async def get_run_details(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = RunService(db)
    run = await service.get_run_with_steps(
        run_id=run_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Run details retrieved successfully",
        data=RunDetailsResponse.model_validate(run),
    )


@router.get(
    "/runs/{run_id}/steps",
    response_model=APIResponse[list[RunStepResponse]],
)
async def get_run_steps(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = RunService(db)
    steps = await service.get_run_steps(run_id=run_id, user_id=current_user.id)
    return APIResponse(
        success=True,
        message="Run steps retrieved successfully",
        data=[RunStepResponse.model_validate(step) for step in steps],
    )


@router.get(
    "/messages/{message_id}/run-steps",
    response_model=APIResponse[RunDetailsResponse],
)
async def get_message_run_steps(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = RunService(db)
    run = await service.get_message_run_steps(
        message_id=message_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Message run steps retrieved successfully",
        data=RunDetailsResponse.model_validate(run),
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=APIResponse[RunDetailsResponse],
)
async def cancel_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel an in-progress run and update its status in the database."""
    service = RunService(db)
    run = await service.cancel_run(run_id=run_id, user_id=current_user.id)
    message = (
        f"Run is already {run.status}"
        if run.status in ["completed", "failed"]
        else "Run cancelled successfully"
    )
    return APIResponse(
        success=True,
        message=message,
        data=RunDetailsResponse.model_validate(run),
    )
