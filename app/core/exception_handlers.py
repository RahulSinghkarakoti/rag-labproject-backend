import http

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.schemas.common import APIError, APIErrorResponse


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    payload = APIErrorResponse(
        success=False,
        message=exc.message,
        error=APIError(
            code=exc.code,
            details=exc.details,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(),
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    try:
        phrase = http.HTTPStatus(exc.status_code).name
    except ValueError:
        phrase = f"HTTP_{exc.status_code}"

    message = exc.detail if isinstance(exc.detail, str) else "Request error occurred."
    details = exc.detail if not isinstance(exc.detail, str) else None

    payload = APIErrorResponse(
        success=False,
        message=message,
        error=APIError(
            code=phrase,
            details=details,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = []

    for error in exc.errors():
        location = error.get("loc", [])

        field = ".".join(
            str(item)
            for item in location
            if item not in ("body", "query", "path")
        )

        errors.append(
            {
                "field": field or None,
                "message": error.get("msg"),
                "type": error.get("type"),
            }
        )

    payload = APIErrorResponse(
        success=False,
        message="Validation failed.",
        error=APIError(
            code="VALIDATION_ERROR",
            details=errors,
        ),
    )
    return JSONResponse(
        status_code=422,
        content=payload.model_dump(),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    payload = APIErrorResponse(
        success=False,
        message="An unexpected error occurred.",
        error=APIError(
            code="INTERNAL_SERVER_ERROR",
            details=None,
        ),
    )
    return JSONResponse(
        status_code=500,
        content=payload.model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
