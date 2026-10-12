from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "error": {
                "code": exc.code,
                "details": exc.details,
            },
        },
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

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation failed.",
            "error": {
                "code": "VALIDATION_ERROR",
                "details": errors,
            },
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred.",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "details": None,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
