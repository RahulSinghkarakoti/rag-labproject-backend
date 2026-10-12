from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    """
    Base application exception.

    All custom application exceptions should inherit from this.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_SERVER_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ):
        self.message = message or self.message
        self.details = details

        if status_code is not None:
            self.status_code = status_code

        if code is not None:
            self.code = code

        super().__init__(self.message)


# ---------------------------------------------------------
# Authentication / Authorization
# ---------------------------------------------------------

class InvalidTokenError(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "INVALID_TOKEN"
    message = "Invalid or expired access token."


class AuthenticationError(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "AUTHENTICATION_REQUIRED"
    message = "Authentication is required."


class PermissionDeniedError(AppException):
    status_code = status.HTTP_403_FORBIDDEN
    code = "PERMISSION_DENIED"
    message = "You do not have permission to perform this action."


# ---------------------------------------------------------
# Resources
# ---------------------------------------------------------

class ResourceNotFoundError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    code = "RESOURCE_NOT_FOUND"
    message = "The requested resource was not found."


class ResourceAlreadyExistsError(AppException):
    status_code = status.HTTP_409_CONFLICT
    code = "RESOURCE_ALREADY_EXISTS"
    message = "The resource already exists."


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

class ValidationError(AppException):
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
    code = "VALIDATION_ERROR"
    message = "Validation failed."


# ---------------------------------------------------------
# AI Model Provider Errors
# ---------------------------------------------------------

class ModelProviderError(AppException):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "MODEL_PROVIDER_ERROR"
    message = "An error occurred with the AI model provider."

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        error_type: str = "provider_error",
        details: Any | None = None,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
    ):
        clean_msg = message or f"AI model provider '{provider or 'unknown'}' encountered an error."
        err_details = details or {
            "provider": provider,
            "model": model,
            "error_type": error_type,
            "message": clean_msg,
        }
        super().__init__(
            message=clean_msg,
            details=err_details,
            status_code=status_code,
            code="MODEL_PROVIDER_ERROR",
        )


def sanitize_model_error(exc: Exception, provider: str = "unknown", model: str = "unknown") -> ModelProviderError:
    """
    Safely sanitizes exceptions from AI model providers (Groq, Gemini, OpenAI, etc.),
    redacting API keys, bearer tokens, or internal URLs, and classifying the error
    into clean, user-friendly error messages for the frontend.
    """
    import re

    raw_str = str(exc)
    # Mask OpenAI, Google, Groq, or Bearer tokens
    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{15,}", "[REDACTED_API_KEY]", raw_str)
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{25,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"gsk_[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"Bearer\s+[a-zA-Z0-9\-_.]+", "Bearer [REDACTED]", sanitized)

    lower = sanitized.lower()
    exc_type = type(exc).__name__.lower()

    if any(k in lower or k in exc_type for k in ["auth", "invalid api key", "unauthorized", "invalid_api_key", "permission denied", "forbidden", "api_key"]):
        return ModelProviderError(
            message=f"Authentication failed for {provider}. Please verify your API key configuration.",
            provider=provider,
            model=model,
            error_type="authentication_failed",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    if any(k in lower or k in exc_type for k in ["rate limit", "ratelimit", "too many requests", "429", "resourceexhausted", "resource_exhausted"]):
        return ModelProviderError(
            message=f"Rate limit or quota reached for {provider} model '{model}'. Please wait before retrying.",
            provider=provider,
            model=model,
            error_type="rate_limit_exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    if any(k in lower or k in exc_type for k in ["quota", "insufficient_quota", "billing", "credit"]):
        return ModelProviderError(
            message=f"Account quota or credits exceeded for provider '{provider}'.",
            provider=provider,
            model=model,
            error_type="quota_exceeded",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    if any(k in lower or k in exc_type for k in ["not found", "model_not_found", "does not exist", "404"]):
        return ModelProviderError(
            message=f"Model '{model}' is not recognized or not available on provider '{provider}'.",
            provider=provider,
            model=model,
            error_type="model_not_found",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if any(k in lower or k in exc_type for k in ["context length", "maximum context", "token limit", "context_length_exceeded", "prompt is too long"]):
        return ModelProviderError(
            message=f"The input exceeded the maximum token/context window for model '{model}'.",
            provider=provider,
            model=model,
            error_type="context_length_exceeded",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if any(k in lower or k in exc_type for k in ["timeout", "timed out", "connection", "connecterror", "service unavailable", "503"]):
        return ModelProviderError(
            message=f"Network timeout or service unavailable connecting to provider '{provider}'.",
            provider=provider,
            model=model,
            error_type="service_unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # General provider fallback with sanitized message snippet
    # Truncate sanitized string to avoid long trace dumps
    safe_snippet = sanitized.strip().split("\n")[0]
    if len(safe_snippet) > 200:
        safe_snippet = safe_snippet[:197] + "..."

    return ModelProviderError(
        message=f"Provider '{provider}' error: {safe_snippet}",
        provider=provider,
        model=model,
        error_type="provider_error",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )


# ---------------------------------------------------------
# Generic application error
# ---------------------------------------------------------

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