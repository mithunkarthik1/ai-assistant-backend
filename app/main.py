"""
FastAPI application entry point.
Configures logging, lifespan events, CORS middleware, and global exception handlers.
"""
import http
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.chat_bot.api import router as chat_router
from app.config import settings
from app.database import init_db

# Configure unified application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


def get_error_code(status_code: int) -> str:
    """Translates an HTTP status code into its standard descriptive string."""
    try:
        return http.HTTPStatus(status_code).name
    except ValueError:
        return "ERROR"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Initializes PostgreSQL tables and automatically indexes the company policy handbook.
    """
    logger.info("Initializing %s...", settings.app_name)
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e, exc_info=True)

    try:
        from app.rag.indexing import index_company_policy
        indexed_count = await index_company_policy()
        logger.info("Policy vector indexing completed (%d chunks ready).", indexed_count)
    except Exception as e:
        logger.error("Policy indexing failed during startup: %s", e, exc_info=True)

    logger.info("%s startup completed and ready for requests.", settings.app_name)
    yield
    logger.info("%s shutting down.", settings.app_name)


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# CORS configuration for frontend web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(HTTPException)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException | StarletteHTTPException
) -> JSONResponse:
    code = get_error_code(exc.status_code)
    message = exc.detail
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code", code)
        message = exc.detail.get("message", str(exc.detail))

    logger.warning("HTTP %d error on %s: %s", exc.status_code, request.url.path, message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "status_code": exc.status_code,
                "message": message,
            },
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    err_msgs = []
    for err in exc.errors():
        loc = " -> ".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        err_msgs.append(f"{loc}: {msg}" if loc else msg)

    formatted_msg = "; ".join(err_msgs) if err_msgs else "Validation error occurred."
    logger.warning("Validation failure on %s: %s", request.url.path, formatted_msg)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "UNPROCESSABLE_ENTITY",
                "status_code": 422,
                "message": formatted_msg,
            },
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled internal server exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "status_code": 500,
                "message": "An unexpected internal server error occurred.",
            },
        },
    )


# Register API routers
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health", summary="Health check probe")
async def health_check() -> dict[str, str]:
    """Simple service health probe."""
    return {"status": "healthy", "app": settings.app_name}
