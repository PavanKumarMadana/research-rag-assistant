"""
Main Application Module.

FastAPI application entry point with middleware, routers, and startup/shutdown events.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app.core.config import settings
from backend.app.core.database import init_db, close_db
from backend.app.core.logging import setup_logging
from backend.app.routes import document_routes, rag_routes, ml_routes, analytics_routes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup and shutdown events.

    Args:
        app: FastAPI application instance.
    """
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")

    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    await close_db()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions.

    Args:
        request: The request that caused the exception.
        exc: The exception instance.

    Returns:
        JSONResponse: Error response.
    """
    logger.error(f"Unhandled exception: {exc} | Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "path": request.url.path,
        },
    )


# Include routers
app.include_router(document_routes.router)
app.include_router(rag_routes.router)
app.include_router(ml_routes.router)
app.include_router(analytics_routes.router)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend-assets")


@app.get("/")
async def root():
    """Root endpoint.

    Returns:
        dict: API information.
    """
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/analytics/health",
    }


@app.get("/api")
async def api_info():
    """API information endpoint.

    Returns:
        dict: Available API endpoints.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "documents": {
                "upload": "POST /api/documents/upload",
                "list": "GET /api/documents/",
                "get": "GET /api/documents/{id}",
                "delete": "DELETE /api/documents/{id}",
                "reprocess": "POST /api/documents/reprocess/{id}",
                "search": "GET /api/documents/search/{query}",
            },
            "rag": {
                "ask": "POST /api/rag/ask",
                "summarize": "POST /api/rag/summarize",
                "compare": "POST /api/rag/compare",
                "chat": "POST /api/rag/chat",
                "conversations": "GET /api/rag/conversations/{session_id}",
            },
            "ml": {
                "classify": "POST /api/ml/classify/{document_id}",
                "classify_text": "POST /api/ml/classify-text",
                "model_info": "GET /api/ml/model-info",
                "train": "POST /api/ml/train",
                "evaluate": "GET /api/ml/evaluate",
                "classify_all": "POST /api/ml/classify-all",
            },
            "analytics": {
                "overview": "GET /api/analytics/overview",
                "full": "GET /api/analytics/full",
                "health": "GET /api/analytics/health",
            },
        },
    }


@app.get("/{full_path:path}", include_in_schema=False)
async def frontend_fallback(full_path: str):
    """Serve the React frontend for non-API paths when bundled in Docker."""
    if (
        FRONTEND_INDEX.exists()
        and not full_path.startswith(("api", "docs", "redoc", "openapi.json", "assets"))
    ):
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse(status_code=404, content={"detail": "Not found"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
