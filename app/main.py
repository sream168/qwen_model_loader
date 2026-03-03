from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import routes
from app.config_store import ConfigStore
from app.model_manager import ModelManager
from app.service import ChatService

logger = logging.getLogger(__name__)


def create_app(config_path: str = "config/config.json") -> FastAPI:
    config_store = ConfigStore(Path(config_path))
    model_manager = ModelManager(config_store)
    service = ChatService(config_store, model_manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """应用生命周期管理：退出时停止所有引擎子进程。"""
        yield
        logger.info("应用关闭，正在释放引擎资源...")
        model_manager.shutdown()

    app = FastAPI(
        title="Qwen Offline Loader",
        version="0.1.0",
        lifespan=lifespan,
    )

    def _get_service() -> ChatService:
        return service

    app.dependency_overrides[routes.get_service] = _get_service

    @app.exception_handler(FileNotFoundError)
    def _file_not_found(_, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def _value_err(_, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    def _runtime_err(_, exc: RuntimeError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(routes.router)
    return app


app = create_app()
