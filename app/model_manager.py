from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from app.config_store import ConfigStore
from app.engine.base import BaseEngine

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config_store: ConfigStore):
        self._config_store = config_store
        self._cache: OrderedDict[str, BaseEngine] = OrderedDict()
        self._lock = RLock()
        # 正在创建中的模型名称集合，防止并发重复启动子进程
        self._creating: set[str] = set()

    def _resolve_path(self, model_name: str) -> str:
        """解析模型名称为磁盘上的绝对路径。"""
        cfg = self._config_store.get()
        filename = cfg.model_mapping.get(model_name)
        if not filename:
            raise ValueError(f"unknown model: {model_name}")

        path = Path(filename)
        if not path.is_absolute():
            path = Path(cfg.model_dir) / path

        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return str(path)

    def _create_engine(self, model_path: str) -> BaseEngine:
        """根据配置中的 engine_type 创建对应的引擎实例。"""
        cfg = self._config_store.get()

        if cfg.engine_type == "llama_server":
            from app.engine.llama_server_engine import LlamaServerEngine
            return LlamaServerEngine(
                model_path=model_path,
                llama_server_path=cfg.llama_server_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                startup_timeout=cfg.startup_timeout,
                mmproj_path=cfg.mmproj_path,
            )
        else:
            from app.engine.llama_engine import LlamaCppEngine
            return LlamaCppEngine(
                model_path=model_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
            )

    def get_engine(self, model_name: str) -> BaseEngine:
        """获取指定模型的引擎实例，使用 LRU 缓存策略。

        内存安全策略：
        1. 先淘汰旧引擎释放内存，再创建新引擎
        2. 通过 _creating 集合防止并发请求重复启动子进程
        """
        with self._lock:
            # 快速路径：缓存命中
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
                return self._cache[model_name]

            # 防止并发创建同一模型：其他线程正在创建则等待
            if model_name in self._creating:
                raise RuntimeError(
                    f"模型 {model_name} 正在加载中，请稍后重试"
                )
            self._creating.add(model_name)

        try:
            # 先校验目标模型路径，避免目标模型无效时错误淘汰已有缓存。
            model_path = self._resolve_path(model_name)
        except Exception:
            with self._lock:
                self._creating.discard(model_name)
            raise

        with self._lock:
            # 先淘汰旧引擎释放内存，再创建新引擎（内存安全优先）
            cfg = self._config_store.get()
            evicted: list[tuple[str, BaseEngine]] = []
            while len(self._cache) >= cfg.max_loaded_models:
                evicted_name, evicted_engine = self._cache.popitem(last=False)
                evicted.append((evicted_name, evicted_engine))

        # 锁外停止被淘汰的引擎，先释放内存
        for evicted_name, evicted_engine in evicted:
            logger.info("缓存淘汰模型: %s，正在停止引擎以释放内存", evicted_name)
            try:
                evicted_engine.stop()
            except Exception:
                logger.exception("停止被淘汰引擎 %s 时出错", evicted_name)

        # 锁外创建新引擎（耗时操作）
        engine: BaseEngine | None = None
        try:
            engine = self._create_engine(model_path)
        except Exception:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    logger.exception("清理失败的引擎时出错")
            # 创建失败，移除创建标记
            with self._lock:
                self._creating.discard(model_name)
            raise

        # 重新获取锁，将新引擎放入缓存
        with self._lock:
            self._creating.discard(model_name)
            # 双重检查：其他线程可能已加载同一模型
            if model_name in self._cache:
                engine.stop()
                self._cache.move_to_end(model_name)
                return self._cache[model_name]
            self._cache[model_name] = engine
            return engine

    def shutdown(self) -> None:
        """停止所有已加载引擎，用于应用退出时的资源清理。"""
        with self._lock:
            engines = list(self._cache.items())
            self._cache.clear()

        for name, engine in engines:
            logger.info("正在关闭引擎: %s", name)
            try:
                engine.stop()
            except Exception:
                logger.exception("关闭引擎 %s 时出错", name)
