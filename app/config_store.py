from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock

import orjson

from app.schemas import ConfigPatch, ServerConfig


class ConfigStore:
    def __init__(self, path: Path):
        self._path = path
        self._lock = RLock()
        self._config = self._load()

    def _load(self) -> ServerConfig:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            cfg = ServerConfig()
            self._write(cfg)
            return cfg
        data = orjson.loads(self._path.read_bytes())
        return ServerConfig.model_validate(data)

    def _write(self, cfg: ServerConfig) -> None:
        """原子性写入配置文件：先写临时文件，再重命名替换。"""
        data = orjson.dumps(cfg.model_dump(), option=orjson.OPT_INDENT_2)
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with open(fd, "wb") as f:
                f.write(data)
            Path(tmp_path).replace(self._path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def get(self) -> ServerConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def update(self, patch: ConfigPatch) -> ServerConfig:
        with self._lock:
            cfg = self._config.model_copy(deep=True)
            updates = patch.model_dump(exclude_none=True)

            if "generation_defaults" in updates:
                gen_updates = updates.pop("generation_defaults")
                current_gen = cfg.generation_defaults.model_dump()
                current_gen.update({k: v for k, v in gen_updates.items() if v is not None})
                updates["generation_defaults"] = current_gen

            cfg_data = cfg.model_dump()
            cfg_data.update(updates)
            cfg = ServerConfig.model_validate(cfg_data)

            if cfg.default_model not in cfg.model_mapping:
                raise ValueError("default_model must exist in model_mapping")

            self._write(cfg)
            self._config = cfg
            return cfg.model_copy(deep=True)
