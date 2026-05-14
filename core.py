import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch


class RuntimeMode(Enum):
    GPU_FULL = "GPU_FULL"
    GPU_BALANCED = "GPU_BALANCED"
    CPU_SAFE = "CPU_SAFE"
    RECOVERY = "RECOVERY"


@dataclass
class Config:
    work_dir: str = "/content/drive/MyDrive/cinedub"
    project_name: str = ""
    target_language: str = "es"
    enable_lipsync: bool = False
    quality: str = "balanced"
    subtitle_style: str = "netflix"
    source_url: str = ""
    mode: RuntimeMode = RuntimeMode.GPU_FULL

    @property
    def project_dir(self) -> str:
        return os.path.join(self.work_dir, "projects", self.project_name)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.work_dir, "checkpoints")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.work_dir, "models")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.work_dir, "cache")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.work_dir, "logs")

    def ensure_dirs(self):
        for d in [self.project_dir, self.checkpoint_dir, self.model_dir,
                  self.cache_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    def subdir(self, name: str) -> str:
        p = os.path.join(self.project_dir, name)
        os.makedirs(p, exist_ok=True)
        return p


@dataclass
class StageResult:
    success: bool
    stage_name: str
    message: str = ""
    output_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    config: Config
    video_path: str = ""
    audio_path: str = ""
    sample_rate: int = 16000
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def stage_dir(self, stage_name: str) -> str:
        return self.config.subdir(stage_name)


class Stage(ABC):
    def __init__(self, name: str, depends_on: Optional[list[str]] = None):
        self.name = name
        self.depends_on = depends_on or []

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> StageResult:
        pass

    def validate(self, ctx: PipelineContext) -> bool:
        return True


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _registry: dict[str, Any] = {}

    def __init__(self, cache_dir: str = "/content/drive/MyDrive/cinedub/cache"):
        self.cache_dir = cache_dir
        os.environ.setdefault("TORCH_HOME", cache_dir)
        os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))

    @classmethod
    def get(cls, cache_dir: Optional[str] = None) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(cache_dir or "/content/drive/MyDrive/cinedub/cache")
        return cls._instance

    def load(self, key: str, loader: Callable, **kwargs) -> Any:
        if key in self._registry:
            return self._registry[key]
        model = loader(**kwargs)
        self._registry[key] = model
        return model

    def unload(self, key: str):
        if key in self._registry:
            obj = self._registry.pop(key)
            del obj
            self._cleanup()

    def unload_all(self):
        keys = list(self._registry.keys())
        for k in keys:
            self.unload(k)
        self._cleanup()

    def get_device_map(self, model_size_gb: float) -> dict[str, str]:
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if model_size_gb * 1.5 > vram:
            return {"device_map": "auto", "torch_dtype": torch.float16}
        return {"device": "cuda", "torch_dtype": torch.float16}

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def resolve_mode() -> RuntimeMode:
        if not torch.cuda.is_available():
            return RuntimeMode.CPU_SAFE
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram < 10:
            return RuntimeMode.GPU_BALANCED
        return RuntimeMode.GPU_FULL

    @staticmethod
    def vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9
