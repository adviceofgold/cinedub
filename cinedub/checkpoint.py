import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from .core import Config, RuntimeMode


class CheckpointManager:
    def __init__(self, config: Config):
        self.config = config
        self._checkpoint_dir = config.checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def _path(self, stage: str) -> str:
        return os.path.join(self._checkpoint_dir, f"{stage}.done")

    def save(self, stage: str, output_files: list[str] = None,
             duration_s: float = 0.0, mode: RuntimeMode = RuntimeMode.GPU_FULL,
             metadata: dict = None):
        if output_files is None:
            output_files = []
        checksums = {}
        for fp in output_files:
            if os.path.isfile(fp):
                checksums[fp] = self._checksum(fp)

        data = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration_s,
            "output_files": output_files,
            "checksums": checksums,
            "runtime_mode": mode.value,
            "metadata": metadata or {},
        }
        with open(self._path(stage), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, stage: str) -> Optional[dict]:
        p = self._path(stage)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return json.load(f)

    def exists(self, stage: str) -> bool:
        return os.path.isfile(self._path(stage))

    def verify(self, stage: str) -> bool:
        data = self.load(stage)
        if not data:
            return False
        for fp in data.get("output_files", []):
            if not os.path.isfile(fp):
                return False
        for fp, expected in data.get("checksums", {}).items():
            if not os.path.isfile(fp):
                return False
            if self._checksum(fp) != expected:
                return False
        return True

    def completed_stages(self) -> list[str]:
        stages = []
        for f in sorted(os.listdir(self._checkpoint_dir)):
            if f.endswith(".done"):
                stages.append(f.replace(".done", ""))
        return stages

    def latest_stage(self) -> Optional[str]:
        stages = self.completed_stages()
        return stages[-1] if stages else None

    def resume_from(self, pipeline_stages: list[str]) -> int:
        completed = set(self.completed_stages())
        for i, s in enumerate(pipeline_stages):
            if s not in completed or not self.verify(s):
                return i
        return len(pipeline_stages)

    def clear(self, stage: str):
        p = self._path(stage)
        if os.path.isfile(p):
            os.remove(p)

    def clear_all(self):
        for s in self.completed_stages():
            self.clear(s)

    @staticmethod
    def _checksum(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
