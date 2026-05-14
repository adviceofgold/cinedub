import os
import shutil

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class CleanupStage(Stage):
    def __init__(self):
        super().__init__("cleanup")

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting cleanup")

        config = ctx.config
        cleaned = {"temp": 0, "cache": 0}

        temp_dir = os.path.join(config.work_dir, "temp")
        if os.path.isdir(temp_dir):
            cleaned["temp"] = self._rmtree_size(temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
            log.info(f"Cleaned temp: {cleaned['temp']:.1f} MB")

        cache_dir = config.cache_dir
        if os.path.isdir(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if item.startswith(".") or item.endswith(".lock"):
                    if os.path.isfile(item_path):
                        cleaned["cache"] += os.path.getsize(item_path)
                        os.remove(item_path)
            log.info(f"Cleaned cache locks: {cleaned['cache']:.1f} MB")

        log.info("Cleanup complete")
        return StageResult(True, self.name,
                           metrics={f"cleaned_{k}_mb": round(v / 1e6, 1)
                                    for k, v in cleaned.items()})

    @staticmethod
    def _rmtree_size(path: str) -> float:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    continue
        return total
