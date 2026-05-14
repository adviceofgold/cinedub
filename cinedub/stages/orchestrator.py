import time

from ..core import Config, PipelineContext, StageResult, RuntimeMode, ModelManager
from ..checkpoint import CheckpointManager
from ..logging import CineDubLogger
from .bootstrap import BootstrapStage
from .download import DownloadStage
from .normalize import NormalizeStage
from .separation import SeparationStage
from .diarization import DiarizationStage
from .transcription import TranscriptionStage
from .alignment import AlignmentStage
from .translation import TranslationStage
from .speaker_db import SpeakerDatabaseStage
from .cloning import VoiceCloningStage
from .sync import AudioSyncStage
from .subtitles import SubtitleStage
from .render import RenderStage
from .lipsync import LipsyncStage


class PipelineOrchestrator:
    PIPELINE_STAGES = [
        "bootstrap",
        "download",
        "normalize",
        "separation",
        "diarization",
        "transcription",
        "alignment",
        "translation",
        "speaker_db",
        "voice_cloning",
        "audio_sync",
        "subtitles",
        "render",
        "lipsync",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.ctx = PipelineContext(config=config)
        self.checkpoint = CheckpointManager(config)
        self.log = CineDubLogger.get(config.log_dir)

        self._stages = {
            "bootstrap": BootstrapStage(),
            "download": DownloadStage(),
            "normalize": NormalizeStage(),
            "separation": SeparationStage(),
            "diarization": DiarizationStage(),
            "transcription": TranscriptionStage(),
            "alignment": AlignmentStage(),
            "translation": TranslationStage(),
            "speaker_db": SpeakerDatabaseStage(),
            "voice_cloning": VoiceCloningStage(),
            "audio_sync": AudioSyncStage(),
            "subtitles": SubtitleStage(),
            "render": RenderStage(),
            "lipsync": LipsyncStage(),
        }

    def run(self) -> list[StageResult]:
        mode = self.config.mode
        self.log.info(f"Pipeline starting (mode={mode.value})")

        if mode == RuntimeMode.RECOVERY:
            start_idx = self.checkpoint.resume_from(self.PIPELINE_STAGES)
            if start_idx >= len(self.PIPELINE_STAGES):
                self.log.info("All stages already completed")
                return []
            self.log.info(f"Resuming from stage index {start_idx}: "
                          f"{self.PIPELINE_STAGES[start_idx]}")
        else:
            start_idx = 0
            self.checkpoint.clear_all()

        results = []
        for i in range(start_idx, len(self.PIPELINE_STAGES)):
            name = self.PIPELINE_STAGES[i]
            stage = self._stages[name]

            if not stage.validate(self.ctx):
                self.log.info(f"Skipping stage '{name}' (validation failed)")
                results.append(StageResult(True, name,
                                           message="Skipped by validation"))
                continue

            self.log.info(f"Executing stage: {name}")
            self._log_stage_progress(name, i)

            result = self._execute_with_retry(stage)
            results.append(result)

            if result.success:
                self.checkpoint.save(
                    name,
                    output_files=result.output_files,
                    duration_s=result.metrics.get("duration_s", 0),
                    mode=mode,
                    metadata=result.metrics,
                )
                self.log.info(f"Stage '{name}' completed in "
                              f"{result.metrics.get('duration_s', 0):.1f}s"
                              if "duration_s" in result.metrics
                              else f"Stage '{name}' completed")
            else:
                self.log.error(f"Stage '{name}' FAILED: {result.message}")
                if mode == RuntimeMode.RECOVERY:
                    self.log.info("Recovery mode: continuing despite failure")
                else:
                    break

        return results

    def _execute_with_retry(self, stage, max_retries=3):
        import torch

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                result = stage.execute(self.ctx)
                result.metrics["duration_s"] = time.time() - t0
                return result

            except torch.cuda.OutOfMemoryError:
                self.log.warn(f"CUDA OOM on '{stage.name}' (attempt "
                              f"{attempt+1}/{max_retries})")
                ModelManager.get().unload_all()
                if attempt < max_retries - 1:
                    continue
                return StageResult(False, stage.name,
                                   message="CUDA OOM after retries")

            except Exception as e:
                self.log.exception(f"Error on '{stage.name}': {e}")
                if attempt < max_retries - 1:
                    continue
                return StageResult(False, stage.name,
                                   message=str(e)[:200])

        return StageResult(False, stage.name, message="Max retries exceeded")

    def _log_stage_progress(self, name: str, idx: int):
        total = len(self.PIPELINE_STAGES)
        pct = (idx / total) * 100
        self.log.info(f"[{idx+1}/{total}] ({pct:.0f}%) - {name}")
