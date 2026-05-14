import os
import subprocess
import sys

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode
from ..logging import CineDubLogger


class SeparationStage(Stage):
    def __init__(self):
        super().__init__("separation", depends_on=["normalize"])

    def validate(self, ctx: PipelineContext) -> bool:
        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            return False
        return True

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()

        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            log.info("CPU_SAFE mode: skipping Demucs separation, using raw audio")
            out_dir = ctx.config.subdir("separated_audio")
            # copy raw audio as "vocals"
            vocals = os.path.join(out_dir, "vocals.wav")
            subprocess.run(["cp" if sys.platform != "win32" else "copy",
                           ctx.audio_path, vocals],
                          capture_output=True, text=True, timeout=60)
            ctx.metadata["vocals_path"] = vocals
            return StageResult(True, self.name,
                               message="CPU mode: raw audio used as vocals",
                               output_files=[vocals])

        log.info("Running Demucs HT vocal separation")
        out_dir = ctx.config.subdir("separated_audio")
        os.makedirs(out_dir, exist_ok=True)

        demucs_model = "htdemucs"
        if ctx.config.mode == RuntimeMode.GPU_BALANCED:
            demucs_model = "htdemucs_ft"

        result = subprocess.run([
            sys.executable, "-m", "demucs",
            "-n", demucs_model,
            "-o", out_dir,
            "--two-stems", "vocals",
            ctx.audio_path,
        ], capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            log.error(f"Demucs failed: {result.stderr[:500]}")
            return StageResult(False, self.name,
                               message=f"Separation failed: {result.stderr[:200]}")

        base = os.path.splitext(os.path.basename(ctx.audio_path))[0]
        sep_dir = os.path.join(out_dir, demucs_model, base)
        vocals = os.path.join(sep_dir, "vocals.wav")
        no_vocals = os.path.join(sep_dir, "no_vocals.wav")

        if not os.path.isfile(vocals):
            return StageResult(False, self.name,
                               message=f"Vocals not found at {vocals}")

        ctx.metadata["vocals_path"] = vocals
        ctx.metadata["no_vocals_path"] = no_vocals
        log.info(f"Separation done: {vocals}")
        return StageResult(True, self.name,
                           output_files=[vocals, no_vocals])
