import os
import subprocess

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class NormalizeStage(Stage):
    def __init__(self):
        super().__init__("normalize", depends_on=["download"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        if not os.path.isfile(ctx.video_path):
            return StageResult(False, self.name, message="Video not found")

        out_dir = ctx.config.subdir("input")
        base = os.path.splitext(os.path.basename(ctx.video_path))[0]

        # Normalize video: force H264, fixed FPS, AAC audio
        norm_video = os.path.join(out_dir, f"{base}_norm.mp4")
        log.info(f"Normalizing video: {ctx.video_path} -> {norm_video}")

        result = subprocess.run([
            "ffmpeg", "-i", ctx.video_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-r", "25",
            "-c:a", "aac", "-b:a", "128k",
            "-y", norm_video
        ], capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            log.error(f"Video normalization failed: {result.stderr[:500]}")
            return StageResult(False, self.name,
                               message=f"Normalize failed: {result.stderr[:200]}")

        # Extract 16kHz WAV for AI processing
        audio_path = os.path.join(out_dir, f"{base}_audio.wav")
        result = subprocess.run([
            "ffmpeg", "-i", norm_video,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ], capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            log.error(f"Audio extraction failed: {result.stderr[:500]}")
            return StageResult(False, self.name,
                               message=f"Audio extraction failed: {result.stderr[:200]}")

        ctx.video_path = norm_video
        ctx.audio_path = audio_path
        ctx.sample_rate = 16000

        log.info(f"Normalized: video={norm_video}, audio={audio_path}")
        return StageResult(True, self.name,
                           output_files=[norm_video, audio_path])
