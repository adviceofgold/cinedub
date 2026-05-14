import os
import subprocess
import sys

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class BootstrapStage(Stage):
    def __init__(self):
        super().__init__("bootstrap")

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Bootstrap: starting environment setup")

        pip_packages = [
            "torch>=2.0",
            "torchaudio",
            "torchvision",
            "yt-dlp>=2026.3.17",
            "whisperx>=3.8.5",
            "pyannote.audio>=4.0.4",
            "transformers>=5.5",
            "librosa",
            "pydub",
            "gradio>=5.0",
            "soundfile",
            "accelerate",
            "sentencepiece",
            "protobuf",
        ]

        for pkg in pip_packages:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log.warn(f"pip install {pkg} failed: {result.stderr[:200]}")

        log.info("Bootstrap: mounting Google Drive")
        drive_path = "/content/drive/MyDrive"
        if not os.path.isdir(drive_path):
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except ImportError:
                log.warn("Not in Colab; Drive mount skipped")
                os.makedirs(drive_path, exist_ok=True)

        ctx.config.ensure_dirs()
        log.info("Bootstrap: environment ready")
        return StageResult(True, self.name)
