import os
import subprocess
import sys

import torch

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class LipsyncStage(Stage):
    def __init__(self):
        super().__init__("lipsync", depends_on=["render"])

    def validate(self, ctx: PipelineContext) -> bool:
        if not ctx.config.enable_lipsync:
            return False
        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            return False
        if ctx.config.mode == RuntimeMode.GPU_BALANCED and \
                ModelManager.vram_gb() < 8:
            return False
        return True

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting Wav2Lip lip-sync")

        dubbed_video = ctx.metadata.get("dubbed_video_path", "")
        synced_audio = ctx.metadata.get("merged_audio_path", "")

        if not os.path.isfile(dubbed_video):
            return StageResult(False, self.name,
                               message="Dubbed video not found")
        if not os.path.isfile(synced_audio):
            return StageResult(False, self.name,
                               message="Synced audio not found")

        out_dir = ctx.config.subdir("lipsync")
        wav2lip_root = os.path.join(ctx.config.cache_dir, "Wav2Lip")

        if not os.path.isdir(wav2lip_root):
            log.info("Cloning Wav2Lip")
            subprocess.run([
                "git", "clone", "--depth=1",
                "https://github.com/Rudrabha/Wav2Lip.git",
                wav2lip_root
            ], capture_output=True, text=True, timeout=120)
            sys.path.insert(0, wav2lip_root)

        checkpoint = os.path.join(wav2lip_root, "checkpoints",
                                   "wav2lip_gan.pth")
        if not os.path.isfile(checkpoint):
            log.info("Downloading Wav2Lip GAN checkpoint")
            subprocess.run([
                "gdown", "--fuzzy",
                "https://drive.google.com/uc?id=1aUYc4EXgJ9-"
                "B6I1M3TlG2QmOz0rYvNx",
                "-O", checkpoint,
            ], capture_output=True, text=True, timeout=300)

        result = subprocess.run([
            sys.executable, os.path.join(wav2lip_root, "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", dubbed_video,
            "--audio", synced_audio,
            "--outfile", os.path.join(out_dir, "lipsynced.mp4"),
            "--pads", "0", "0", "0", "0",
            "--resize_factor", "2",
        ], capture_output=True, text=True, timeout=1800)

        lipsync_path = os.path.join(out_dir, "lipsynced.mp4")

        if result.returncode != 0 or not os.path.isfile(lipsync_path):
            log.warn(f"Wav2Lip failed: {result.stderr[:300]}")
            log.info("Falling back: using dubbed video without lip-sync")
            return StageResult(True, self.name,
                               message="Lipsync failed, using raw dubbed video",
                               output_files=[dubbed_video])

        ctx.metadata["lipsync_path"] = lipsync_path
        log.info(f"Lip-sync done: {lipsync_path}")
        return StageResult(True, self.name,
                           output_files=[lipsync_path])
