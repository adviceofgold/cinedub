import os
import subprocess
import json

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode
from ..logging import CineDubLogger


class DownloadStage(Stage):
    def __init__(self):
        super().__init__("download", depends_on=["bootstrap"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        url = ctx.config.source_url
        if not url:
            return StageResult(False, self.name, message="No source URL provided")

        out_dir = ctx.config.subdir("input")
        log.info(f"Downloading: {url}")

        result = subprocess.run(
            ["yt-dlp",
             "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
             "-o", os.path.join(out_dir, "%(title)s.%(ext)s"),
             "--print", "after_video:json",
             url],
            capture_output=True, text=True, timeout=3600
        )

        if result.returncode != 0:
            log.error(f"yt-dlp failed: {result.stderr[:500]}")
            return StageResult(False, self.name,
                               message=f"Download failed: {result.stderr[:200]}")

        info = None
        for line in result.stdout.strip().split("\n"):
            try:
                info = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        if info and info.get("_filename"):
            ctx.video_path = info["_filename"]
            ctx.metadata["title"] = info.get("title", "")
            ctx.metadata["duration"] = info.get("duration", 0)
            ctx.duration_s = float(info.get("duration", 0))
        else:
            candidates = [f for f in os.listdir(out_dir)
                          if f.endswith((".mp4", ".mkv", ".webm"))]
            if not candidates:
                return StageResult(False, self.name, message="No video found")
            ctx.video_path = os.path.join(out_dir, candidates[0])

        log.info(f"Downloaded: {ctx.video_path}")
        return StageResult(True, self.name,
                           output_files=[ctx.video_path],
                           metrics={"duration_s": ctx.duration_s})
