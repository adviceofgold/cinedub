import os
import subprocess

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode
from ..logging import CineDubLogger


class RenderStage(Stage):
    def __init__(self):
        super().__init__("render", depends_on=["audio_sync", "subtitles"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting FFmpeg video render")

        video = ctx.video_path
        subtitle = ctx.metadata.get("subtitle_path", "")
        synced_files = ctx.metadata.get("synced_audio_files", [])
        no_vocals = ctx.metadata.get("no_vocals_path", "")

        if not video or not os.path.isfile(video):
            return StageResult(False, self.name,
                               message="Video file not found")

        if not synced_files:
            return StageResult(False, self.name,
                               message="No synced audio to mix")

        out_dir = ctx.config.subdir("renders")
        final_dir = ctx.config.subdir("final")

        merged_audio = os.path.join(out_dir, "merged_audio.wav")
        self._merge_audio(synced_files, merged_audio, no_vocals, log)

        dubbed_video = os.path.join(final_dir, "dubbed_final.mp4")
        self._render_video(video, merged_audio, subtitle, dubbed_video, log)

        ctx.metadata["merged_audio_path"] = merged_audio
        ctx.metadata["dubbed_video_path"] = dubbed_video

        log.info(f"Render complete: {dubbed_video}")
        return StageResult(True, self.name,
                           output_files=[dubbed_video, merged_audio])

    def _merge_audio(self, synced_files: list, output_path: str,
                     no_vocals: str, log):
        import soundfile as sf
        import numpy as np

        log.info(f"Merging {len(synced_files)} synced audio segments")
        merged = []
        for fp in sorted(synced_files):
            if os.path.isfile(fp):
                try:
                    y, _ = sf.read(fp)
                    merged.append(y)
                except Exception:
                    continue

        if not merged:
            log.warn("No valid synced audio, using empty track")
            sf.write(output_path, np.zeros(16000), 16000)
            return

        combined = np.concatenate(merged)

        if os.path.isfile(no_vocals):
            log.info("Mixing cloned vocals with original background")
            bg, sr = sf.read(no_vocals)
            min_len = min(len(combined), len(bg))
            combined = combined[:min_len] * 0.85 + bg[:min_len] * 0.15

        sf.write(output_path, combined, 24000)
        log.info(f"Merged audio: {output_path}")

    def _render_video(self, video: str, audio: str, subtitle: str,
                      output_path: str, log):
        log.info("Rendering final video with FFmpeg")

        cmd = [
            "ffmpeg", "-i", video, "-i", audio,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
        ]

        if subtitle and os.path.isfile(subtitle):
            log.info("Embedding subtitles")
            cmd.extend(["-vf", f"subtitles={subtitle}"])

        cmd.append("-y")
        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=3600)
        if result.returncode != 0:
            log.warn(f"FFmpeg failed: {result.stderr[:300]}")
            log.info("Retrying without subtitles")
            retry_cmd = [
                "ffmpeg", "-i", video, "-i", audio,
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path,
            ]
            subprocess.run(retry_cmd, capture_output=True, text=True,
                          timeout=3600)
