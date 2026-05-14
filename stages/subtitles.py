import os
import re

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class SubtitleStage(Stage):
    def __init__(self):
        super().__init__("subtitles", depends_on=["translation"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Generating subtitles")

        translated = ctx.metadata.get("translated_segments", [])
        if not translated:
            return StageResult(False, self.name, message="No translated segments")

        out_dir = ctx.config.subdir("subtitles")
        target = ctx.config.target_language

        srt_path = os.path.join(out_dir, f"subtitles_{target}.srt")
        ass_path = os.path.join(out_dir, f"subtitles_{target}.ass")
        vtt_path = os.path.join(out_dir, f"subtitles_{target}.vtt")

        self._write_srt(translated, srt_path)
        self._write_ass(translated, ass_path)
        self._write_vtt(translated, vtt_path)

        ctx.metadata["subtitle_path"] = srt_path
        log.info(f"Subtitles generated: {srt_path}")
        return StageResult(True, self.name,
                           output_files=[srt_path, ass_path, vtt_path],
                           metrics={"format": "srt+ass+vtt"})

    def _chunk_text(self, text: str) -> list[str]:
        max_chars = 42
        words = text.split()
        chunks = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = w
            else:
                current += " " + w if current else w
        if current:
            chunks.append(current.strip())
        return chunks

    def _format_timestamp(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _format_vtt_timestamp(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _write_srt(self, segments: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            idx = 1
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("translated", "").strip()
                if not text:
                    continue

                dur = end - start
                if dur < 1.0:
                    end = start + 1.0
                if dur > 6.0:
                    end = start + 6.0

                chunks = self._chunk_text(text)
                if len(chunks) > 1:
                    mid = start + (end - start) / 2
                    f.write(f"{idx}\n")
                    f.write(f"{self._format_timestamp(start)} --> "
                            f"{self._format_timestamp(mid)}\n")
                    f.write(f"{chunks[0]}\n\n")
                    idx += 1
                    f.write(f"{idx}\n")
                    f.write(f"{self._format_timestamp(mid)} --> "
                            f"{self._format_timestamp(end)}\n")
                    f.write(f"{chunks[1]}\n\n")
                else:
                    f.write(f"{idx}\n")
                    f.write(f"{self._format_timestamp(start)} --> "
                            f"{self._format_timestamp(end)}\n")
                    f.write(f"{text}\n\n")
                idx += 1

    def _write_ass(self, segments: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n\n")
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, "
                    "SecondaryColour, OutlineColour, BackColour, Bold, "
                    "Italic, Underline, StrikeOut, ScaleX, ScaleY, "
                    "Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,"
                    "&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,"
                    "2,10,10,10,1\n\n")
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, "
                    "MarginL, MarginR, MarginV, Effect, Text\n")

            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("translated", "").strip()
                if not text:
                    continue

                s = self._format_timestamp(start).replace(",", ".")
                e = self._format_timestamp(end).replace(",", ".")
                safe_text = text.replace("{", "\\{").replace("}", "\\}")
                f.write(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{safe_text}\n")

    def _write_vtt(self, segments: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("translated", "").strip()
                if not text:
                    continue

                f.write(f"{self._format_vtt_timestamp(start)} --> "
                        f"{self._format_vtt_timestamp(end)}\n")
                f.write(f"{text}\n\n")
