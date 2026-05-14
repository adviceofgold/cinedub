import json
import os

import numpy as np

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class AudioSyncStage(Stage):
    def __init__(self):
        super().__init__("audio_sync", depends_on=["voice_cloning"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting audio synchronization")

        cloned = ctx.metadata.get("cloned_audio_files", [])
        translated = ctx.metadata.get("translated_segments", [])
        if not cloned:
            return StageResult(False, self.name, message="No cloned audio to sync")

        import librosa
        import soundfile as sf

        out_dir = ctx.config.subdir("cloned_audio")
        synced_files = []

        for i, seg in enumerate(translated):
            text = seg.get("translated", "").strip()
            if not text:
                continue

            start = seg.get("start", 0)
            end = seg.get("end", 0)
            original_dur = end - start

            cloned_path = os.path.join(out_dir, f"seg_{start:.1f}_{end:.1f}.wav")
            if not os.path.isfile(cloned_path):
                continue

            synced_path = cloned_path.replace(".wav", "_synced.wav")
            if self._sync_segment(cloned_path, synced_path, original_dur):
                synced_files.append(synced_path)

        ctx.metadata["synced_audio_files"] = synced_files
        log.info(f"Sync done: {len(synced_files)} segments")
        return StageResult(True, self.name,
                           output_files=synced_files,
                           metrics={"segments_synced": len(synced_files)})

    def _sync_segment(self, input_path: str, output_path: str,
                      target_dur: float) -> bool:
        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(input_path, sr=24000, mono=True)
            current_dur = len(y) / sr

            if current_dur < 0.1:
                return False

            ratio = current_dur / max(target_dur, 0.1)
            ratio = max(0.5, min(ratio, 2.0))

            if abs(ratio - 1.0) > 0.05:
                try:
                    import pyrubberband as pyrb
                    y_stretched = pyrb.time_stretch(y, sr, ratio)
                except ImportError:
                    y_stretched = librosa.effects.time_stretch(y=y,
                        rate=ratio)

                target_len = int(target_dur * sr)
                if len(y_stretched) < target_len:
                    pad = target_len - len(y_stretched)
                    y_stretched = np.pad(y_stretched, (0, pad))
                else:
                    y_stretched = y_stretched[:target_len]

                sf.write(output_path, y_stretched, sr)
            else:
                import shutil
                shutil.copy2(input_path, output_path)

            return True

        except Exception as e:
            return False
