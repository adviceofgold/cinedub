import gc
import json
import os
import subprocess
import sys
import tempfile

import torch

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class VoiceCloningStage(Stage):
    def __init__(self):
        super().__init__("voice_cloning", depends_on=["speaker_db", "translation"])

    def validate(self, ctx: PipelineContext) -> bool:
        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            return False
        return True

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting GPT-SoVITS voice cloning")

        speakers = ctx.metadata.get("speaker_db", {})
        translated = ctx.metadata.get("translated_segments", [])
        if not speakers:
            return StageResult(False, self.name, message="No speaker database")
        if not translated:
            return StageResult(False, self.name, message="No translations")

        out_dir = ctx.config.subdir("cloned_audio")
        clone_root = os.path.join(ctx.config.cache_dir, "GPT-SoVITS")

        self._ensure_gpt_sovits(clone_root, log)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cloned_files = []

        for i, seg in enumerate(translated):
            text = seg.get("translated", "").strip()
            if not text:
                continue

            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = self._assign_speaker(start, end, speakers)
            ref_audio = speakers.get(speaker, {}).get("reference_audio", "")

            if not os.path.isfile(ref_audio):
                log.warn(f"Reference audio not found for {speaker}, skipping segment {i}")
                continue

            out_path = os.path.join(out_dir, f"seg_{start:.1f}_{end:.1f}.wav")
            if self._clone_segment(text, ref_audio, out_path, clone_root, device, log):
                cloned_files.append(out_path)

            if (i + 1) % 10 == 0:
                log.info(f"Cloned {i+1}/{len(translated)} segments")

        ctx.metadata["cloned_audio_files"] = cloned_files
        log.info(f"Voice cloning done: {len(cloned_files)} segments generated")

        return StageResult(True, self.name,
                           output_files=cloned_files,
                           metrics={"segments_generated": len(cloned_files)})

    def _ensure_gpt_sovits(self, clone_root: str, log):
        if os.path.isdir(os.path.join(clone_root, "GPT_SoVITS")):
            sys.path.insert(0, clone_root)
            return

        log.info("Cloning GPT-SoVITS repository")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/RVC-Boss/GPT-SoVITS.git",
            clone_root
        ], capture_output=True, text=True, timeout=300)

        sys.path.insert(0, clone_root)

        log.info("Installing GPT-SoVITS dependencies")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r",
            os.path.join(clone_root, "requirements.txt")
        ], capture_output=True, text=True, timeout=300)

    def _assign_speaker(self, start: float, end: float,
                        speakers: dict) -> str:
        best_spk = list(speakers.keys())[0]
        best_overlap = 0
        for spk, data in speakers.items():
            for utt in data.get("utterances", []):
                overlap = min(end, utt["end"]) - max(start, utt["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = spk
        return best_spk

    def _clone_segment(self, text: str, ref_audio: str,
                       out_path: str, clone_root: str,
                       device: str, log) -> bool:
        try:
            import librosa
            import soundfile as sf

            audio, sr = librosa.load(ref_audio, sr=24000, mono=True, duration=10.0)
            temp_ref = out_path.replace(".wav", "_ref.wav")
            sf.write(temp_ref, audio, sr)

            infer_script = os.path.join(clone_root, "GPT_SoVITS", "inference_cli.py")
            if not os.path.isfile(infer_script):
                infer_script = os.path.join(clone_root, "api.py")

            if os.path.isfile(infer_script):
                result = subprocess.run([
                    sys.executable, infer_script,
                    "--text", text,
                    "--ref_audio", temp_ref,
                    "--output", out_path,
                    "--device", device,
                ], capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    log.warn(f"Clone failed for segment: {result.stderr[:200]}")
                    os.remove(temp_ref)
                    return False
            else:
                log.warn("GPT-SoVITS inference script not found, using fallback TTS")
                self._fallback_tts(text, out_path)

            if os.path.isfile(temp_ref):
                os.remove(temp_ref)

            if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1000:
                log.warn(f"Clone output too small or missing: {out_path}")
                return False

            return True

        except Exception as e:
            log.warn(f"Clone exception: {e}")
            return False

    def _fallback_tts(self, text: str, out_path: str):
        import torch
        from transformers import pipeline

        pipe = pipeline(
            "text-to-speech",
            model="suno/bark-small",
            device=0 if torch.cuda.is_available() else -1,
        )
        output = pipe(text)
        import soundfile as sf
        sf.write(out_path, output["audio"][0], output["sampling_rate"])
