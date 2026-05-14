import json
import os

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class TranscriptionStage(Stage):
    def __init__(self):
        super().__init__("transcription", depends_on=["diarization"])

    def validate(self, ctx: PipelineContext) -> bool:
        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            return False
        return True

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting WhisperX transcription")

        audio = ctx.metadata.get("vocals_path") or ctx.audio_path
        if not os.path.isfile(audio):
            return StageResult(False, self.name, message=f"Audio not found: {audio}")

        import whisperx
        import torch

        mm = ModelManager.get()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if torch.cuda.is_available() else "int8"

        model_name = "large-v3"
        if ctx.config.mode == RuntimeMode.GPU_BALANCED:
            model_name = "medium"
        elif ctx.config.mode == RuntimeMode.CPU_SAFE:
            model_name = "base"

        log.info(f"Loading WhisperX model: {model_name} ({compute})")

        def _load_asr(**kw):
            return whisperx.load_model(
                kw["model"], device=device, compute_type=compute,
                asr_options={"hotwords": kw.get("hotwords", None)},
            )

        model = mm.load("whisperx", _load_asr, model=model_name)

        log.info("Running batched transcription")
        result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)

        lang = result.get("language", "en")
        log.info(f"Detected language: {lang}")

        out_dir = ctx.config.subdir("transcripts")
        raw_path = os.path.join(out_dir, "raw_transcript.json")
        with open(raw_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        ctx.metadata["transcript_language"] = lang
        ctx.metadata["transcript_raw"] = result
        log.info(f"Transcription done: {len(result.get('segments', []))} segments")

        mm.unload("whisperx")
        return StageResult(True, self.name,
                           output_files=[raw_path],
                           metrics={"language": lang,
                                    "segments": len(result.get("segments", []))})
