import json
import os

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class AlignmentStage(Stage):
    def __init__(self):
        super().__init__("alignment", depends_on=["transcription"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting forced alignment")

        audio = ctx.metadata.get("vocals_path") or ctx.audio_path
        raw = ctx.metadata.get("transcript_raw")
        if not raw:
            return StageResult(False, self.name, message="No transcript to align")

        import whisperx
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mm = ModelManager.get()
        lang = raw.get("language", "en")

        def _load_align(**kw):
            return whisperx.load_align_model(
                language_code=kw["lang"], device=device
            )

        align_model, metadata = mm.load("align_model", _load_align, lang=lang)

        log.info(f"Running alignment (lang={lang})")
        result = whisperx.align(
            raw["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        out_dir = ctx.config.subdir("transcripts")
        aligned_path = os.path.join(out_dir, "aligned_transcript.json")
        with open(aligned_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        ctx.metadata["aligned_transcript"] = result
        log.info(f"Alignment done: {len(result.get('segments', []))} segments")

        mm.unload("align_model")
        mm.unload_model = None
        return StageResult(True, self.name,
                           output_files=[aligned_path],
                           metrics={"word_count": sum(
                               len(s.get("words", [])) for s in result.get("segments", [])
                           )})
