import json
import os

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class DiarizationStage(Stage):
    def __init__(self):
        super().__init__("diarization", depends_on=["separation"])

    def validate(self, ctx: PipelineContext) -> bool:
        if ctx.config.mode == RuntimeMode.CPU_SAFE:
            return False
        return True

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Starting diarization with pyannote.audio 4.0.4")

        audio = ctx.metadata.get("vocals_path") or ctx.audio_path
        if not os.path.isfile(audio):
            return StageResult(False, self.name, message=f"Audio not found: {audio}")

        mm = ModelManager.get()

        def _load_pipeline(**kw):
            from pyannote.audio import Pipeline
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=os.environ.get("HF_TOKEN", True),
            )
            if torch.cuda.is_available():
                pipe.cuda()
            return pipe

        import torch
        pipeline = mm.load("pyannote", _load_pipeline)

        log.info("Running diarization pipeline")
        diarization = pipeline(audio)

        out_dir = ctx.config.subdir("diarization")
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
            })

        segments_path = os.path.join(out_dir, "segments.json")
        with open(segments_path, "w") as f:
            json.dump(segments, f, indent=2)

        rttm_path = os.path.join(out_dir, "diarization.rttm")
        with open(rttm_path, "w") as f:
            diarization.write_rttm(f)

        ctx.metadata["diarization_segments"] = segments
        ctx.metadata["num_speakers"] = len(set(s["speaker"] for s in segments))
        log.info(f"Diarization done: {len(segments)} segments, "
                 f"{ctx.metadata['num_speakers']} speakers")

        mm.unload("pyannote")
        return StageResult(True, self.name,
                           output_files=[segments_path, rttm_path],
                           metrics={"num_segments": len(segments),
                                    "num_speakers": ctx.metadata["num_speakers"]})
