import json
import os

from ..core import Config, PipelineContext, Stage, StageResult
from ..logging import CineDubLogger


class SpeakerDatabaseStage(Stage):
    def __init__(self):
        super().__init__("speaker_db", depends_on=["diarization", "alignment"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        log.info("Building speaker database")

        segments = ctx.metadata.get("diarization_segments", [])
        aligned = ctx.metadata.get("aligned_transcript", {})
        if not segments:
            return StageResult(False, self.name, message="No diarization segments")

        aligned_segs = aligned.get("segments", [])
        audio_path = ctx.metadata.get("vocals_path") or ctx.audio_path

        import librosa
        import soundfile as sf

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        speaker_utts = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_utts:
                speaker_utts[spk] = []
            speaker_utts[spk].append(seg)

        out_dir = ctx.config.subdir("speakers")
        speakers_db = {}

        for spk, utts in speaker_utts.items():
            log.info(f"Processing speaker {spk}: {len(utts)} utterances")

            ref_audio = os.path.join(out_dir, f"{spk}_reference.wav")
            longest = max(utts, key=lambda u: u["end"] - u["start"])
            start_s = int(longest["start"] * sr)
            end_s = int(longest["end"] * sr)
            ref_segment = audio[start_s:end_s]
            sf.write(ref_audio, ref_segment, sr)

            speakers_db[spk] = {
                "speaker_id": spk,
                "reference_audio": ref_audio,
                "utterances": utts,
            }

        db_path = os.path.join(out_dir, "speakers_db.json")
        with open(db_path, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "reference_audio"}
                       for k, v in speakers_db.items()},
                      f, indent=2, default=str)

        ctx.metadata["speaker_db"] = speakers_db
        log.info(f"Speaker DB done: {len(speakers_db)} speakers")
        return StageResult(True, self.name,
                           output_files=[db_path] + [s["reference_audio"]
                                                      for s in speakers_db.values()],
                           metrics={"num_speakers": len(speakers_db)})
