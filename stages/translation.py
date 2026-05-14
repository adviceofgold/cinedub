import json
import os

from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode, ModelManager
from ..logging import CineDubLogger


class TranslationStage(Stage):
    def __init__(self):
        super().__init__("translation", depends_on=["alignment"])

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        target = ctx.config.target_language
        log.info(f"Starting NLLB translation to '{target}'")

        aligned = ctx.metadata.get("aligned_transcript")
        if not aligned:
            return StageResult(False, self.name, message="No aligned transcript")

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        mm = ModelManager.get()
        model_name = "facebook/nllb-200-distilled-600M"

        device_map = mm.get_device_map(2.1)
        device = device_map.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = device_map.get("torch_dtype", torch.float16 if device == "cuda" else torch.float32)

        def _load_tokenizer(**kw):
            return AutoTokenizer.from_pretrained(kw["name"])

        def _load_model(**kw):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                kw["name"],
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                model = model.to("cpu")
            return model

        tokenizer = mm.load("nllb_tokenizer", _load_tokenizer, name=model_name)
        model = mm.load("nllb_model", _load_model, name=model_name)

        lang_map = {
            "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
            "pt": "por_Latn", "it": "ita_Latn", "ru": "rus_Cyrl",
            "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "ar": "ara_Arab", "hi": "hin_Deva", "bn": "ben_Beng",
            "ur": "urd_Arab", "tr": "tur_Latn", "nl": "nld_Latn",
            "pl": "pol_Latn", "vi": "vie_Latn", "th": "tha_Thai",
            "en": "eng_Latn",
        }
        target_flores = lang_map.get(target, "spa_Latn")
        source_lang = aligned.get("language", "en")
        source_flores = lang_map.get(source_lang, "eng_Latn")

        tokenizer.src_lang = source_flores

        segments = []
        for seg in aligned.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                translated = model.generate(
                    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_flores],
                    max_length=512
                )
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "original": text,
                "translated": translated_text,
                "words": seg.get("words", []),
            })

        out_dir = ctx.config.subdir("translations")
        trans_path = os.path.join(out_dir, f"translated_{target}.json")
        with open(trans_path, "w") as f:
            json.dump(segments, f, indent=2, default=str)

        ctx.metadata["translated_segments"] = segments
        log.info(f"Translation done: {len(segments)} segments to '{target}'")

        mm.unload("nllb_tokenizer")
        mm.unload("nllb_model")
        return StageResult(True, self.name,
                           output_files=[trans_path],
                           metrics={"target_language": target,
                                    "segments": len(segments)})
