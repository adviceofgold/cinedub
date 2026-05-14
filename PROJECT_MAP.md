# CineDub OSS — Project Map

> Generated: 2026-05-14
> Author: Staff Engineer / Tech Lead

---

## [TECH_STACK]

### Pinned Dependency Versions (verified May 2026)

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| Python | >=3.10 | — | Colab default is 3.10+ |
| yt-dlp | **2026.03.17** | PyPI | Requires Python >=3.10 |
| WhisperX | **v3.8.5** | PyPI | Apr 1, 2026. torch 2.8 compat |
| pyannote.audio | **4.0.4** | PyPI | Feb 7, 2026. Uses `community-1` pipeline |
| Demucs (HT) | **v4** (adefossez fork) | GitHub | FB/archived. Use `adefossez/demucs` |
| GPT-SoVITS | **20250606v2pro** | GitHub | V3 model available. Use API mode |
| Transformers | **v5.8.0** | PyPI | May 5, 2026. NLLB-200 integration |
| NLLB-200 | distilled-600M | HuggingFace | T4-safe. Fallback: 1.3B |
| Wav2Lip | original (patched) | GitHub | Python 3.6 upstream. Wrapped compat layer |
| FFmpeg | system | Colab | Included by default |
| Gradio | latest | PyPI | UI layer |
| Rubberband | system | apt | Audio stretching |
| librosa | latest | PyPI | Audio analysis |
| pydub | latest | PyPI | Audio merging |

### Deprecation Warnings
- `facebookresearch/demucs` **archived** Jan 1, 2025 → use `adefossez/demucs`
- `Rudrabha/Wav2Lip` requires Python 3.6 → must use `wav2lip_gan.pth` weights only, with modern PyTorch wrapper
- `pyannote.audio` 3.x is legacy → use 4.0.4 with `community-1`

---

## [SYSTEM_FLOW]

```
YouTube URL
    │
    ▼ [00_bootstrap · 01_drive_mount]
Mount Drive, install deps, detect runtime
    │
    ▼ [02_dependency_manager · 03_runtime_detector]
Resolve GPU_FULL | GPU_BALANCED | CPU_SAFE | RECOVERY
    │
    ▼ [04_global_config · 05_project_state · 06_checkpoint_manager]
Init config, state, checkpoint DB
    │
    ╔══════════════════════════════════════════════╗
    ║          STAGE PIPELINE (all resumable)      ║
    ╠══════════════════════════════════════════════╣
    ║  07_download_pipeline   (yt-dlp)            ║
    ║  08_media_normalization (FFmpeg)            ║
    ║  09_audio_separation    (Demucs)            ║
    ║  10_diarization         (pyannote 4.0.4)    ║
    ║  11_transcription       (WhisperX v3.8.5)   ║
    ║  12_alignment           (WhisperX forced)   ║
    ║  13_translation         (NLLB-200 via HF)   ║
    ║  14_speaker_database    (embedding store)   ║
    ║  15_voice_cloning       (GPT-SoVITS v3)    ║
    ║  16_audio_sync          (Rubberband+librosa)║
    ║  17_subtitle_engine     (SRT/ASS/VTT)       ║
    ║  18_lipsync_engine      (Wav2Lip patched)  ║
    ║  19_video_rendering     (FFmpeg)            ║
    ╚══════════════════════════════════════════════╝
    │
    ▼ [20_pipeline_orchestrator]
Sequential stage dispatch with checkpoint gating
    │
    ▼ [21_gradio_ui]
User interface
    │
    ▼ [22_recovery_system · 23_cleanup_engine]
Post-run cleanup
```

---

## [ARCHITECTURE]

### Directory Layout

```
/content/drive/MyDrive/cinedub/
├── projects/{project_name}/
│   ├── input/              # raw video
│   ├── separated_audio/    # Demucs output (vocals, music, etc.)
│   ├── diarization/        # pyannote RTTM + segments JSON
│   ├── transcripts/        # WhisperX JSON + aligned words
│   ├── translations/       # NLLB-200 per-language
│   ├── speakers/           # reference audios + embeddings
│   ├── cloned_audio/       # GPT-SoVITS per-segment WAVs
│   ├── subtitles/          # SRT/ASS/VTT
│   ├── lipsync/            # Wav2Lip frames
│   ├── renders/            # stage renders
│   └── final/              # final export
├── checkpoints/            # {stage}.done marker files
├── models/                 # cached model weights
├── cache/                  # HF cache, torch hub
├── temp/                   # transient work
├── outputs/                # convenience symlinks
└── logs/                   # structured logs
```

### Stage Abstraction Contract

Every stage implements:
```python
class Stage(ABC):
    name: str
    depends_on: list[str]

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> StageResult: ...
    @abstractmethod
    def validate(self, ctx: PipelineContext) -> bool: ...  # pre-flight

    def save_checkpoint(self, ctx): ...
    def load_checkpoint(self, ctx) -> bool: ...
```

### Runtime Mode Resolution

```python
def resolve_runtime_mode() -> RuntimeMode:
    if not torch.cuda.is_available():        return CPU_SAFE
    vram_gb = torch.cuda.get_device_props(0).total_memory / 1e9
    if vram_gb < 10:                          return GPU_BALANCED
    if checkpoint_exists():                   return RECOVERY
    return GPU_FULL
```

### Model Manager (Singleton)

```python
class ModelManager:
    _registry: dict[str, any] = {}

    def load(self, key: str, loader: Callable, **kwargs) -> any:
        # lazy load, track GPU mem
    def unload(self, key: str):
        # del + gc.collect + torch.cuda.empty_cache
    def unload_all(self):
        # evict everything
    def get_device_map(self, model_size: str) -> dict:
        # auto device_map for HF models on T4
```

### Chunking Strategy

| Stage | Chunk Type | Size | Overlap |
|-------|-----------|------|---------|
| Demucs | audio | 30s | 1s |
| WhisperX | audio (VAD) | 30s | 0 |
| GPT-SoVITS | text segment | per-utterance | 0 |
| Wav2Lip | video frames | 5-10s | 0.5s |
| FFmpeg | video segment | GOP-aligned | keyframes |

### Checkpoint Protocol

Each stage writes `<stage_name>.done` with JSON metadata:
```json
{
  "stage": "diarization",
  "timestamp": "2026-05-14T12:00:00Z",
  "duration_s": 124.5,
  "output_files": [...],
  "checksum": "sha256:...",
  "runtime_mode": "GPU_FULL"
}
```

Recovery reads all `.done` files → determines `resume_from`.

### Error Handling Strategy

- All stage calls wrapped in `retry(max=3, backoff=2.0, on=OOM)`
- CUDA OOM → `ModelManager.unload_all()` → reduce batch size → retry
- Network errors → retry with exponential backoff
- FFmpeg crashes → validate output file existence
- Partial outputs always persisted before retry

### Memory Optimization Rules

1. **Never** load >1 large model concurrently
2. **Always** use `fp16` on GPU
3. **Always** use `device_map="auto"` for HF models
4. **Always** write audio/video to disk immediately (no in-memory accumulation)
5. **Always** call `gc.collect()` + `torch.cuda.empty_cache()` between stages
6. **Prefer** `NLLB-distilled-600M` on T4 (2.1GB fp16); upgrade to 1.3B only on A100

---

## [ORPHANS & PENDING]

### Implemented (v1)
- Stage abstraction (`Stage`, `StageResult`, `PipelineContext`) → `cinedub/core.py`
- Model Manager (lazy load / unload / GPU cleanup) → `cinedub/core.py`
- Checkpoint Protocol (`.done` markers, resume, verify) → `cinedub/checkpoint.py`
- Structured JSON logging → `cinedub/logging.py`
- Bootstrap + Drive mount → `cinedub/stages/bootstrap.py`
- yt-dlp download → `cinedub/stages/download.py`
- FFmpeg normalization → `cinedub/stages/normalize.py`
- Demucs vocal separation → `cinedub/stages/separation.py`
- pyannote 4.0.4 diarization → `cinedub/stages/diarization.py`
- WhisperX transcription → `cinedub/stages/transcription.py`
- Forced alignment → `cinedub/stages/alignment.py`
- NLLB-200 translation (distilled-600M, user-selectable lang) → `cinedub/stages/translation.py`
- Speaker database → `cinedub/stages/speaker_db.py`
- GPT-SoVITS v3 voice cloning → `cinedub/stages/cloning.py`
- Rubberband/librosa audio sync → `cinedub/stages/sync.py`
- SRT/ASS/VTT subtitle engine → `cinedub/stages/subtitles.py`
- Wav2Lip optional lip-sync → `cinedub/stages/lipsync.py`
- FFmpeg video render → `cinedub/stages/render.py`
- Pipeline orchestrator (sequential stage dispatch) → `cinedub/stages/orchestrator.py`
- Gradio UI → `cinedub/stages/ui.py`
- Recovery system → `cinedub/stages/recovery.py`
- Cleanup engine → `cinedub/stages/cleanup.py`
- Colab notebook (24 cells) → `cinedub.ipynb`
- 44 unit/integration tests → `tests/`

### Deferred to v2
- InfiniteTalk cinematic lip-sync engine (too GPU-heavy, unstable on T4)
- Real-time streaming mode
- Web-based job queue
- Custom fine-tuning UI
- CI pipeline (GitHub Actions for Colab execution)
- Golden test data for multi-speaker scenarios
