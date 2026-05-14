import json

import gradio as gr

from ..core import Config, RuntimeMode, ModelManager
from ..checkpoint import CheckpointManager
from ..logging import CineDubLogger
from .orchestrator import PipelineOrchestrator


def create_interface():
    log = CineDubLogger.get()

    def run_pipeline(url, language, enable_lipsync, quality, mode_override):
        if not url:
            return "", "", "", "Error: No URL provided"

        config = Config()
        config.source_url = url
        config.target_language = language or "es"
        config.enable_lipsync = enable_lipsync
        config.quality = quality or "balanced"

        if mode_override == "auto":
            config.mode = ModelManager.resolve_mode()
        else:
            config.mode = RuntimeMode(mode_override)

        if config.mode == RuntimeMode.RECOVERY:
            cp = CheckpointManager(config)
            completed = cp.completed_stages()
            progress_text = f"Recovery mode: {len(completed)} stages found"

        project_name = url.split("/")[-1][:40]
        config.project_name = project_name
        config.ensure_dirs()

        log.info(f"Starting pipeline: url={url}, lang={language}, "
                 f"mode={config.mode.value}")

        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run()

        summary_lines = []
        for r in results:
            status = "OK" if r.success else "FAIL"
            summary_lines.append(f"[{status}] {r.stage_name}: {r.message}")

        final_video = orchestrator.ctx.metadata.get("dubbed_video_path", "")
        subtitle_path = orchestrator.ctx.metadata.get("subtitle_path", "")

        summary = "\n".join(summary_lines)
        return final_video, subtitle_path, summary, f"Mode: {config.mode.value}"

    def check_status():
        cp = CheckpointManager(Config())
        completed = cp.completed_stages()
        mode = ModelManager.resolve_mode()
        info = json.dumps({
            "mode": mode.value,
            "vram_gb": round(ModelManager.vram_gb(), 1),
            "completed_stages": completed,
        }, indent=2)
        return info

    with gr.Blocks(title="CineDub OSS", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# CineDub OSS — AI Video Dubbing")

        with gr.Row():
            with gr.Column(scale=2):
                url = gr.Textbox(label="YouTube URL",
                                 placeholder="https://youtube.com/watch?v=...")
                lang = gr.Dropdown(
                    choices=["es", "fr", "de", "pt", "it", "ru", "zh",
                             "ja", "ko", "ar", "hi", "tr", "nl", "pl",
                             "vi", "th", "en"],
                    value="es", label="Target Language"
                )
                quality = gr.Radio(
                    choices=["fast", "balanced", "quality"],
                    value="balanced", label="Quality"
                )
                lipsync = gr.Checkbox(label="Enable Lip Sync", value=False)
                mode = gr.Radio(
                    choices=["auto", "GPU_FULL", "GPU_BALANCED",
                             "CPU_SAFE", "RECOVERY"],
                    value="auto", label="Runtime Mode"
                )
                run_btn = gr.Button("Start Dubbing", variant="primary")

            with gr.Column(scale=3):
                video_out = gr.Video(label="Dubbed Video")
                subtitle_out = gr.File(label="Subtitles")
                summary_out = gr.Textbox(label="Pipeline Summary", lines=10)
                status_out = gr.Textbox(label="System Status")

        run_btn.click(
            fn=run_pipeline,
            inputs=[url, lang, lipsync, quality, mode],
            outputs=[video_out, subtitle_out, summary_out, status_out],
        )

        status_btn = gr.Button("Check System Status")
        status_btn.click(fn=check_status, outputs=status_out)

    return demo
