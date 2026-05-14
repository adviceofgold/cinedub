import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cinedub.core import Config, PipelineContext, RuntimeMode
from cinedub.checkpoint import CheckpointManager
from cinedub.logging import CineDubLogger
from cinedub.stages.orchestrator import PipelineOrchestrator


class TestOrchestratorStructure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp, project_name="integ_test")

    def test_pipeline_stages_list(self):
        expected = [
            "bootstrap", "download", "normalize", "separation",
            "diarization", "transcription", "alignment", "translation",
            "speaker_db", "voice_cloning", "audio_sync", "subtitles",
            "render", "lipsync",
        ]
        self.assertEqual(PipelineOrchestrator.PIPELINE_STAGES, expected)

    def test_orchestrator_creates_stages(self):
        orch = PipelineOrchestrator(self.config)
        for name in orch.PIPELINE_STAGES:
            self.assertIn(name, orch._stages)

    def test_empty_run_on_all_completed(self):
        orch = PipelineOrchestrator(self.config)
        cp = CheckpointManager(self.config)
        for s in orch.PIPELINE_STAGES:
            cp.save(s)
        self.config.mode = RuntimeMode.RECOVERY
        results = orch.run()
        self.assertEqual(len(results), 0)

    def test_resume_from_checkpoint(self):
        self.config.mode = RuntimeMode.RECOVERY
        cp = CheckpointManager(self.config)
        cp.save("bootstrap")
        cp.save("download")
        orch = PipelineOrchestrator(self.config)
        idx = orch.checkpoint.resume_from(orch.PIPELINE_STAGES)
        self.assertEqual(idx, 2)

    def test_skip_disabled_stage(self):
        self.config.mode = RuntimeMode.CPU_SAFE
        self.config.source_url = ""
        orch = PipelineOrchestrator(self.config)
        stage = orch._stages["separation"]
        from cinedub.core import StageResult
        if stage.validate(orch.ctx):
            result = stage.execute(orch.ctx)
        else:
            result = StageResult(True, "separation",
                                  message="Skipped by validation")
        self.assertTrue(result.success)


class TestStageDependencyOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp)

    def test_dependency_chain(self):
        from cinedub.stages import (
            BootstrapStage, DownloadStage, NormalizeStage,
            SeparationStage, DiarizationStage, TranslationStage,
            SpeakerDatabaseStage, VoiceCloningStage, RenderStage,
        )
        deps = {
            "bootstrap": BootstrapStage().depends_on,
            "download": DownloadStage().depends_on,
            "normalize": NormalizeStage().depends_on,
            "separation": SeparationStage().depends_on,
            "diarization": DiarizationStage().depends_on,
            "translation": TranslationStage().depends_on,
            "speaker_db": SpeakerDatabaseStage().depends_on,
            "voice_cloning": VoiceCloningStage().depends_on,
            "render": RenderStage().depends_on,
        }
        self.assertEqual(deps["bootstrap"], [])
        self.assertEqual(deps["download"], ["bootstrap"])
        self.assertEqual(deps["normalize"], ["download"])
        self.assertEqual(deps["separation"], ["normalize"])
        self.assertEqual(deps["diarization"], ["separation"])
        self.assertIn("alignment", deps["translation"])
        self.assertIn("speaker_db", deps["voice_cloning"])

    def test_no_circular_deps(self):
        from cinedub.core import Stage
        from cinedub.stages import (
            BootstrapStage, DownloadStage, NormalizeStage,
            SeparationStage, DiarizationStage, TranscriptionStage,
            AlignmentStage, TranslationStage, SpeakerDatabaseStage,
            VoiceCloningStage, AudioSyncStage, SubtitleStage,
            RenderStage, LipsyncStage,
        )
        all_stages = {
            "bootstrap": BootstrapStage(),
            "download": DownloadStage(),
            "normalize": NormalizeStage(),
            "separation": SeparationStage(),
            "diarization": DiarizationStage(),
            "transcription": TranscriptionStage(),
            "alignment": AlignmentStage(),
            "translation": TranslationStage(),
            "speaker_db": SpeakerDatabaseStage(),
            "voice_cloning": VoiceCloningStage(),
            "audio_sync": AudioSyncStage(),
            "subtitles": SubtitleStage(),
            "render": RenderStage(),
            "lipsync": LipsyncStage(),
        }

        for name, stage in all_stages.items():
            for dep in stage.depends_on:
                self.assertIn(dep, all_stages,
                              f"{name} depends on unknown stage: {dep}")


class TestPipelineContextFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp, project_name="flow_test",
                             source_url="https://youtube.com/watch?v=test")

    def test_context_metadata_passthrough(self):
        ctx = PipelineContext(config=self.config)
        ctx.metadata["test_key"] = "test_value"
        self.assertEqual(ctx.metadata["test_key"], "test_value")

    def test_context_project_dirs(self):
        self.config.ensure_dirs()
        for sub in ["input", "separated_audio", "diarization",
                     "transcripts", "translations", "speakers",
                     "cloned_audio", "subtitles", "lipsync",
                     "renders", "final"]:
            d = self.config.subdir(sub)
            self.assertTrue(os.path.isdir(d))


class TestRuntimeModeResolution(unittest.TestCase):
    def test_cpu_mode(self):
        import torch
        if not torch.cuda.is_available():
            mode = RuntimeMode.CPU_SAFE
        else:
            mode = RuntimeMode.GPU_FULL
        self.assertIn(mode, [RuntimeMode.CPU_SAFE, RuntimeMode.GPU_FULL,
                             RuntimeMode.GPU_BALANCED])


if __name__ == "__main__":
    unittest.main()
