import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cinedub.core import Config, PipelineContext, StageResult, ModelManager, RuntimeMode
from cinedub.checkpoint import CheckpointManager
from cinedub.logging import CineDubLogger


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp, project_name="test_proj")

    def test_project_dir(self):
        expected = os.path.join(self.tmp, "projects", "test_proj")
        self.assertEqual(self.config.project_dir, expected)

    def test_checkpoint_dir(self):
        expected = os.path.join(self.tmp, "checkpoints")
        self.assertEqual(self.config.checkpoint_dir, expected)

    def test_ensure_dirs(self):
        self.config.ensure_dirs()
        self.assertTrue(os.path.isdir(self.config.project_dir))
        self.assertTrue(os.path.isdir(self.config.checkpoint_dir))

    def test_subdir(self):
        sub = self.config.subdir("test_sub")
        expected = os.path.join(self.config.project_dir, "test_sub")
        self.assertEqual(sub, expected)
        self.assertTrue(os.path.isdir(sub))

    def test_subdirs_under_project(self):
        self.config.ensure_dirs()
        paths = ["input", "separated_audio", "transcripts", "cloned_audio",
                 "subtitles", "renders", "final"]
        for p in paths:
            sub = self.config.subdir(p)
            self.assertTrue(sub.startswith(self.config.project_dir))


class TestStageResult(unittest.TestCase):
    def test_success_result(self):
        r = StageResult(True, "test_stage", message="ok",
                        output_files=["a.wav"], metrics={"dur": 1.0})
        self.assertTrue(r.success)
        self.assertEqual(r.stage_name, "test_stage")
        self.assertIn("a.wav", r.output_files)
        self.assertEqual(r.metrics["dur"], 1.0)

    def test_failure_result(self):
        r = StageResult(False, "failed_stage", message="error occurred")
        self.assertFalse(r.success)
        self.assertIn("error", r.message)


class TestPipelineContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp, project_name="ctx_test")

    def test_stage_dir(self):
        ctx = PipelineContext(config=self.config)
        d = ctx.stage_dir("test_stage")
        expected = os.path.join(self.config.project_dir, "test_stage")
        self.assertEqual(d, expected)

    def test_default_values(self):
        ctx = PipelineContext(config=self.config)
        self.assertEqual(ctx.sample_rate, 16000)
        self.assertEqual(ctx.duration_s, 0.0)
        self.assertEqual(ctx.video_path, "")


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp)
        self.cp = CheckpointManager(self.config)

    def test_save_and_exists(self):
        self.cp.save("test_stage", output_files=[])
        self.assertTrue(self.cp.exists("test_stage"))

    def test_load(self):
        self.cp.save("test_stage", output_files=["/tmp/test.wav"],
                     duration_s=10.0, mode=RuntimeMode.GPU_FULL)
        data = self.cp.load("test_stage")
        self.assertIsNotNone(data)
        self.assertEqual(data["stage"], "test_stage")
        self.assertEqual(data["runtime_mode"], "GPU_FULL")

    def test_completed_stages(self):
        self.cp.save("stage_a")
        self.cp.save("stage_b")
        stages = self.cp.completed_stages()
        self.assertIn("stage_a", stages)
        self.assertIn("stage_b", stages)

    def test_latest_stage(self):
        self.cp.save("first")
        self.cp.save("second")
        self.assertEqual(self.cp.latest_stage(), "second")

    def test_resume_from_completed(self):
        stages = ["a", "b", "c", "d"]
        self.cp.save("a")
        self.cp.save("b")
        idx = self.cp.resume_from(stages)
        self.assertEqual(idx, 2)

    def test_resume_from_none(self):
        stages = ["a", "b", "c"]
        idx = self.cp.resume_from(stages)
        self.assertEqual(idx, 0)

    def test_resume_from_all_completed(self):
        stages = ["a", "b", "c"]
        for s in stages:
            self.cp.save(s)
        idx = self.cp.resume_from(stages)
        self.assertEqual(idx, len(stages))

    def test_clear(self):
        self.cp.save("stage")
        self.assertTrue(self.cp.exists("stage"))
        self.cp.clear("stage")
        self.assertFalse(self.cp.exists("stage"))

    def test_clear_all(self):
        self.cp.save("a")
        self.cp.save("b")
        self.cp.clear_all()
        self.assertEqual(len(self.cp.completed_stages()), 0)

    def test_verify_with_file(self):
        tmp_file = os.path.join(self.tmp, "test_asset.wav")
        with open(tmp_file, "w") as f:
            f.write("dummy content")
        self.cp.save("verify_stage", output_files=[tmp_file])
        self.assertTrue(self.cp.verify("verify_stage"))

    def test_verify_missing_file(self):
        self.cp.save("missing", output_files=["/nonexistent/file.wav"])
        self.assertFalse(self.cp.verify("missing"))


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mm = ModelManager.get(self.tmp)
        self.mm.unload_all()

    def test_load_and_unload(self):
        called = []
        def loader(**kw):
            called.append(True)
            return {"loaded": kw.get("val")}

        obj = self.mm.load("test_key", loader, val=42)
        self.assertEqual(obj["loaded"], 42)
        self.assertEqual(len(called), 1)

        cached = self.mm.load("test_key", loader, val=99)
        self.assertEqual(cached["loaded"], 42)
        self.assertEqual(len(called), 1)

        self.mm.unload("test_key")
        self.assertNotIn("test_key", self.mm._registry)

    def test_unload_all(self):
        def loader(**kw):
            return {"v": kw.get("v")}

        self.mm.load("a", loader, v=1)
        self.mm.load("b", loader, v=2)
        self.mm.unload_all()
        self.assertEqual(len(self.mm._registry), 0)


class TestRuntimeMode(unittest.TestCase):
    def test_mode_values(self):
        self.assertEqual(RuntimeMode.GPU_FULL.value, "GPU_FULL")
        self.assertEqual(RuntimeMode.GPU_BALANCED.value, "GPU_BALANCED")
        self.assertEqual(RuntimeMode.CPU_SAFE.value, "CPU_SAFE")
        self.assertEqual(RuntimeMode.RECOVERY.value, "RECOVERY")


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_logger_creates_file(self):
        log = CineDubLogger.get(self.tmp)
        log.info("test message")
        log_path = os.path.join(self.tmp, "pipeline.log")
        self.assertTrue(os.path.isfile(log_path))

    def test_logger_info(self):
        log = CineDubLogger.get(self.tmp)
        log.info("info test")
        log.warn("warn test")
        log.error("error test")
        log_path = os.path.join(self.tmp, "pipeline.log")
        with open(log_path) as f:
            content = f.read()
        self.assertIn("info test", content)
        self.assertIn("warn test", content)
        self.assertIn("error test", content)


class TestStagesManifest(unittest.TestCase):
    def test_all_stages_importable(self):
        from cinedub.stages import (
            BootstrapStage, DownloadStage, NormalizeStage,
            SeparationStage, DiarizationStage, TranscriptionStage,
            AlignmentStage, TranslationStage, SpeakerDatabaseStage,
            VoiceCloningStage, AudioSyncStage, SubtitleStage,
            RenderStage, PipelineOrchestrator,
        )
        self.assertTrue(True)

    def test_stage_hierarchy(self):
        from cinedub.core import Stage
        from cinedub.stages import (
            BootstrapStage, DownloadStage, NormalizeStage,
            SeparationStage, DiarizationStage, TranscriptionStage,
            AlignmentStage, TranslationStage, SpeakerDatabaseStage,
            VoiceCloningStage, AudioSyncStage, SubtitleStage,
            RenderStage,
        )
        for cls in [BootstrapStage, DownloadStage, NormalizeStage,
                    SeparationStage, DiarizationStage, TranscriptionStage,
                    AlignmentStage, TranslationStage, SpeakerDatabaseStage,
                    VoiceCloningStage, AudioSyncStage, SubtitleStage,
                    RenderStage]:
            self.assertTrue(issubclass(cls, Stage))


if __name__ == "__main__":
    unittest.main()
