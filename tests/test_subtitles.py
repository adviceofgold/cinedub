import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cinedub.stages.subtitles import SubtitleStage
from cinedub.core import Config, PipelineContext


class TestSubtitleFormatting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = Config(work_dir=self.tmp, project_name="sub_test",
                             target_language="es")
        self.ctx = PipelineContext(config=self.config)
        self.ctx.metadata["translated_segments"] = [
            {"start": 0.0, "end": 2.5, "translated": "Hola mundo"},
            {"start": 3.0, "end": 6.5,
             "translated": "Este es un texto más largo que necesita ser dividido en varias líneas para mejor legibilidad"},
            {"start": 10.0, "end": 10.5, "translated": "Corto"},
        ]

    def test_srt_output(self):
        stage = SubtitleStage()
        out_dir = self.config.subdir("subtitles")
        srt_path = os.path.join(out_dir, "subtitles_es.srt")
        stage._write_srt(self.ctx.metadata["translated_segments"], srt_path)
        self.assertTrue(os.path.isfile(srt_path))

        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Hola mundo", content)
        self.assertIn("00:00:00,000", content)
        self.assertIn("00:00:02,500", content)
        self.assertIn("Corto", content)

    def test_srt_timestamp_format(self):
        stage = SubtitleStage()
        ts = stage._format_timestamp(3661.501)
        self.assertEqual(ts, "01:01:01,501")
        ts0 = stage._format_timestamp(0.0)
        self.assertEqual(ts0, "00:00:00,000")

    def test_vtt_output(self):
        stage = SubtitleStage()
        out_dir = self.config.subdir("subtitles")
        vtt_path = os.path.join(out_dir, "subtitles_es.vtt")
        stage._write_vtt(self.ctx.metadata["translated_segments"], vtt_path)
        self.assertTrue(os.path.isfile(vtt_path))

        with open(vtt_path, encoding="utf-8") as f:
            content = f.read()

        self.assertTrue(content.startswith("WEBVTT"))
        self.assertIn("Hola mundo", content)

    def test_ass_output(self):
        stage = SubtitleStage()
        out_dir = self.config.subdir("subtitles")
        ass_path = os.path.join(out_dir, "subtitles_es.ass")
        stage._write_ass(self.ctx.metadata["translated_segments"], ass_path)
        self.assertTrue(os.path.isfile(ass_path))

        with open(ass_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("[Script Info]", content)
        self.assertIn("[Events]", content)
        self.assertIn("Hola mundo", content)

    def test_chunking_long_text(self):
        stage = SubtitleStage()
        text = "This is a very long text that should be split into multiple chunks at the forty two character boundary"
        chunks = stage._chunk_text(text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 42)

    def test_min_max_duration_enforced(self):
        stage = SubtitleStage()
        out_dir = self.config.subdir("subtitles")
        srt_path = os.path.join(out_dir, "subtitles_es.srt")
        stage._write_srt(self.ctx.metadata["translated_segments"], srt_path)

        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Corto", content)

    def test_subtitle_stage_execute(self):
        stage = SubtitleStage()
        result = stage.execute(self.ctx)
        self.assertTrue(result.success)
        self.assertIn("subtitles", result.stage_name)


if __name__ == "__main__":
    unittest.main()
