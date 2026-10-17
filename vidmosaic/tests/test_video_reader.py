import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from tests.helpers import decode_index, write_index_video
from vidmosaic import cli
from vidmosaic.config import ConfigError, SamplingConfig
from vidmosaic.video_reader import (
    VideoOpenError, VideoReader, VideoReadError, resize_to_max_dim,
)
from vidmosaic.viz import make_contact_sheet

FPS = 25.0
N = 60


class VideoTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "idx.avi")
        write_index_video(cls.path, n_frames=N, fps=FPS, size=(200, 100))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, **kw):
        kw.setdefault("max_dim", None)
        with VideoReader(self.path, SamplingConfig(**kw)) as vr:
            return vr.read_all()


class TestMetadata(VideoTestCase):
    def test_info(self):
        with VideoReader(self.path) as vr:
            self.assertEqual((vr.info.width, vr.info.height), (200, 100))
            self.assertAlmostEqual(vr.info.fps, FPS)
            self.assertEqual(vr.info.frame_count, N)
            self.assertAlmostEqual(vr.info.duration_s, N / FPS)


class TestSampling(VideoTestCase):
    def test_every_frame_content_and_timestamps(self):
        recs = self.read()
        self.assertEqual(len(recs), N)
        for k, r in enumerate(recs):
            self.assertEqual(r.sample_index, k)
            self.assertEqual(r.frame_index, k)
            # Pixel content proves the frame really is source frame k.
            self.assertEqual(decode_index(r.image), k)
            self.assertAlmostEqual(r.timestamp_s, k / FPS, places=9)

    def test_frame_step(self):
        recs = self.read(frame_step=7)
        self.assertEqual([r.frame_index for r in recs], list(range(0, N, 7)))
        self.assertEqual([r.sample_index for r in recs], list(range(len(recs))))
        for r in recs:
            self.assertEqual(decode_index(r.image), r.frame_index)
            self.assertAlmostEqual(r.timestamp_s, r.frame_index / FPS)

    def test_target_fps_converts_to_step(self):
        recs = self.read(target_fps=5.0)           # 25 / 5 = step 5
        self.assertEqual([r.frame_index for r in recs], list(range(0, N, 5)))

    def test_target_fps_higher_than_video_means_every_frame(self):
        recs = self.read(target_fps=1000.0)
        self.assertEqual(len(recs), N)

    def test_range_is_start_inclusive_end_exclusive(self):
        recs = self.read(start_frame=10, end_frame=20)
        self.assertEqual([r.frame_index for r in recs], list(range(10, 20)))

    def test_step_grid_is_anchored_at_start_frame(self):
        recs = self.read(start_frame=4, frame_step=10)
        self.assertEqual([r.frame_index for r in recs], [4, 14, 24, 34, 44, 54])

    def test_max_frames(self):
        recs = self.read(frame_step=2, max_frames=5)
        self.assertEqual([r.frame_index for r in recs], [0, 2, 4, 6, 8])

    def test_second_iteration_restarts(self):
        with VideoReader(self.path, SamplingConfig(frame_step=10, max_dim=None)) as vr:
            a = [r.frame_index for r in vr]
            b = [r.frame_index for r in vr]
        self.assertEqual(a, b)
        self.assertEqual(a, [0, 10, 20, 30, 40, 50])

    def test_range_beyond_video_raises_in_read_all(self):
        with self.assertRaises(VideoReadError):
            self.read(start_frame=1000)


class TestResize(VideoTestCase):
    def test_downscale_keeps_aspect_and_reports_scale(self):
        recs = self.read(max_dim=100)              # 200x100 -> 100x50
        r = recs[0]
        self.assertEqual(r.size, (100, 50))
        self.assertEqual(r.original_size, (200, 100))
        self.assertAlmostEqual(r.scale, 0.5)
        self.assertEqual(decode_index(recs[17].image), 17)  # content survives

    def test_never_upscales(self):
        r = self.read(max_dim=4000)[0]
        self.assertEqual(r.size, (200, 100))
        self.assertEqual(r.scale, 1.0)

    def test_resize_function_odd_sizes(self):
        img = np.zeros((101, 303, 3), np.uint8)
        out, s = resize_to_max_dim(img, 100)
        self.assertEqual(max(out.shape[:2]), 100)
        self.assertAlmostEqual(s, out.shape[1] / 303.0)

    def test_none_keeps_original(self):
        img = np.zeros((10, 20, 3), np.uint8)
        out, s = resize_to_max_dim(img, None)
        self.assertIs(out, img)
        self.assertEqual(s, 1.0)


class TestConfigValidation(unittest.TestCase):
    def test_conflicting_sampling(self):
        with self.assertRaises(ConfigError):
            SamplingConfig(frame_step=2, target_fps=5).validate()

    def test_bad_values(self):
        for kw in [dict(frame_step=0), dict(target_fps=0), dict(start_frame=-1),
                   dict(start_frame=5, end_frame=5), dict(max_frames=0),
                   dict(max_dim=4)]:
            with self.subTest(kw=kw), self.assertRaises(ConfigError):
                SamplingConfig(**kw).validate()

    def test_roundtrip_for_reproducibility(self):
        c = SamplingConfig(target_fps=3.0, max_dim=640, max_frames=9)
        self.assertEqual(SamplingConfig.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_reader_rejects_invalid_config_early(self):
        with self.assertRaises(ConfigError):
            VideoReader("whatever.avi", SamplingConfig(frame_step=-3))


class TestErrors(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(VideoOpenError):
            VideoReader("/nonexistent/video.mp4")

    def test_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"this is not a video")
            p = f.name
        try:
            with self.assertRaises(VideoOpenError):
                VideoReader(p)
        finally:
            os.unlink(p)


class TestVizAndCli(VideoTestCase):
    def test_contact_sheet_shape(self):
        recs = self.read(frame_step=5)             # 12 frames
        sheet = make_contact_sheet(recs, columns=4, thumb_width=100)
        self.assertEqual(sheet.shape, (3 * 50, 4 * 100, 3))

    def test_contact_sheet_subsamples_when_many(self):
        recs = self.read()
        sheet = make_contact_sheet(recs, columns=5, thumb_width=40, max_thumbs=10)
        self.assertEqual(sheet.shape[1], 5 * 40)
        self.assertEqual(sheet.shape[0], 2 * 20)

    def test_cli_info_and_preview(self):
        out = os.path.join(self.tmp.name, "sheet.png")
        dump = os.path.join(self.tmp.name, "frames")
        self.assertEqual(cli.main(["info", self.path]), 0)
        rc = cli.main(["preview", self.path, "--step", "20", "--max-dim", "100",
                       "--out", out, "--dump-dir", dump])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.getsize(out) > 0)
        self.assertEqual(sorted(os.listdir(dump)),
                         ["frame_000000.png", "frame_000020.png", "frame_000040.png"])
        img = cv2.imread(os.path.join(dump, "frame_000020.png"))
        self.assertEqual(img.shape[:2], (50, 100))
        self.assertEqual(decode_index(img), 20)

    def test_cli_reports_errors_with_nonzero_exit(self):
        self.assertEqual(cli.main(["info", "/no/such/file.mp4"]), 2)
        self.assertEqual(cli.main(["preview", self.path, "--start-frame", "999",
                                   "--out", os.path.join(self.tmp.name, "x.png")]), 2)


if __name__ == "__main__":
    unittest.main()
