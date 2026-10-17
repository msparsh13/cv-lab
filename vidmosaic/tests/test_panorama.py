"""Phase 3-4 tests: trajectory accumulation, canvas bounds, warping, blending,
and the end-to-end pipeline, checked against synthetic ground truth."""

import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

from tools.make_demo_video import main as make_demo_main, make_scene
from vidmosaic import cli
from vidmosaic.config import (CanvasConfig, ConfigError, MotionConfig, RenderConfig,
                              SamplingConfig, TrajectoryConfig)
from vidmosaic.mosaic import FeatherBlender
from vidmosaic.pipeline import render_panorama, run_panorama
from vidmosaic.trajectory import (CanvasError, PanoramaError, Trajectory, TrackingLostError,
                                  build_trajectory, compute_canvas, homography_disagreement)
from vidmosaic.video_reader import FrameRecord, VideoReader
from vidmosaic.viz import draw_trajectory
from vidmosaic.warping import warp_to_canvas

W, H = 640, 360
_SCENE = None


def scene():
    global _SCENE
    if _SCENE is None:
        _SCENE = make_scene(1500, 420)
    return _SCENE


def crop(x0, y0, base=None):
    base = scene() if base is None else base
    return base[y0:y0 + H, x0:x0 + W].copy()


def records(images, start_index=0):
    return [FrameRecord(i, start_index + i, (start_index + i) / 25.0, im, 1.0, (im.shape[1], im.shape[0]))
            for i, im in enumerate(images)]


def T(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


# ---------------------------------------------------------------- canvas
class TestCanvas(unittest.TestCase):
    def test_simple_translation_bounds(self):
        c = compute_canvas([np.eye(3), T(100, 0)], (W, H))
        self.assertEqual(c.size, (W + 100, H))
        self.assertEqual(c.origin, (0, 0))
        self.assertEqual(c.scale, 1.0)

    def test_negative_coordinates_are_offset(self):
        c = compute_canvas([T(-250, -40), np.eye(3), T(30, 10)], (W, H))
        self.assertEqual(c.origin, (-250, -40))
        self.assertEqual(c.size, (W + 280, H + 50))
        # every frame must land at non-negative canvas coordinates inside the canvas
        for Hk in (T(-250, -40), np.eye(3), T(30, 10)):
            M = c.C @ Hk
            for corner in ([0, 0, 1], [W - 1, H - 1, 1]):
                p = M @ np.array(corner, float)
                self.assertGreaterEqual(p[0], 0); self.assertGreaterEqual(p[1], 0)
                self.assertLess(p[0], c.size[0]); self.assertLess(p[1], c.size[1])

    def test_rotation_expands_canvas(self):
        a = np.radians(10)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        c = compute_canvas([np.eye(3), R], (W, H))
        self.assertGreater(c.size[1], H)

    def test_exceeding_limits_scales_down_within_limits(self):
        cfg = CanvasConfig(max_pixels=1_000_000, max_width=5000, max_height=5000)
        c = compute_canvas([np.eye(3), T(8000, 0)], (W, H), cfg)    # ~8640x360 = 3.1 Mpx
        self.assertLess(c.scale, 1.0)
        self.assertLessEqual(c.pixels, cfg.max_pixels)
        self.assertLessEqual(c.size[0], cfg.max_width)
        self.assertTrue(c.warnings)
        # the whole content still fits after scaling
        p = c.C @ (T(8000, 0) @ np.array([W - 1, H - 1, 1.0]))
        self.assertLessEqual(p[0] / p[2], c.size[0]); self.assertLessEqual(p[1] / p[2], c.size[1])

    def test_exceeding_limits_errors_when_requested(self):
        cfg = CanvasConfig(max_pixels=1_000_000, on_exceed="error")
        with self.assertRaises(CanvasError) as cm:
            compute_canvas([np.eye(3), T(8000, 0)], (W, H), cfg)
        self.assertIn("exceeding the limits", str(cm.exception))

    def test_absurdly_large_canvas_is_rejected_as_divergence(self):
        with self.assertRaises(CanvasError):
            compute_canvas([np.eye(3), T(5e7, 0)], (W, H))      # > hard_limit
        with self.assertRaises(CanvasError):                     # needs scale < min_scale
            compute_canvas([np.eye(3), T(150000, 0)], (W, H),
                           CanvasConfig(max_width=1000, max_pixels=10_000_000))

    def test_non_finite_rejected(self):
        bad = np.eye(3); bad[0, 0] = np.nan
        with self.assertRaises(CanvasError):
            compute_canvas([np.eye(3), bad], (W, H))

    def test_empty_rejected(self):
        with self.assertRaises(CanvasError):
            compute_canvas([], (W, H))


# ---------------------------------------------------------------- warping
class TestWarping(unittest.TestCase):
    def test_integer_translation_is_exact_with_exact_mask(self):
        img = crop(100, 30)
        wf = warp_to_canvas(img, T(50, 20), (W + 100, H + 40))
        full = np.zeros((H + 40, W + 100, 3), np.uint8)
        valid = np.zeros((H + 40, W + 100), bool)
        full[wf.y0:wf.y0 + wf.mask.shape[0], wf.x0:wf.x0 + wf.mask.shape[1]] = wf.image
        valid[wf.y0:wf.y0 + wf.mask.shape[0], wf.x0:wf.x0 + wf.mask.shape[1]] = wf.mask > 0
        self.assertEqual(int(valid.sum()), W * H)               # exactly the frame area
        self.assertTrue(valid[20:20 + H, 50:50 + W].all())
        np.testing.assert_array_equal(full[20:20 + H, 50:50 + W], img)  # pixels untouched

    def test_no_dark_fringe_at_rotated_edges(self):
        img = np.full((200, 300, 3), (200, 150, 100), np.uint8)
        a = np.radians(12)
        M = np.array([[np.cos(a), -np.sin(a), 150], [np.sin(a), np.cos(a), 40], [0, 0, 1]])
        wf = warp_to_canvas(img, M, (700, 500))
        vals = wf.image[wf.mask > 0].reshape(-1, 3)
        self.assertGreater(len(vals), 40000)
        # every valid pixel keeps the constant colour: no blending with black border
        self.assertTrue((np.abs(vals.astype(int) - [200, 150, 100]) <= 1).all())

    def test_black_source_pixels_are_still_valid(self):
        img = np.zeros((50, 60, 3), np.uint8)                   # genuinely black content
        wf = warp_to_canvas(img, T(10, 10), (100, 100))
        self.assertEqual(int((wf.mask > 0).sum()), 50 * 60)

    def test_footprint_outside_canvas_returns_none(self):
        self.assertIsNone(warp_to_canvas(crop(0, 0), T(5000, 0), (W, H)))

    def test_partially_outside_canvas_is_clipped(self):
        wf = warp_to_canvas(crop(0, 0), T(-300, 0), (W, H))
        self.assertEqual(wf.x0, 0)
        cols = (wf.mask > 0).any(axis=0)
        self.assertEqual(int(cols.sum()), W - 300)               # exactly the part inside

    def test_nonfinite_matrix_raises(self):
        M = np.eye(3); M[0, 2] = np.inf
        with self.assertRaises(ValueError):
            warp_to_canvas(crop(0, 0), M, (W, H))


# ---------------------------------------------------------------- blending
class TestFeather(unittest.TestCase):
    def test_single_observation_preserved_exactly(self):
        img = crop(100, 30)
        b = FeatherBlender((W, H))
        b.add(warp_to_canvas(img, np.eye(3), (W, H)))
        out, valid = b.result()
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(out, img)

    def test_identical_overlap_reproduces_scene_and_uncovered_is_invalid(self):
        s = scene()
        canvas_wh = (W + 200, H)
        b = FeatherBlender(canvas_wh)
        for dx in (0, 100, 200):
            b.add(warp_to_canvas(crop(100 + dx, 30), T(dx, 0), canvas_wh))
        out, valid = b.result()
        self.assertTrue(valid.all())
        expected = s[30:30 + H, 100:100 + W + 200]
        self.assertLessEqual(np.abs(out.astype(int) - expected.astype(int)).max(), 1)

    def test_uncovered_pixels_are_masked_not_black_content(self):
        b = FeatherBlender((W + 300, H))
        b.add(warp_to_canvas(crop(0, 0), np.eye(3), (W + 300, H)))
        out, valid = b.result()
        self.assertFalse(valid[:, W + 10:].any())
        self.assertTrue(valid[:, :W].all())

    def test_weights_fall_towards_the_frame_border(self):
        # two frames with different constant colours: the blend must transition smoothly
        a = np.full((100, 200, 3), 0, np.uint8)
        c = np.full((100, 200, 3), 200, np.uint8)
        b = FeatherBlender((300, 100))
        b.add(warp_to_canvas(a, np.eye(3), (300, 100)))
        b.add(warp_to_canvas(c, T(100, 0), (300, 100)))
        out, _ = b.result()
        row = out[50, :, 0].astype(int)
        self.assertEqual(row[10], 0); self.assertEqual(row[290], 200)
        ov = row[105:195]
        self.assertTrue((np.diff(ov) >= 0).all(), "blend must be monotonic across the overlap")
        self.assertTrue(10 < row[150] < 190)

    def test_memory_estimate(self):
        self.assertEqual(FeatherBlender.bytes_needed((1000, 500)), 8_000_000)


# ---------------------------------------------------------------- trajectory
class TestTrajectory(unittest.TestCase):
    def build(self, offsets, **tkw):
        imgs = [crop(x, y) for x, y in offsets]
        return build_trajectory(records(imgs), MotionConfig(), TrajectoryConfig(**tkw))

    def test_translations_accumulate_with_correct_direction_and_order(self):
        offs = [(100, 30), (120, 34), (140, 30), (160, 26), (180, 30), (200, 36), (220, 30)]
        tr = self.build(offs, reference="first", keyframe_shift=0.05)
        self.assertEqual(len(tr.entries), 7)
        for e, (x, y) in zip(tr.entries, offs):
            self.assertAlmostEqual(e.H[0, 2], x - 100, delta=0.5)
            self.assertAlmostEqual(e.H[1, 2], y - 30, delta=0.5)
        self.assertTrue(np.allclose(tr.entries[0].H, np.eye(3)))

    def test_middle_reference_is_identity_and_others_relative(self):
        offs = [(100 + 25 * k, 30) for k in range(9)]
        tr = self.build(offs, reference="middle", keyframe_shift=0.05)
        ref = next(e for e in tr.entries if e.frame_index == tr.reference_frame_index)
        np.testing.assert_allclose(ref.H, np.eye(3), atol=1e-9)
        x_ref = offs[ref.frame_index][0]
        for e in tr.entries:
            self.assertAlmostEqual(e.H[0, 2], offs[e.frame_index][0] - x_ref, delta=0.6)
        # entries before the reference sit at negative coordinates
        self.assertLess(tr.entries[0].H[0, 2], 0)

    def test_leftward_pan_negative_coordinates_end_to_end(self):
        offs = [(600 - 25 * k, 30) for k in range(9)]
        tr = self.build(offs, reference="first", keyframe_shift=0.05)
        self.assertLess(tr.entries[-1].H[0, 2], -150)
        c = compute_canvas([e.H for e in tr.entries], tr.frame_size)
        self.assertEqual(c.origin[0], int(np.floor(tr.entries[-1].H[0, 2])))
        self.assertGreater(c.size[0], W + 190)

    def test_keyframe_selection_by_shift(self):
        offs = [(100 + 10 * k, 30) for k in range(30)]          # 10 px/frame
        tr = self.build(offs, reference="first", keyframe_shift=0.2)   # 128 px
        keys = tr.keyframes()
        self.assertTrue(2 <= len(keys) < 12)
        self.assertEqual(keys[0].frame_index, 0)
        self.assertEqual(keys[-1].frame_index, tr.entries[-1].frame_index)  # tail covered
        gaps = np.diff([e.H[0, 2] for e in keys])
        self.assertTrue((gaps[:-1] >= 0.2 * W - 1).all())

    def test_fast_pan_recovers_by_promoting_candidate(self):
        # 130 px/frame with a keyframe threshold above that: the far frame cannot be matched
        # to the keyframe once overlap is gone, so tracking must continue via promotion.
        offs = [(20 + 130 * k, 30) for k in range(7)]
        tr = self.build(offs, reference="first", keyframe_shift=0.9)
        self.assertTrue(tr.complete)
        self.assertEqual(len(tr.entries), 7)
        for e, (x, _) in zip(tr.entries, offs):
            self.assertAlmostEqual(e.H[0, 2], x - 20, delta=1.0)

    def test_drift_diagnostics_are_small_on_clean_data(self):
        offs = [(50 + 30 * k, 30) for k in range(24)]            # stays inside the 1500px scene
        tr = self.build(offs, reference="first", keyframe_shift=0.15)
        self.assertGreater(len(tr.consistency), 2)
        self.assertLess(max(c["disagreement_px"] for c in tr.consistency), 1.0)
        self.assertEqual(tr.warnings, [])

    def test_drift_warning_triggers_on_threshold(self):
        offs = [(50 + 30 * k, 30) for k in range(24)]
        tr = self.build(offs, reference="first", keyframe_shift=0.15, drift_warn_px=1e-6)
        self.assertTrue(any("possible drift" in w for w in tr.warnings))

    def test_homography_disagreement_metric(self):
        self.assertAlmostEqual(homography_disagreement(T(5, 0), T(5, 0), (W, H)), 0, places=9)
        self.assertAlmostEqual(homography_disagreement(T(5, 0), T(8, 4), (W, H)), 5.0, places=9)

    def test_hard_cut_truncates_with_warning(self):
        other = make_scene(1500, 420, seed=5)
        imgs = [crop(100 + 20 * k, 30) for k in range(6)] + \
               [crop(100 + 20 * k, 30, other) for k in range(12)]
        tr = build_trajectory(records(imgs), MotionConfig(),
                              TrajectoryConfig(reference="first", max_consecutive_failures=3))
        self.assertFalse(tr.complete)
        self.assertEqual(tr.lost_at, 6 + 3)
        self.assertEqual(len(tr.entries), 6)
        self.assertEqual(len(tr.rejected), 4)
        self.assertTrue(any("tracking lost" in w for w in tr.warnings))

    def test_hard_cut_strict_raises(self):
        other = make_scene(1500, 420, seed=5)
        imgs = [crop(100 + 20 * k, 30) for k in range(5)] + [crop(100 + 20 * k, 30, other) for k in range(10)]
        with self.assertRaises(TrackingLostError):
            build_trajectory(records(imgs), MotionConfig(),
                             TrajectoryConfig(on_lost="error", max_consecutive_failures=2))

    def test_isolated_bad_frame_is_skipped_not_fatal(self):
        imgs = [crop(100 + 20 * k, 30) for k in range(10)]
        imgs[4] = np.full((H, W, 3), 128, np.uint8)             # one blank frame
        tr = build_trajectory(records(imgs), MotionConfig(), TrajectoryConfig(reference="first"))
        self.assertTrue(tr.complete)
        self.assertEqual(len(tr.rejected), 1)
        self.assertEqual(tr.rejected[0]["frame_index"], 4)
        self.assertEqual(len(tr.entries), 9)
        self.assertAlmostEqual(tr.entries[-1].H[0, 2], 180, delta=1.0)   # chain unaffected

    def test_too_few_aligned_frames_raises(self):
        blank = np.full((H, W, 3), 90, np.uint8)
        with self.assertRaises(PanoramaError):
            build_trajectory(records([crop(0, 0), blank, blank]), MotionConfig(),
                             TrajectoryConfig(max_consecutive_failures=10))
        with self.assertRaises(PanoramaError):
            build_trajectory(records([crop(0, 0)]))
        with self.assertRaises(PanoramaError):
            build_trajectory(records([]))

    def test_frame_size_change_rejected(self):
        recs = records([crop(0, 0), crop(20, 0)])
        recs.append(FrameRecord(2, 2, 0.1, np.zeros((100, 100, 3), np.uint8), 1.0, (100, 100)))
        with self.assertRaises(PanoramaError):
            build_trajectory(recs)

    def test_save_load_roundtrip(self):
        offs = [(100 + 25 * k, 30) for k in range(8)]
        tr = self.build(offs)
        tr.meta["sampling_config"] = SamplingConfig().to_dict()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.json")
            tr.save(p)
            back = Trajectory.load(p)
        self.assertEqual(back.reference_frame_index, tr.reference_frame_index)
        self.assertEqual(back.frame_size, tr.frame_size)
        for a, b in zip(tr.entries, back.entries):
            np.testing.assert_allclose(a.H, b.H)
            self.assertEqual(a.is_key, b.is_key)

    def test_trajectory_diagnostic_image(self):
        tr = self.build([(100 + 25 * k, 30 + (k % 3)) for k in range(8)])
        img = draw_trajectory(tr)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[1], 200)

    def test_config_validation(self):
        for kw in [dict(keyframe_shift=0), dict(keyframe_shift=1), dict(on_lost="x"),
                   dict(reference="last"), dict(max_consecutive_failures=-1), dict(drift_warn_px=0)]:
            with self.subTest(kw=kw), self.assertRaises(ConfigError):
                TrajectoryConfig(**kw).validate()
        for kw in [dict(max_pixels=10), dict(on_exceed="x"), dict(min_scale=0), dict(max_width=4)]:
            with self.subTest(kw=kw), self.assertRaises(ConfigError):
                CanvasConfig(**kw).validate()
        with self.assertRaises(ConfigError):
            RenderConfig(use="some").validate()


# ---------------------------------------------------------------- end to end
class TestEndToEnd(unittest.TestCase):
    """Synthetic panning video (no car, with sway and exposure drift) vs the true scene."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = os.path.join(cls.tmp.name, "demo")
        argv = sys.argv
        sys.argv = ["x", "--out", cls.base, "--frames", "60", "--pan-px", "6", "--no-car"]
        try:
            make_demo_main()
        finally:
            sys.argv = argv
        cls.video = cls.base + ".mp4"
        cls.truth = json.load(open(cls.base + "_truth.json"))["frames"]
        cls.scene = cv2.imread(cls.base + "_scene.png")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    _shared = None

    def shared(self):
        """One default pipeline run reused by several tests (they only read from it)."""
        cls = type(self)
        if cls._shared is None:
            cls._shared = self.run_pipeline("shared")
        return cls._shared

    def run_pipeline(self, name, **over):
        out = os.path.join(self.tmp.name, name)
        kw = dict(scfg=SamplingConfig(frame_step=2, max_dim=None), mcfg=MotionConfig(),
                  tcfg=TrajectoryConfig(), ccfg=CanvasConfig(), rcfg=RenderConfig())
        kw.update(over)
        report = run_panorama(self.video, out, log=None, **kw)
        return out, report

    def test_panorama_matches_ground_truth_scene(self):
        out, rep = self.shared()
        self.assertTrue(rep["complete"])
        pano = cv2.imread(os.path.join(out, "panorama.png"))
        traj = Trajectory.load(os.path.join(out, "trajectory.json"))
        r = self.truth[traj.reference_frame_index]

        # 1. camera positions of all frames vs truth (chained, relative to reference)
        for e in traj.entries:
            t = self.truth[e.frame_index]
            self.assertAlmostEqual(e.H[0, 2], t["x0"] - r["x0"], delta=0.6)
            self.assertAlmostEqual(e.H[1, 2], t["y0"] - r["y0"], delta=0.6)

        # 2. canvas size equals the span covered by the camera
        xs = [self.truth[e.frame_index]["x0"] for e in traj.keyframes()]
        ys = [self.truth[e.frame_index]["y0"] for e in traj.keyframes()]
        self.assertAlmostEqual(rep["canvas"]["size"][0], max(xs) - min(xs) + W, delta=3)
        self.assertAlmostEqual(rep["canvas"]["size"][1], max(ys) - min(ys) + H, delta=3)

        # 3. content: panorama pixel (u, v) shows scene point (u + ox + x0_ref, v + oy + y0_ref)
        ox, oy = rep["canvas"]["origin"]
        M = np.float32([[1, 0, -(ox + r["x0"])], [0, 1, -(oy + r["y0"])]])
        size = tuple(rep["canvas"]["size"])
        expected = cv2.warpAffine(self.scene, M, size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        valid = pano.sum(axis=2) > 0
        g1 = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sl = (slice(0, min(300, size[1])), slice(50, size[0] - 50))
        (sx, sy), resp = cv2.phaseCorrelate(g1[sl], g2[sl])
        self.assertLess(abs(sx), 0.5); self.assertLess(abs(sy), 0.5)   # geometric accuracy
        self.assertGreater(resp, 0.9)
        d = np.abs(pano.astype(float) - expected.astype(float)).mean(axis=2)
        self.assertLess(np.median(d[valid]), 8.0)    # exposure drift (+-6%) allows a few levels

        # 4. coverage and seam quality
        self.assertGreater(rep["coverage"], 0.9)
        self.assertLess(rep["trajectory"]["drift_max_px"], 2.0)

    def test_outputs_and_report_contents(self):
        out, rep = self.shared()
        for f in ("panorama.png", "trajectory.png", "trajectory.json", "report.json"):
            self.assertTrue(os.path.getsize(os.path.join(out, f)) > 0, f)
        saved = json.load(open(os.path.join(out, "report.json")))
        for key in ("configs", "canvas", "coverage", "accumulator_bytes", "seconds", "trajectory"):
            self.assertIn(key, saved)
        self.assertEqual(rep["accumulator_bytes"],
                         rep["canvas"]["size"][0] * rep["canvas"]["size"][1] * 16)
        self.assertFalse(rep["trajectory"]["reused_cache"])

    def test_cache_reuse_and_invalidation(self):
        out, _ = self.run_pipeline("cache")
        _, rep2 = self.run_pipeline("cache", reuse_trajectory=True)
        self.assertTrue(rep2["trajectory"]["reused_cache"])
        # changed settings must not silently reuse the old trajectory
        _, rep3 = self.run_pipeline("cache", reuse_trajectory=True,
                                    scfg=SamplingConfig(frame_step=3, max_dim=None))
        self.assertFalse(rep3["trajectory"]["reused_cache"])

    def test_canvas_limit_scales_panorama(self):
        out, rep = self.run_pipeline("small", ccfg=CanvasConfig(max_pixels=200_000))
        self.assertLessEqual(rep["canvas"]["pixels"], 200_000)
        self.assertLess(rep["canvas"]["scale"], 1.0)
        self.assertTrue(any("scaled" in w for w in rep["warnings"]))
        img = cv2.imread(os.path.join(out, "panorama.png"))
        self.assertEqual((img.shape[1], img.shape[0]), tuple(rep["canvas"]["size"]))
        self.assertGreater(rep["coverage"], 0.9)

    def test_canvas_limit_error_mode(self):
        with self.assertRaises(CanvasError):
            self.run_pipeline("err", ccfg=CanvasConfig(max_pixels=200_000, on_exceed="error"))

    def test_use_all_frames_gives_same_geometry(self):
        _, a = self.shared()
        _, b = self.run_pipeline("all", rcfg=RenderConfig(use="all"))
        self.assertGreater(b["frames_blended"], a["frames_blended"])
        self.assertAlmostEqual(a["canvas"]["size"][0], b["canvas"]["size"][0], delta=3)

    def test_resized_video_gives_proportional_panorama(self):
        _, full = self.shared()
        _, half = self.run_pipeline("half", scfg=SamplingConfig(frame_step=2, max_dim=320))
        self.assertAlmostEqual(half["canvas"]["size"][0] * 2, full["canvas"]["size"][0], delta=8)

    def test_cli_panorama_command(self):
        out = os.path.join(self.tmp.name, "cli")
        rc = cli.main(["panorama", self.video, "--step", "3", "--max-dim", "480",
                       "--out-dir", out, "--name", "wide.png"])
        self.assertEqual(rc, 0)
        img = cv2.imread(os.path.join(out, "wide.png"))
        self.assertIsNotNone(img)
        self.assertGreater(img.shape[1], 2 * img.shape[0])      # clearly panoramic
        rc = cli.main(["panorama", self.video, "--step", "3", "--max-dim", "480",
                       "--out-dir", out, "--name", "wide.png", "--reuse-trajectory"])
        self.assertEqual(rc, 0)

    def test_cli_partial_panorama_returns_nonzero(self):
        other = make_scene(1500, 420, seed=5)
        path = os.path.join(self.tmp.name, "cut.avi")
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 25, (W, H))
        for k in range(8):
            wr.write(crop(100 + 20 * k, 30))
        for k in range(12):
            wr.write(crop(100 + 20 * k, 30, other))
        wr.release()
        out = os.path.join(self.tmp.name, "cut_out")
        rc = cli.main(["panorama", path, "--max-dim", "640", "--out-dir", out, "--max-failures", "2"])
        self.assertEqual(rc, 1)
        rep = json.load(open(os.path.join(out, "report.json")))
        self.assertFalse(rep["complete"])
        self.assertTrue(any("tracking lost" in w for w in rep["warnings"]))
        self.assertTrue(os.path.getsize(os.path.join(out, "panorama.png")) > 0)
        rc = cli.main(["panorama", path, "--max-dim", "640", "--out-dir", out + "2",
                       "--max-failures", "2", "--strict"])
        self.assertEqual(rc, 2)

    def test_cli_reports_missing_video(self):
        self.assertEqual(cli.main(["panorama", "/no/such.mp4", "--out-dir", self.tmp.name + "/x"]), 2)


if __name__ == "__main__":
    unittest.main()
