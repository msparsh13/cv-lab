"""Phase 2 tests. Correctness is judged against known ground truth:
synthetic crops with exact translations, a known general homography, and the
demo video's truth file. Tests assert geometric accuracy, not just completion."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from tests.helpers import write_index_video
from tools.make_demo_video import draw_car, main as make_demo_main, make_scene
from vidmosaic import cli
from vidmosaic import motion_estimation as me
from vidmosaic.config import ConfigError, MotionConfig, SamplingConfig
from vidmosaic.video_reader import VideoReader
from vidmosaic.viz import alignment_overlay, draw_matches

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


def align(ref, mov, cfg=None):
    cfg = cfg or MotionConfig()
    return me.estimate_pair(me.detect_features(ref, cfg), me.detect_features(mov, cfg), cfg)


def corner_error(H_est, H_true, w=W, h=H, margin=40):
    """Max distance (px) between where two homographies send a grid of points."""
    xs = np.linspace(margin, w - margin, 6)
    ys = np.linspace(margin, h - margin, 5)
    pts = np.array([[x, y] for x in xs for y in ys], np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_est.astype(np.float64))
    b = cv2.perspectiveTransform(pts, H_true.astype(np.float64))
    return float(np.linalg.norm(a - b, axis=2).max())


def translation(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


class TestKnownTransforms(unittest.TestCase):
    def test_integer_translation_recovered_accurately(self):
        # mov window starts dx/dy further into the scene => p_ref = p_mov + (dx, dy)
        ref, mov = crop(200, 30), crop(200 + 37, 30 + 9)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, translation(37, 9)), 0.3)
        self.assertAlmostEqual(r.metrics.shift_x, 37, delta=0.3)
        self.assertAlmostEqual(r.metrics.shift_y, 9, delta=0.3)
        self.assertLess(r.metrics.rms_error, 1.0)
        self.assertGreater(r.metrics.n_inliers, 100)

    def test_subpixel_translation(self):
        ref = crop(200, 30)
        M = np.float32([[1, 0, -(200 + 7.4)], [0, 1, -(30 + 2.6)]])
        mov = cv2.warpAffine(scene(), M, (W, H), flags=cv2.INTER_LINEAR)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, translation(7.4, 2.6)), 0.4)

    def test_general_homography_recovered(self):
        ref = crop(300, 30)
        th, s = np.radians(2.0), 1.03
        H_true = np.array([[s * np.cos(th), -s * np.sin(th), 25.0],
                           [s * np.sin(th),  s * np.cos(th), 10.0],
                           [2e-5, 1e-5, 1.0]])
        # mov(p) = ref(H_true p)  =>  H_true maps mov -> ref
        mov = cv2.warpPerspective(ref, np.linalg.inv(H_true), (W, H), flags=cv2.INTER_LINEAR)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, H_true), 1.0)
        self.assertAlmostEqual(r.metrics.rotation_deg, 2.0, delta=0.3)
        self.assertAlmostEqual(r.metrics.scale, 1.03, delta=0.02)

    def test_direction_convention_is_invertible(self):
        a, b = crop(200, 30), crop(240, 35)
        r_ab = align(a, b)      # maps b -> a
        r_ba = align(b, a)      # maps a -> b
        self.assertTrue(r_ab.ok and r_ba.ok)
        self.assertLess(corner_error(r_ab.H @ r_ba.H, np.eye(3)), 0.5)
        self.assertGreater(r_ab.metrics.shift_x, 0)     # b is to the right of a
        self.assertLess(r_ba.metrics.shift_x, 0)

    def test_composition_matches_ground_truth_over_three_frames(self):
        f0, f1, f2 = crop(100, 30), crop(150, 36), crop(190, 28)
        r01, r12 = align(f0, f1), align(f1, f2)
        self.assertTrue(r01.ok and r12.ok)
        H02 = r01.H @ r12.H      # frame 2 -> frame 0 (documented accumulation order)
        self.assertLess(corner_error(H02, translation(90, -2)), 0.6)

    def test_deterministic(self):
        ref, mov = crop(200, 30), crop(230, 35)
        a, b = align(ref, mov), align(ref, mov)
        np.testing.assert_array_equal(a.H, b.H)
        self.assertEqual(a.metrics.n_inliers, b.metrics.n_inliers)


class TestRobustness(unittest.TestCase):
    def test_moving_object_does_not_bias_estimate(self):
        base_a, base_b = scene().copy(), scene().copy()
        draw_car(base_a, 450, 360)         # car at different scene positions
        draw_car(base_b, 530, 360)         # in the two frames (it moved 80 px)
        ref, mov = crop(200, 30, base_a), crop(230, 34, base_b)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, translation(30, 4)), 0.4)
        # some ratio-test matches on the car must have been rejected as outliers
        self.assertLess(r.metrics.n_inliers, r.metrics.n_good)

    def test_exposure_change(self):
        ref = crop(200, 30)
        mov = np.clip(crop(236, 30).astype(np.float32) * 0.65 + 12, 0, 255).astype(np.uint8)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, translation(36, 0)), 0.5)

    def test_mild_noise_and_blur(self):
        rng = np.random.default_rng(0)
        ref = crop(200, 30)
        mov = cv2.GaussianBlur(crop(225, 33), (0, 0), 1.2).astype(np.float32)
        mov = np.clip(mov + rng.normal(0, 4, mov.shape), 0, 255).astype(np.uint8)
        r = align(ref, mov)
        self.assertTrue(r.ok, r.reasons)
        self.assertLess(corner_error(r.H, translation(25, 3)), 0.8)

    def test_flann_agrees_with_bruteforce(self):
        ref, mov = crop(200, 30), crop(228, 36)
        bf = align(ref, mov, MotionConfig(matcher="bf"))
        fl = align(ref, mov, MotionConfig(matcher="flann"))
        self.assertTrue(bf.ok and fl.ok)
        self.assertLess(corner_error(bf.H, fl.H), 0.5)

    def test_ratio_test_is_monotonic(self):
        ref, mov = crop(200, 30), crop(228, 36)
        counts = [align(ref, mov, MotionConfig(ratio=r, min_matches=4,
                                               min_inliers=4)).metrics.n_good
                  for r in (0.5, 0.7, 0.9)]
        self.assertLess(counts[0], counts[1])
        self.assertLessEqual(counts[1], counts[2])


class TestRejection(unittest.TestCase):
    """Unreliable input must be rejected with a reason, never returned as ok."""

    def test_blank_frames_rejected_for_lack_of_features(self):
        blank = np.full((H, W, 3), 128, np.uint8)
        r = align(blank, blank)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, me.INSUFFICIENT_FEATURES)
        self.assertIsNone(r.H)
        self.assertTrue(r.reasons)

    def test_non_overlapping_views_rejected(self):
        r = align(crop(0, 30), crop(800, 30))       # 800 px apart => no shared content
        self.assertFalse(r.ok, "non-overlapping frames must not be accepted")
        self.assertIn(r.status, (me.INSUFFICIENT_MATCHES, me.TOO_FEW_INLIERS,
                                 me.LOW_INLIER_RATIO, me.LOW_COVERAGE,
                                 me.HIGH_REPROJ_ERROR, me.HOMOGRAPHY_FAILED))

    def test_unrelated_noise_images_rejected(self):
        a = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
        b = np.random.default_rng(2).integers(0, 256, (H, W, 3), dtype=np.uint8)
        self.assertFalse(align(a, b).ok)

    def test_different_scenes_rejected(self):
        other = make_scene(1500, 420, seed=99)
        self.assertFalse(align(crop(200, 30), crop(200, 30, other)).ok)

    def test_strict_gates_reject_a_good_pair_with_named_reason(self):
        ref, mov = crop(200, 30), crop(230, 35)
        r = align(ref, mov, MotionConfig(min_inliers=10 ** 6))
        # min_inliers is just a gate; an absurd value must reject by that gate
        self.assertFalse(r.ok)
        self.assertEqual(r.status, me.TOO_FEW_INLIERS)
        self.assertIsNotNone(r.H)       # H kept for diagnostics, but ok is False

    def test_reproj_error_gate(self):
        ref, mov = crop(200, 30), crop(230, 35)
        r = align(ref, mov, MotionConfig(max_reproj_error=1e-6))
        self.assertFalse(r.ok)
        self.assertEqual(r.status, me.HIGH_REPROJ_ERROR)

    def test_coverage_gate(self):
        ref, mov = crop(200, 30), crop(230, 35)
        r = align(ref, mov, MotionConfig(min_inlier_coverage=0.999))
        self.assertFalse(r.ok)
        self.assertEqual(r.status, me.LOW_COVERAGE)

    def test_insufficient_features_threshold(self):
        ref, mov = crop(200, 30), crop(230, 35)
        r = align(ref, mov, MotionConfig(min_features=10 ** 6))
        self.assertEqual(r.status, me.INSUFFICIENT_FEATURES)


class TestGeometryCheck(unittest.TestCase):
    SIZE = (W, H)

    def test_identity_and_translation_accepted(self):
        self.assertEqual(me.check_homography_geometry(np.eye(3), self.SIZE, 1.5), [])
        self.assertEqual(me.check_homography_geometry(translation(50, -7), self.SIZE, 1.5), [])

    def test_excessive_scale_rejected(self):
        Hs = np.diag([3.0, 3.0, 1.0])
        self.assertTrue(me.check_homography_geometry(Hs, self.SIZE, 1.5))
        Hs = np.diag([0.2, 0.2, 1.0])
        self.assertTrue(me.check_homography_geometry(Hs, self.SIZE, 1.5))

    def test_reflection_rejected(self):
        self.assertTrue(me.check_homography_geometry(np.diag([-1.0, 1.0, 1.0]), self.SIZE, 1.5))

    def test_extreme_perspective_rejected(self):
        # w = 1 + 0.01x reaches 7.4 at the right edge: that side shrinks ~7x
        Hp = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, 1.0]])
        self.assertTrue(me.check_homography_geometry(Hp, self.SIZE, 1.5))
        # w = 1 - 0.003x reaches -0.9 at the right edge: beyond the horizon
        Hh = np.array([[1, 0, 0], [0, 1, 0], [-0.003, 0, 1.0]])
        self.assertTrue(me.check_homography_geometry(Hh, self.SIZE, 1.5))
        # mild perspective typical of a real pan must still be accepted
        Hm = np.array([[1, 0, 20], [0, 1, 2], [2e-5, 1e-5, 1.0]])
        self.assertEqual(me.check_homography_geometry(Hm, self.SIZE, 1.5), [])

    def test_non_finite_and_singular_rejected(self):
        Hn = np.eye(3); Hn[0, 0] = np.nan
        self.assertTrue(me.check_homography_geometry(Hn, self.SIZE, 1.5))
        Hz = np.eye(3); Hz[2, 2] = 0
        self.assertTrue(me.check_homography_geometry(Hz, self.SIZE, 1.5))


class TestConfig(unittest.TestCase):
    def test_invalid_values(self):
        for kw in [dict(matcher="x"), dict(ratio=1.0), dict(ratio=0.0), dict(min_matches=3),
                   dict(ransac_thresh=0), dict(min_inliers=2), dict(max_scale_change=0.5),
                   dict(min_inlier_ratio=1.5), dict(max_reproj_error=0),
                   dict(ransac_confidence=1.0), dict(min_features=1)]:
            with self.subTest(kw=kw), self.assertRaises(ConfigError):
                MotionConfig(**kw).validate()

    def test_roundtrip(self):
        c = MotionConfig(ratio=0.6, matcher="flann", min_inliers=40)
        self.assertEqual(MotionConfig.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_aligner_validates_config(self):
        with self.assertRaises(ConfigError):
            me.PairwiseAligner(MotionConfig(ratio=2))


class TestWithDemoVideoGroundTruth(unittest.TestCase):
    """End-to-end: video -> reader -> aligner, compared with the truth file."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = os.path.join(cls.tmp.name, "demo")
        import sys
        argv = sys.argv
        sys.argv = ["x", "--out", cls.base, "--frames", "40", "--pan-px", "5"]
        try:
            make_demo_main()
        finally:
            sys.argv = argv
        cls.path = cls.base + ".mp4"
        with open(cls.base + "_truth.json") as f:
            cls.truth = json.load(f)["frames"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_consecutive_pairs_match_truth(self):
        with VideoReader(self.path, SamplingConfig(frame_step=2, max_dim=None)) as vr:
            results = list(me.PairwiseAligner().run(vr))
        self.assertEqual(len(results), 19)
        for r in results:
            self.assertTrue(r.ok, (r.mov_frame_index, r.reasons))
            a, b = self.truth[r.ref_frame_index], self.truth[r.mov_frame_index]
            true_dx, true_dy = b["x0"] - a["x0"], b["y0"] - a["y0"]
            self.assertAlmostEqual(r.metrics.shift_x, true_dx, delta=0.4)
            self.assertAlmostEqual(r.metrics.shift_y, true_dy, delta=0.4)
            self.assertEqual(r.ref_frame_index + 2, r.mov_frame_index)

    def test_resized_frames_scale_the_shift(self):
        with VideoReader(self.path, SamplingConfig(frame_step=4, max_dim=320)) as vr:
            results = list(me.PairwiseAligner().run(vr))
        self.assertTrue(all(r.ok for r in results))
        for r in results[:5]:
            a, b = self.truth[r.ref_frame_index], self.truth[r.mov_frame_index]
            # 320 / 640 = 0.5 scale: shifts halve
            self.assertAlmostEqual(r.metrics.shift_x, 0.5 * (b["x0"] - a["x0"]), delta=0.4)

    def test_cli_align_writes_report_and_images(self):
        out = os.path.join(self.tmp.name, "align")
        rc = cli.main(["align", self.path, "--step", "3", "--out-dir", out, "--viz-every", "4"])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "pairs.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["summary"]["ok"], rep["summary"]["pairs"])
        self.assertEqual(rep["motion_config"]["ratio"], 0.75)
        self.assertIn("sampling_config", rep)
        names = os.listdir(os.path.join(out, "matches"))
        self.assertTrue(any(n.startswith("matches_") for n in names))
        self.assertTrue(any(n.startswith("overlay_") for n in names))

    def test_cli_returns_nonzero_when_pairs_rejected(self):
        out = os.path.join(self.tmp.name, "align_strict")
        rc = cli.main(["align", self.path, "--step", "3", "--out-dir", out,
                       "--min-inliers", "100000"])
        self.assertEqual(rc, 1)
        with open(os.path.join(out, "pairs.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["summary"]["ok"], 0)
        self.assertEqual(rep["summary"]["by_status"], {"too_few_inliers": rep["summary"]["pairs"]})
        # failed pairs are visualised for debugging
        self.assertTrue(any("too_few_inliers" in n for n in os.listdir(os.path.join(out, "matches"))))


class TestCutAndSizeHandling(unittest.TestCase):
    def test_hard_cut_is_flagged_not_hidden(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            path = os.path.join(tmp.name, "cut.avi")
            other = make_scene(1500, 420, seed=5)
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 25, (W, H))
            for k in range(6):
                wr.write(crop(100 + 10 * k, 30))
            for k in range(6):                      # different scene from frame 6 on
                wr.write(crop(100 + 10 * k, 30, other))
            wr.release()
            with VideoReader(path, SamplingConfig(max_dim=None)) as vr:
                results = list(me.PairwiseAligner().run(vr))
            ok = [r.ok for r in results]
            self.assertEqual(len(ok), 11)
            self.assertFalse(ok[5], "the pair spanning the cut (5->6) must be rejected")
            self.assertTrue(all(ok[:5]) and all(ok[6:]), ok)
        finally:
            tmp.cleanup()

    def test_changing_frame_size_raises(self):
        from vidmosaic.video_reader import FrameRecord
        al = me.PairwiseAligner()
        mk = lambda i, w, h: FrameRecord(i, i, i / 25, np.zeros((h, w, 3), np.uint8), 1.0, (w, h))
        al.process(mk(0, 320, 240))
        with self.assertRaises(ValueError):
            al.process(mk(1, 160, 120))

    def test_first_frame_returns_none(self):
        from vidmosaic.video_reader import FrameRecord
        al = me.PairwiseAligner()
        rec = FrameRecord(0, 0, 0.0, crop(0, 0), 1.0, (W, H))
        self.assertIsNone(al.process(rec))


class TestVisualisation(unittest.TestCase):
    def test_draw_matches_and_overlay(self):
        ref, mov = crop(200, 30), crop(230, 35)
        r = align(ref, mov)
        img = draw_matches(ref, mov, r)
        self.assertEqual(img.shape[1], 2 * W)
        self.assertGreater(img.shape[0], H)             # banner added
        ov = alignment_overlay(ref, mov, r.H)
        self.assertEqual(ov.shape, ref.shape)
        # well-aligned static content => tiny difference in the overlapping interior
        interior = ov[40:-40, 60:W - 80].astype(float)
        self.assertLess(interior.mean(), 6.0)
        # uncovered strip (left edge, 30 px wide) is flagged, not treated as a match
        self.assertTrue(np.all(ov[100, 2] == (80, 0, 0)))

    def test_draw_matches_on_rejected_pair_without_points(self):
        blank = np.full((H, W, 3), 128, np.uint8)
        r = align(blank, blank)
        img = draw_matches(blank, blank, r)             # must not crash
        self.assertEqual(img.shape[1], 2 * W)


if __name__ == "__main__":
    unittest.main()
