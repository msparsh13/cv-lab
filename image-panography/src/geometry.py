import cv2
import numpy as np


def estimate_homography(
    kp1,
    kp2,
    matches,
    threshold=4.0
):
    if len(matches) < 8:
        return None, 0

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        threshold,
        maxIters=5000,
        confidence=0.995
    )

    if H is None or mask is None:
        return None, 0

    inliers = int(mask.sum())

    ratio = inliers / len(matches)

    if inliers < 10:
        return None, 0

    if ratio < 0.25:
        return None, 0

    if not np.all(np.isfinite(H)):
        return None, 0

    return H, inliers