
import cv2
import numpy as np


def estimate_homography(
    image1,
    keypoints1,
    keypoints2,
    matches,
    reprojection_threshold=4.0,
):
    # --------------------------------------------------
    # Keep only actual DMatch objects.
    # --------------------------------------------------

    valid_matches = [
        match
        for match in matches
        if isinstance(match, cv2.DMatch)
    ]

    if len(valid_matches) != len(matches):
        print(
            f"Removed "
            f"{len(matches) - len(valid_matches)} "
            f"invalid match objects"
        )

    matches = valid_matches

    if len(matches) < 4:
        print(
            f"Not enough valid matches: "
            f"{len(matches)}"
        )

        return None, None

    # --------------------------------------------------
    # Convert matches -> point correspondences.
    # --------------------------------------------------

    points1 = np.float32([
        keypoints1[match.queryIdx].pt
        for match in matches
    ])

    points2 = np.float32([
        keypoints2[match.trainIdx].pt
        for match in matches
    ])

    # --------------------------------------------------
    # Estimate projective homography.
    # --------------------------------------------------

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        reprojection_threshold,
        maxIters=10000,
        confidence=0.999,
    )

    if H is None or mask is None:
        print(
            "Homography estimation failed"
        )

        return None, None

    mask = mask.ravel().astype(bool)

    inlier_points1 = points1[mask]
    inlier_points2 = points2[mask]

    inlier_count = int(mask.sum())

    print(
        f"RANSAC inliers: "
        f"{inlier_count}/{len(matches)} "
        f"({inlier_count / len(matches):.1%})"
    )

    # --------------------------------------------------
    # Spatial coverage.
    # --------------------------------------------------

    height, width = image1.shape[:2]

    x_range = np.ptp(
        inlier_points1[:, 0]
    )

    y_range = np.ptp(
        inlier_points1[:, 1]
    )

    print()
    print("Inlier spatial coverage:")
    print(
        f"X range: {x_range:.2f}"
    )
    print(
        f"Y range: {y_range:.2f}"
    )

    print(
        f"Coverage X: "
        f"{x_range / width:.1%}"
    )

    print(
        f"Coverage Y: "
        f"{y_range / height:.1%}"
    )

    # --------------------------------------------------
    # Degenerate homography detection.
    #
    # If all points occupy a thin strip, a full
    # projective transform can become unstable.
    # --------------------------------------------------

    strip_like = (
        y_range < 0.30 * height
        or x_range < 0.30 * width
    )

    if strip_like:
        print()
        print(
            "Inliers are spatially degenerate."
        )

        print(
            "Using affine transform."
        )

        affine, affine_mask = (
            cv2.estimateAffine2D(
                inlier_points1,
                inlier_points2,
                method=cv2.RANSAC,
                ransacReprojThreshold=(
                    reprojection_threshold
                ),
                maxIters=10000,
                confidence=0.999,
                refineIters=10,
            )
        )

        if affine is None:
            print(
                "Affine estimation failed"
            )

            return None, None

        H = np.vstack([
            affine,
            [0.0, 0.0, 1.0],
        ])

        # The original mask corresponds to the original
        # match list and remains useful to the caller.
        print()
        print("Affine transform:")
        print(H)

        return H, mask

    # --------------------------------------------------
    # Normal homography.
    # --------------------------------------------------

    print()
    print("Homography:")
    print(H)

    return H, mask
