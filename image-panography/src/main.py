
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from features import FeatureDetector
from matching import FeatureMatcher
from homography import estimate_homography


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("./dataset")
OUTPUT_DIR = Path("./output")

WORK_SIZE = 800
PREVIEW_SIZE = 400

MAX_FEATURES = 10000
MATCH_RATIO = 0.75

MIN_MATCHES = 150
MIN_INLIERS = 100
MIN_COVERAGE = 0.05

BEAM_WIDTH = 5
MAX_NEIGHBORS = None

MAX_CANVAS_SIZE = 5000
MAX_CANVAS_AREA = 20_000_000




# ============================================================
# Data structures
# ============================================================

@dataclass
class ImageData:
    index: int
    path: Path
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


@dataclass
class Edge:
    """
    Transform from image `src` coordinates -> image `dst` coordinates.
    """

    src: int
    dst: int
    H: np.ndarray

    matches: int
    inliers: int
    coverage: float
    reprojection_error: float
    quality: float


@dataclass
class Candidate:
    """
    A partial panorama hypothesis.

    transforms:
        image_id -> global panorama coordinates

    The first image in the candidate has identity transform.
    """

    images: tuple[int, ...]
    transforms: dict[int, np.ndarray]

    score: float = 0.0
    preview: np.ndarray | None = None


# ============================================================
# Image loading
# ============================================================

def resize_image(
    image: np.ndarray,
    max_size: int,
) -> np.ndarray:

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_size / max(h, w),
    )

    if scale == 1.0:
        return image.copy()

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def load_images() -> list[ImageData]:

    paths = sorted(
        p
        for p in DATASET_DIR.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    )

    if not paths:
        raise RuntimeError(
            f"No images found in {DATASET_DIR}"
        )

    detector = FeatureDetector(
        max_features=MAX_FEATURES,
    )

    images = []

    for index, path in enumerate(paths):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        image = resize_image(
            image,
            WORK_SIZE,
        )

        keypoints, descriptors = detector.detect(
            image,
        )

        print(
            f"[FEATURES] "
            f"{path.name}: "
            f"{len(keypoints)} keypoints"
        )

        images.append(
            ImageData(
                index=index,
                path=path,
                image=image,
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )

    return images


# ============================================================
# Geometry helpers
# ============================================================

def image_corners(
    image: np.ndarray,
) -> np.ndarray:

    h, w = image.shape[:2]

    return np.array(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def transform_corners(
    image: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    corners = image_corners(image)

    return cv2.perspectiveTransform(
        corners,
        H,
    ).reshape(-1, 2)


def polygon_area(
    points: np.ndarray,
) -> float:

    if len(points) < 3:
        return 0.0

    hull = cv2.convexHull(
        points.astype(np.float32)
    )

    return float(
        cv2.contourArea(hull)
    )


def spatial_coverage(
    image: np.ndarray,
    points: np.ndarray,
) -> float:
    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    if image_area <= 0 or len(points) < 1:
        return 0.0

    return polygon_area(points) / image_area


def grid_coverage(
    image: np.ndarray,
    points: np.ndarray,
    grid_size=3,
    min_points_per_cell=1,
):
    """
    Measure how widely the inlier points are distributed
    across the image.

    Returns:
        coverage:
            fraction of occupied grid cells.

        occupied:
            number of occupied cells.

        total:
            total number of cells.
    """

    if len(points) == 0:
        return 0.0, 0, grid_size * grid_size

    height, width = image.shape[:2]

    cell_width = width / grid_size
    cell_height = height / grid_size

    occupied = set()

    for point in points:
        x, y = point

        col = int(x / cell_width)
        row = int(y / cell_height)

        # Handle points exactly on the right/bottom edge.
        col = min(
            grid_size - 1,
            max(0, col),
        )

        row = min(
            grid_size - 1,
            max(0, row),
        )

        occupied.add(
            (row, col)
        )

    occupied_count = len(occupied)

    total_cells = (
        grid_size * grid_size
    )

    coverage = (
        occupied_count / total_cells
    )

    return (
        coverage,
        occupied_count,
        total_cells,
    )



def reprojection_error(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
) -> float:

    valid = mask.ravel().astype(bool)

    if not np.any(valid):
        return float("inf")

    projected = cv2.perspectiveTransform(
        src_points.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected[valid] - dst_points[valid],
        axis=1,
    )

    return float(
        np.median(errors)
    )


# ============================================================
# Pairwise geometry
# ============================================================

def estimate_edge(
    src: ImageData,
    dst: ImageData,
    matcher: FeatureMatcher,
) -> Edge | None:

    if (
        src.descriptors is None
        or dst.descriptors is None
    ):
        return None

    matches = matcher.match(
        src.descriptors,
        dst.descriptors,
    )

    if len(matches) < MIN_MATCHES:
        return None

    src_points = np.float32(
        [
            src.keypoints[m.queryIdx].pt
            for m in matches
        ]
    )

    dst_points = np.float32(
        [
            dst.keypoints[m.trainIdx].pt
            for m in matches
        ]
    )

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        4.0,
        maxIters=10000,
        confidence=0.999,
    )

    if H is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)

    inliers = int(
        mask.sum()
    )

    if inliers < MIN_INLIERS:
        return None

    # --------------------------------------------------
    # Keep only RANSAC inliers.
    # --------------------------------------------------

    inlier_points1 = src_points[mask]
    inlier_points2 = dst_points[mask]

    # --------------------------------------------------
    # Convex-hull spatial coverage.
    # --------------------------------------------------

    coverage = spatial_coverage(
        src.image,
        inlier_points1,
    )

    if coverage < MIN_COVERAGE:
        return None

    # --------------------------------------------------
    # Grid spatial coverage.
    #
    # This prevents a large convex hull from hiding
    # concentrated/repeated texture matches.
    # --------------------------------------------------

    grid_coverage_value, occupied_cells, total_cells = (
        grid_coverage(
            src.image,
            inlier_points1,
            grid_size=3,
        )
    )

    print(
        f"[GRID] "
        f"{src.path.stem} -> "
        f"{dst.path.stem} | "
        f"cells={occupied_cells}/{total_cells} "
        f"({grid_coverage_value:.1%})"
    )

    # --------------------------------------------------
    # Require inliers to occupy at least 4/9 cells.
    # --------------------------------------------------

    MIN_GRID_CELLS = 4
    

    if occupied_cells < MIN_GRID_CELLS:
        print(
            f"[REJECT] "
            f"{src.path.stem} -> "
            f"{dst.path.stem}: "
            f"poor spatial distribution "
            f"({occupied_cells}/{total_cells} cells)"
        )

        return None

    # --------------------------------------------------
    # Reprojection error.
    # --------------------------------------------------

    error = reprojection_error(
        src_points,
        dst_points,
        H,
        mask,
    )

    # --------------------------------------------------
    # Score.
    # --------------------------------------------------

    inlier_score = min(
        1.0,
        inliers / 100.0,
    )

    coverage_score = min(
        1.0,
        coverage / 0.5,
    )

    grid_score = (
        grid_coverage_value
    )

    error_score = np.exp(
        -error / 3.0
    )

    quality = (
        0.40 * inlier_score
        + 0.20 * coverage_score
        + 0.20 * grid_score
        + 0.20 * error_score
    )

    return Edge(
        src=src.index,
        dst=dst.index,
        H=H,
        matches=len(matches),
        inliers=inliers,
        coverage=coverage,
        reprojection_error=error,
        quality=float(quality),
    )


def build_graph(
    images: list[ImageData],
) -> dict[tuple[int, int], Edge]:

    matcher = FeatureMatcher(
        ratio=MATCH_RATIO,
    )

    graph = {}

    total = len(images)

    print()
    print(
        f"[GRAPH] Building pairwise graph "
        f"for {total} images..."
    )

    for i in range(total):

        for j in range(i + 1, total):

            edge_ij = estimate_edge(
                images[i],
                images[j],
                matcher,
            )

            if edge_ij is None:
                continue

            graph[(i, j)] = edge_ij

            try:
                H_ji = np.linalg.inv(
                    edge_ij.H
                )
            except np.linalg.LinAlgError:
                continue

            graph[(j, i)] = Edge(
                src=j,
                dst=i,
                H=H_ji,
                matches=edge_ij.matches,
                inliers=edge_ij.inliers,
                coverage=edge_ij.coverage,
                reprojection_error=edge_ij.reprojection_error,
                quality=edge_ij.quality,
            )

            print(
                f"[EDGE] "
                f"{images[i].path.name} <-> "
                f"{images[j].path.name} | "
                f"matches={edge_ij.matches} "
                f"inliers={edge_ij.inliers} "
                f"coverage={edge_ij.coverage:.3f} "
                f"error={edge_ij.reprojection_error:.2f}"
            )

    print()
    print(
        f"[GRAPH] {len(graph) // 2} usable pairs"
    )

    return graph


# ============================================================
# Candidate geometry
# ============================================================

def get_edge(
    graph: dict[tuple[int, int], Edge],
    src: int,
    dst: int,
) -> Edge | None:

    return graph.get(
        (src, dst)
    )


def candidate_bounds(
    candidate: Candidate,
    images: list[ImageData],
):

    all_points = []

    for image_id in candidate.images:

        image = images[image_id]

        H = candidate.transforms[
            image_id
        ]

        points = transform_corners(
            image.image,
            H,
        )

        if not np.all(
            np.isfinite(points)
        ):
            return (
                np.array(
                    [0, 0],
                    dtype=int,
                ),
                np.array(
                    [0, 0],
                    dtype=int,
                ),
                0,
                0,
            )

        all_points.append(
            points
        )

    all_points = np.vstack(
        all_points
    )

    min_xy = np.floor(
        all_points.min(axis=0)
    ).astype(int)

    max_xy = np.ceil(
        all_points.max(axis=0)
    ).astype(int)

    width = int(
        max_xy[0] - min_xy[0]
    )

    height = int(
        max_xy[1] - min_xy[1]
    )

    return (
        min_xy,
        max_xy,
        width,
        height,
    )


def valid_candidate(
    candidate: Candidate,
    images: list[ImageData],
) -> bool:

    _, _, width, height = candidate_bounds(
        candidate,
        images,
    )

    if width <= 0 or height <= 0:
        return False

    if width > MAX_CANVAS_SIZE:
        return False

    if height > MAX_CANVAS_SIZE:
        return False

    if width * height > MAX_CANVAS_AREA:
        return False

    return True


# ============================================================
# Rendering
# ============================================================

def render_candidate(
    candidate: Candidate,
    images: list[ImageData],
    max_size: int | None = None,
) -> np.ndarray:

    min_xy, _, width, height = candidate_bounds(
        candidate,
        images,
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Candidate has invalid canvas dimensions."
        )

    translation = np.array(
        [
            [1.0, 0.0, -min_xy[0]],
            [0.0, 1.0, -min_xy[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.float32,
    )

    weights = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for image_id in candidate.images:

        image = images[image_id].image

        H = (
            translation
            @ candidate.transforms[image_id]
        )

        warped = cv2.warpPerspective(
            image,
            H,
            (width, height),
        )

        mask = np.ones(
            image.shape[:2],
            dtype=np.uint8,
        )

        warped_mask = cv2.warpPerspective(
            mask,
            H,
            (width, height),
        )

        valid = warped_mask > 0

        canvas[valid] += (
            warped[valid].astype(
                np.float32
            )
        )

        weights[valid] += 1.0

    valid = weights > 0

    result = np.zeros_like(
        canvas,
        dtype=np.uint8,
    )

    result[valid] = (
        canvas[valid]
        / weights[valid, None]
    ).astype(np.uint8)

    if max_size is not None:
        result = resize_image(
            result,
            max_size,
        )

    return result


# ============================================================
# Candidate scoring
# ============================================================

def candidate_score(
    candidate: Candidate,
    edge: Edge,
    previous_width: int,
    previous_height: int,
    images: list[ImageData],
) -> float:

    _, _, width, height = candidate_bounds(
        candidate,
        images,
    )

    old_area = max(
        1,
        previous_width * previous_height,
    )

    new_area = width * height

    growth = new_area / old_area

    growth_penalty = min(
        1.0,
        max(
            0.0,
            growth - 1.0,
        ) / 3.0,
    )

    size_score = (
        1.0 - growth_penalty
    )

    score = (
        0.55 * edge.quality
        + 0.25 * size_score
        + 0.20 * min(
            1.0,
            len(candidate.images) / 10.0,
        )
    )

    return (
        candidate.score
        + score
    )


# ============================================================
# Beam search
# ============================================================

def expand_candidate(
    candidate: Candidate,
    new_image_id: int,
    edge: Edge,
    images: list[ImageData],
) -> Candidate | None:

    existing_id = edge.src
    H_existing_to_new = edge.H

    try:
        H_new_to_existing = np.linalg.inv(
            H_existing_to_new
        )
    except np.linalg.LinAlgError:
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"singular homography"
        )
        return None

    T_existing = candidate.transforms[
        existing_id
    ]

    T_new = (
        T_existing
        @ H_new_to_existing
    )

    if not np.all(
        np.isfinite(T_new)
    ):
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"non-finite transform"
        )
        return None

    transforms = dict(
        candidate.transforms
    )

    transforms[new_image_id] = T_new

    new_candidate = Candidate(
        images=(
            *candidate.images,
            new_image_id,
        ),
        transforms=transforms,
    )

    min_xy, max_xy, width, height = candidate_bounds(
        new_candidate,
        images,
    )

    if width <= 0 or height <= 0:
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"invalid canvas "
            f"{width}x{height}"
        )
        return None

    if width > MAX_CANVAS_SIZE:
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"canvas too wide "
            f"{width}px"
        )
        return None

    if height > MAX_CANVAS_SIZE:
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"canvas too tall "
            f"{height}px"
        )
        return None

    area = width * height

    if area > MAX_CANVAS_AREA:
        print(
            f"[REJECT] "
            f"{images[new_image_id].path.stem}: "
            f"canvas too large "
            f"{width}x{height} "
            f"({area:,} pixels)"
        )
        return None

    _, _, old_width, old_height = candidate_bounds(
        candidate,
        images,
    )

    new_candidate.score = candidate_score(
        new_candidate,
        edge,
        old_width,
        old_height,
        images,
    )

    return new_candidate


def get_neighbors(
    image_id: int,
    graph: dict[tuple[int, int], Edge],
) -> list[Edge]:

    edges = [
        edge
        for (src, _), edge in graph.items()
        if src == image_id
    ]

    edges.sort(
        key=lambda e: e.quality,
        reverse=True,
    )

    if MAX_NEIGHBORS is not None:
        edges = edges[:MAX_NEIGHBORS]

    return edges


def beam_search(
    images: list[ImageData],
    graph: dict[tuple[int, int], Edge],
) -> Candidate:

    root = Candidate(
        images=(4,),
        transforms={
            4: np.eye(
                3,
                dtype=np.float64,
            )
        },
        score=0.0,
    )

    beam = [root]
    best_partial = root

    target_count = len(images)

    print()
    print(
        "[SEARCH] Starting beam search..."
    )

    while True:

        # ----------------------------------------------------
        # Complete candidate
        # ----------------------------------------------------

        complete = [
            candidate
            for candidate in beam
            if len(candidate.images)
            == target_count
        ]

        if complete:

            complete.sort(
                key=lambda c: c.score,
                reverse=True,
            )

            print(
                f"[SEARCH] Complete candidate found "
                f"score={complete[0].score:.3f}"
            )

            return complete[0]

        # ----------------------------------------------------
        # Expand every candidate.
        #
        # IMPORTANT:
        # Keep ALL possible edges to a new image.
        # The old version retained only the strongest edge,
        # which could discard a geometrically better route.
        # ----------------------------------------------------

        expanded = []

        for candidate in beam:

            frontier_edges = []

            for existing_id in candidate.images:

                for edge in get_neighbors(
                    existing_id,
                    graph,
                ):

                    new_id = edge.dst

                    if new_id in candidate.images:
                        continue

                    frontier_edges.append(
                        edge
                    )

            for edge in frontier_edges:

                child = expand_candidate(
                    candidate,
                    edge.dst,
                    edge,
                    images,
                )

                if child is not None:
                    expanded.append(
                        child
                    )

        # ----------------------------------------------------
        # No more valid expansions.
        #
        # Return the best partial panorama instead of
        # falling through and returning None.
        # ----------------------------------------------------

        if not expanded:

            print(
                "[SEARCH] No valid expansions remaining."
            )

            print(
                f"[SEARCH] Using best panorama with "
                f"{len(best_partial.images)} images."
            )

            return best_partial

        # ----------------------------------------------------
        # Deduplicate identical image sets.
        # ----------------------------------------------------

        unique = {}

        for candidate in expanded:

            key = tuple(
                sorted(candidate.images)
            )

            previous = unique.get(
                key
            )

            if (
                previous is None
                or candidate.score
                > previous.score
            ):
                unique[key] = candidate

        expanded = list(
            unique.values()
        )

        expanded.sort(
            key=lambda c: c.score,
            reverse=True,
        )

        beam = expanded[
            :BEAM_WIDTH
        ]

        # ----------------------------------------------------
        # Track best partial candidate.
        # ----------------------------------------------------

        for candidate in beam:

            if (
                len(candidate.images)
                > len(best_partial.images)
                or (
                    len(candidate.images)
                    == len(best_partial.images)
                    and candidate.score
                    > best_partial.score
                )
            ):
                best_partial = candidate

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"[SEARCH] "
            f"depth={len(beam[0].images)} "
            f"expanded={len(expanded)} "
            f"beam={len(beam)}"
        )

        for rank, candidate in enumerate(
            beam,
            start=1,
        ):

            names = [
                images[i].path.stem
                for i in candidate.images
            ]

            print(
                f"    #{rank} "
                f"score={candidate.score:.3f} "
                f"{names}"
            )


# ============================================================
# Final rendering
# ============================================================

def render_final(
    candidate: Candidate,
    images: list[ImageData],
) -> np.ndarray:

    print()
    print(
        "[FINAL] Rendering selected panorama..."
    )

    print(
        "[FINAL] Images:"
    )

    for image_id in candidate.images:

        print(
            f"    {images[image_id].path.name}"
        )

    result = render_candidate(
        candidate,
        images,
        max_size=None,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load + resize + detect features
    # --------------------------------------------------------

    images = load_images()

    print()
    print(
        f"[MAIN] Loaded {len(images)} images"
    )

    # --------------------------------------------------------
    # 2. Build image compatibility graph
    # --------------------------------------------------------

    graph = build_graph(
        images,
    )

    if not graph:
        raise RuntimeError(
            "No usable image pairs were found."
        )

    # --------------------------------------------------------
    # 3. Search possible panorama constructions
    # --------------------------------------------------------

    best = beam_search(
        images,
        graph,
    )

    if best is None:
        raise RuntimeError(
            "Beam search did not produce a valid panorama."
        )

    print()
    print(
        f"[MAIN] Beam search selected "
        f"{len(best.images)}/{len(images)} images."
    )

    if len(best.images) < len(images):

        skipped = [
            images[i].path.name
            for i in range(len(images))
            if i not in best.images
        ]

        print(
            "[MAIN] Skipped images:"
        )

        for name in skipped:
            print(
                f"    {name}"
            )

    # --------------------------------------------------------
    # 4. Render ONLY the final result
    # --------------------------------------------------------

    panorama = render_final(
        best,
        images,
    )

    if panorama is None or panorama.size == 0:
        raise RuntimeError(
            "Final panorama rendering failed."
        )

    # --------------------------------------------------------
    # 5. Save result
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        "panorama_final.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        panorama,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Could not save panorama to "
            f"{output_path}"
        )

    print()
    print(
        f"[DONE] Saved {output_path}"
    )

    print(
        f"[DONE] Final size: "
        f"{panorama.shape[1]}x"
        f"{panorama.shape[0]}"
    )


if __name__ == "__main__":
    main()
