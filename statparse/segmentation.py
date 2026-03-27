"""Step 3a: Geometric segmentation — connected components + hierarchical grouping."""
import numpy as np
import cv2
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from sklearn.neighbors import NearestNeighbors




def segment(binary_image: np.ndarray) -> list[dict]:
    """Segment a binary image into blocks using statistical spatial clustering.

    Pipeline:
        1. Extract connected components
        2. Compute k-NN distances and angles between components
        3. Model distance distributions (KDE)
        4. Find valley thresholds to separate intra-group vs inter-group gaps
        5. Group hierarchically: characters → words → lines → blocks → columns

    Returns:
        List of block dicts, each with keys:
            - 'bbox': (x, y, w, h)
            - 'components': list of connected component bboxes inside this block
    """
    components = _extract_components(binary_image)
    if len(components) == 0:
        return []

    words = _group_level(components, angle_range=(-15, 15), level="word")
    lines = _group_level(words, angle_range=(-15, 15), level="line")
    blocks = _group_level(lines, angle_range=None, level="block")

    return [
        {
            "bbox": b["bbox"],
            "components": b["children"],
        }
        for b in blocks
    ]


# ─────────────────────────── helpers ────────────────────────────


def _extract_components(binary_image: np.ndarray) -> list[dict]:
    """Return connected components as list of {'bbox': (x,y,w,h), 'center': (cx,cy)}."""
    inverted = cv2.bitwise_not(binary_image)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    components = []
    for i in range(1, n):  # skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < 4:  # drop noise pixels
            continue
        cx, cy = centroids[i]
        components.append({"bbox": (int(x), int(y), int(w), int(h)), "center": (cx, cy)})

    return components


def _centers(items: list[dict]) -> np.ndarray:
    return np.array([it["center"] for it in items])


def _union_bbox(bboxes: list[tuple]) -> tuple:
    xs = [b[0] for b in bboxes]
    ys = [b[1] for b in bboxes]
    x2s = [b[0] + b[2] for b in bboxes]
    y2s = [b[1] + b[3] for b in bboxes]
    x, y = min(xs), min(ys)
    return (x, y, max(x2s) - x, max(y2s) - y)


def _kde_valley_threshold(distances: np.ndarray, bandwidth: float = 0.15) -> float:
    """Use KDE to find the first valley in the distance distribution.
    Falls back to median if no valley found.
    """
    if len(distances) < 5:
        return float(np.median(distances))

    log_d = np.log1p(distances)
    kde = gaussian_kde(log_d, bw_method=bandwidth)
    xs = np.linspace(log_d.min(), log_d.max(), 300)
    density = kde(xs)

    valleys, = argrelmin(density, order=5)
    if len(valleys) == 0:
        return float(np.expm1(np.median(log_d)))

    # pick the first (smallest-distance) valley
    threshold_log = xs[valleys[0]]
    return float(np.expm1(threshold_log))


def _group_level(
    items: list[dict],
    angle_range: tuple | None,
    level: str,
) -> list[dict]:
    """Group items into clusters at one hierarchical level.

    Args:
        items:       list of dicts with 'bbox' and 'center'
        angle_range: if set, only consider neighbours within this angular range (degrees)
                     — used to restrict word/line grouping to horizontal neighbours
        level:       label for debugging only
    """
    if len(items) <= 1:
        return [_make_group(items, items)] if items else []

    centers = _centers(items)
    k = min(5, len(items) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(centers)
    distances, indices = nbrs.kneighbors(centers)

    # collect relevant pairwise distances (filtered by angle if needed)
    relevant_distances = []
    edges = []  # (i, j, distance)

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for d, j in zip(dists, idxs):
            if angle_range is not None:
                dy = centers[j][1] - centers[i][1]
                dx = centers[j][0] - centers[i][0]
                angle = np.degrees(np.arctan2(dy, dx))
                if not (angle_range[0] <= angle <= angle_range[1]):
                    continue
            relevant_distances.append(d)
            edges.append((i, j, d))

    if not relevant_distances:
        # no valid edges at this level: each item is its own group
        return [_make_group([it], [it]) for it in items]

    threshold = _kde_valley_threshold(np.array(relevant_distances))

    # union-find grouping
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, j, d in edges:
        if d <= threshold:
            union(i, j)

    # collect groups
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(items)):
        groups[find(idx)].append(idx)

    result = []
    for member_indices in groups.values():
        members = [items[m] for m in member_indices]
        result.append(_make_group(members, members))

    return result


def _make_group(members: list[dict], children: list[dict]) -> dict:
    bbox = _union_bbox([m["bbox"] for m in members])
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    return {"bbox": bbox, "center": (cx, cy), "children": children}