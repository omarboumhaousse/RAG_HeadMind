"""
Step 3a — Geometric segmentation of a binarized document page.

Produces a hierarchical decomposition:  blocks → lines → words → components.

Method:  2D Variational Bayesian Gaussian Mixture Model (VBGMM) on
         inter-component displacement vectors, with XY-cut pre-segmentation
         and projection-profile word detection.

Input:   binary image (H × W, uint8, ink = 0, background = 255)
Output:  list of block dicts:
             {
                 "bbox": (x, y, w, h),
                 "lines": [
                     {
                         "bbox": (x, y, w, h),
                         "words": [
                             {
                                 "bbox": (x, y, w, h),
                                 "components": [{"bbox": (x, y, w, h)}, ...]
                             }, ...
                         ]
                     }, ...
                 ],
                 "components": [{"bbox": (x, y, w, h)}, ...]   # flat list
             }

Pipeline overview:
    0.  Extract connected components (ink blobs) from the page.
    1.  Merge sub-character fragments (split radicals, diacritics, CJK strokes).
    2.  XY-cut pre-segmentation: split the page at large whitespace gaps so
        that each region has a homogeneous gap distribution.
    3.  For each region independently:
        a.  Build a neighbourhood graph (Delaunay triangulation).
        b.  Compute 2D displacement vectors between neighbour centroids.
        c.  Fit a VBGMM (auto-selects the number of gap types).
        d.  Classify clusters into word / line / block via polar decomposition.
        e.  Label every edge with Bayes' MAP decision rule.
        f.  Hierarchically merge components (3-pass union-find).
    4.  Post-process: projection-profile validation + degenerate block removal.
"""

import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay, QhullError
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════
# STEP 0 — Connected-component extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_connected_components(binary_image: np.ndarray):
    """Extract ink blobs from a binary page image.

    Uses OpenCV's optimised two-pass algorithm (He et al., 2017).
    Row 0 of the stats/centroids arrays is the background, we skip it.

    Returns
    -------
    stats : ndarray, shape (N, 5)
        [x, y, w, h, area] for each component.
    centroids : ndarray, shape (N, 2)
        [cx, cy] for each component.
    """
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    # Index 0 = white background of the page — drop it.
    return stats[1:], centroids[1:]


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Sub-character merging
# ═══════════════════════════════════════════════════════════════════

def _merge_subcharacter_components(stats, centroids):
    """Merge CCs that are fragments of the same character.

    Many CJK characters (and Latin diacritics like ï, é) are split into
    separate connected components by binarisation. This function
    re-assembles them using spatial proximity (cKDTree), aspect-ratio
    constraints, and area limits to avoid merging unrelated blobs.

    Merge criteria (all must hold):
        - edge-to-edge gap < 0.35 * median_h  (close enough)
        - merged aspect ratio ∈ [0.25, 4.0]   (plausible glyph shape)
        - merged area < 6 * median_area       (not merging across words)
    """
    n = len(stats)
    if n <= 1:
        return stats, centroids

    median_h = max(float(np.median(stats[:, 3])), 1.0)
    median_w = max(float(np.median(stats[:, 2])), 1.0)
    median_area = median_h * median_w
    max_gap = 0.35 * median_h

    # Union-find for merging
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Pre-compute bounding boxes as (x1, y1, x2, y2)
    bboxes = []
    for i in range(n):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        bboxes.append((x, y, x + w, y + h))

    # Spatial index: find candidate pairs within merging distance
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=max_gap + max(median_h, median_w))

    for (i, j) in pairs:
        if find(i) == find(j):
            continue
        x1_i, y1_i, x2_i, y2_i = bboxes[i]
        x1_j, y1_j, x2_j, y2_j = bboxes[j]

        # Edge-to-edge gap (0 if overlapping)
        gap_x = max(0, max(x1_i, x1_j) - min(x2_i, x2_j))
        gap_y = max(0, max(y1_i, y1_j) - min(y2_i, y2_j))
        if gap_x > max_gap or gap_y > max_gap:
            continue

        # Merged bounding box sanity checks
        merged_w = max(x2_i, x2_j) - min(x1_i, x1_j)
        merged_h = max(y2_i, y2_j) - min(y1_i, y1_j)
        if merged_w == 0 or merged_h == 0:
            continue
        aspect = merged_w / merged_h
        if aspect < 0.25 or aspect > 4.0:
            continue
        if merged_w * merged_h > 6.0 * median_area:
            continue

        union(i, j)

    # Rebuild stats and centroids from merged groups
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    new_stats, new_centroids = [], []
    for members in groups.values():
        xs = [int(stats[m, 0]) for m in members]
        ys = [int(stats[m, 1]) for m in members]
        x2s = [int(stats[m, 0]) + int(stats[m, 2]) for m in members]
        y2s = [int(stats[m, 1]) + int(stats[m, 3]) for m in members]
        total_area = sum(int(stats[m, 4]) for m in members)
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(x2s), max(y2s)
        w, h = x_max - x_min, y_max - y_min
        new_stats.append([x_min, y_min, w, h, total_area])
        new_centroids.append([x_min + w / 2.0, y_min + h / 2.0])

    return (np.array(new_stats, dtype=np.int32),
            np.array(new_centroids, dtype=np.float64))


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — XY-cut pre-segmentation
# ═══════════════════════════════════════════════════════════════════

def _xy_cut_presegment(binary_image, min_width=50, min_height=None,
                       ink_threshold_ratio=0.005):
    """Recursively split the page into regions at large whitespace gaps.

    Different page regions (code blocks, prose, headers, figures) have
    fundamentally different gap distributions.  A single global VBGMM
    averages these distributions and loses discriminative power.  By
    splitting at large structural gaps first, each region gets its own
    VBGMM that models its local gap structure accurately.

    The cut only fires at large gaps (column separators, section breaks),
    NOT between normal text lines.

    Returns a list of (x, y, w, h) region rectangles.
    """
    img_h, img_w = binary_image.shape[:2]

    # Estimate median text height to set adaptive thresholds
    inverted = cv2.bitwise_not(binary_image)
    n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if n_labels > 1:
        heights = cc_stats[1:, 3]
        areas = cc_stats[1:, 4]
        valid = areas >= 4
        median_h = float(np.median(heights[valid])) if np.any(valid) else 20.0
    else:
        median_h = 20.0

    # A region must contain ≥4 text lines to justify its own VBGMM
    if min_height is None:
        min_height = max(50, int(4 * median_h * 1.5))

    # Only split at gaps ≥3× median character height (section-level gaps)
    min_h_gap = max(10, int(3.0 * median_h))

    def _find_best_gap(gap_mask, min_gap=10):
        """Find the widest gap near the center of the profile."""
        gaps = []
        in_gap = False
        gap_start = 0
        for i in range(len(gap_mask)):
            if gap_mask[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not gap_mask[i] and in_gap:
                if i - gap_start >= min_gap:
                    gaps.append((gap_start, i, i - gap_start))
                in_gap = False
        if in_gap and len(gap_mask) - gap_start >= min_gap:
            gaps.append((gap_start, len(gap_mask), len(gap_mask) - gap_start))
        if not gaps:
            return None
        # Score gaps: prefer wide gaps near the center, ignore edge margins
        center = len(gap_mask) / 2.0
        edge_margin = len(gap_mask) * 0.05
        best, best_score = None, -1
        for gs, ge, gw in gaps:
            if gs < edge_margin or ge > len(gap_mask) - edge_margin:
                continue
            gap_center = (gs + ge) / 2.0
            score = gw - 0.5 * abs(gap_center - center) / len(gap_mask) * gw
            if score > best_score:
                best_score = score
                best = (gs, ge)
        return best

    def _recursive_cut(rx, ry, rw, rh, depth=0):
        if rw < min_width or rh < min_height or depth > 20:
            return [(rx, ry, rw, rh)]

        y1, y2 = max(0, ry), min(img_h, ry + rh)
        x1, x2 = max(0, rx), min(img_w, rx + rw)
        if y2 <= y1 or x2 <= x1:
            return [(rx, ry, rw, rh)]
        region = binary_image[y1:y2, x1:x2]

        # Try vertical cut first (splits columns)
        v_profile = np.sum(region == 0, axis=0)
        v_threshold = max(1, int(ink_threshold_ratio * rh))
        v_gap = v_profile <= v_threshold
        best_v = _find_best_gap(v_gap, min_gap=max(10, rw // 20))
        if best_v is not None:
            gs, ge = best_v
            return (_recursive_cut(rx, ry, gs, rh, depth + 1) +
                    _recursive_cut(rx + ge, ry, rw - ge, rh, depth + 1))

        # Try horizontal cut (splits rows/sections)
        h_profile = np.sum(region == 0, axis=1)
        h_threshold = max(1, int(ink_threshold_ratio * rw))
        h_gap = h_profile <= h_threshold
        best_h = _find_best_gap(h_gap, min_gap=min_h_gap)
        if best_h is not None:
            gs, ge = best_h
            return (_recursive_cut(rx, ry, rw, gs, depth + 1) +
                    _recursive_cut(rx, ry + ge, rw, rh - ge, depth + 1))

        return [(rx, ry, rw, rh)]

    return _recursive_cut(0, 0, img_w, img_h)


# ═══════════════════════════════════════════════════════════════════
# STEP 3a — Neighbourhood graph (Delaunay triangulation)
# ═══════════════════════════════════════════════════════════════════

def _delaunay_edges(centroids: np.ndarray) -> list[tuple]:
    """Build a neighbourhood graph via Delaunay triangulation.

    Delaunay edges are a superset of the nearest-neighbour graph and the
    minimum spanning tree.  They adapt to local density without a fixed k,
    and computation is O(N log N).
    """
    tri = Delaunay(centroids)
    unique_edges = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            unique_edges.add(tuple(sorted((simplex[a], simplex[b]))))
    return list(unique_edges)


def _fallback_knn_edges(centroids: np.ndarray) -> list[tuple]:
    """Fallback when Delaunay fails (collinear / <3 points).  Uses 5-NN."""
    n = len(centroids)
    k = min(5, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroids)
    _, indices = nbrs.kneighbors(centroids)
    unique_edges = set()
    for i in range(n):
        for j in indices[i]:
            if i != j:
                unique_edges.add(tuple(sorted((i, j))))
    return list(unique_edges)


# ═══════════════════════════════════════════════════════════════════
# STEP 3b — 2D displacement vectors
# ═══════════════════════════════════════════════════════════════════

def _compute_displacement_vectors(edges, centroids, stats,
                                  max_distance_ratio=6.0):
    """Compute normalised, symmetrised 2D displacement vectors for edges.

    Why 2D (not scalar distance)?
        A scalar distance loses direction.  Two 50px gaps are identical
        whether horizontal (word spacing) or vertical (line spacing).
        Keeping (dx, dy) lets the VBGMM discover clusters that encode
        BOTH magnitude and direction.

    Why normalise by median height?
        Makes vectors scale-invariant across font sizes and DPI values.
        A word gap is always ~1–2 median-heights wide.

    Why symmetrise?
        Each undirected edge has arbitrary orientation.  Without both
        (dx, dy) and (-dx, -dy), the VBGMM would see an asymmetric
        cloud and fit two Gaussians per gap type.  Symmetrisation
        produces a single pair of mirror Gaussians, easy to deduplicate.

    Why prune long edges?
        Delaunay connects convex-hull points with page-spanning edges.
        These hull artefacts would create spurious clusters.

    Returns
    -------
    vectors_2d   : ndarray (M, 2) — normalised, symmetrised vectors
    pruned_edges : list of (i, j) — edges that survived pruning
    edge_mapping : list of int — for each row in vectors_2d, the edge index
    """
    median_h = max(float(np.median(stats[:, 3])), 1.0)
    pruned_edges, vectors, edge_mapping = [], [], []

    for i_cc, j_cc in edges:
        dx = centroids[j_cc][0] - centroids[i_cc][0]
        dy = centroids[j_cc][1] - centroids[i_cc][1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist / median_h > max_distance_ratio:
            continue  # hull artefact

        dx_n, dy_n = dx / median_h, dy / median_h
        edge_idx = len(pruned_edges)
        pruned_edges.append((i_cc, j_cc))

        vectors.append([dx_n, dy_n])        # forward
        edge_mapping.append(edge_idx)
        vectors.append([-dx_n, -dy_n])      # mirror
        edge_mapping.append(edge_idx)

    vectors_2d = np.array(vectors) if vectors else np.empty((0, 2))
    return vectors_2d, pruned_edges, edge_mapping


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — VBGMM fitting
# ═══════════════════════════════════════════════════════════════════

def _fit_vbgmm(vectors_2d: np.ndarray) -> BayesianGaussianMixture:
    """Fit a Variational Bayesian GMM on the 2D displacement vectors.

    Why Variational Bayesian (not standard EM-GMM)?
        A standard GMM requires the user to fix K.  Some pages have 2 gap
        types, others 5+.  The Dirichlet Process prior drives unnecessary
        component weights toward zero, auto-selecting K* from the data.

    Hyperparameters
    ---------------
    n_components : min(15, n_samples // 2)
        Generous upper bound, capped to ensure enough data per cluster.
    weight_concentration_prior : 0.1
        Small α₀ favours sparse solutions (fewer active components).
    covariance_type : 'full'
        Full 2×2 covariance captures elliptical clusters (word gaps are
        horizontally elongated, line gaps are vertically elongated).
    """
    n_samples = vectors_2d.shape[0]
    k_init = min(15, max(2, n_samples // 2))

    model = BayesianGaussianMixture(
        n_components=k_init,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.1,
        max_iter=300,
        random_state=42,
    )
    model.fit(vectors_2d)
    return model


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Cluster classification (word / line / block)
# ═══════════════════════════════════════════════════════════════════

def _classify_clusters(model: BayesianGaussianMixture,
                       weight_threshold: float = 0.01) -> dict[int, str]:
    """Assign a semantic label to each surviving VBGMM component.

    Each component's mean (μx, μy) is converted to polar coordinates:
        radius = ‖μ‖       → gap SIZE
        angle  = atan2(|μy|, |μx|) → gap DIRECTION

    Classification rules:
        angle < 30°  → horizontal → "word"   (character / word spacing)
        angle ≥ 30°  → vertical   → "line" or "block"

    Among vertical clusters (sorted by radius), the largest radius ratio
    separates "line" (below the jump) from "block" (above).

    Mirror Gaussians (produced by symmetrisation) are deduplicated.
    """
    weights = model.weights_
    means = model.means_

    # ── Filter active components ──────────────────────────────────
    active = np.where(weights > weight_threshold)[0]
    if len(active) == 0:
        return {}

    # ── Deduplicate mirror pairs ──────────────────────────────────
    # Two components are mirrors if μ_a ≈ -μ_b (sum near zero).
    used = set()
    unique_reps = []
    for idx in active:
        if idx in used:
            continue
        mu = means[idx]
        for jdx in active:
            if jdx <= idx or jdx in used:
                continue
            sum_norm = np.linalg.norm(mu + means[jdx])
            avg_r = (np.linalg.norm(mu) + np.linalg.norm(means[jdx])) / 2.0
            if avg_r > 0 and sum_norm / avg_r < 0.3:
                used.add(jdx)
                break
        used.add(idx)
        unique_reps.append(idx)

    # ── Polar coordinates for each unique representative ──────────
    cluster_info = []
    for idx in unique_reps:
        mu = means[idx]
        radius = np.linalg.norm(mu)
        angle = np.degrees(np.arctan2(abs(mu[1]), abs(mu[0])))
        cluster_info.append((idx, radius, angle))
    cluster_info.sort(key=lambda x: x[1])  # ascending radius

    # ── Assign labels ─────────────────────────────────────────────
    labels = {}
    horizontal = [(idx, r, a) for idx, r, a in cluster_info if a < 30.0]
    vertical = [(idx, r, a) for idx, r, a in cluster_info if a >= 30.0]

    for idx, r, a in horizontal:
        labels[idx] = "word"

    if len(vertical) == 0:
        pass
    elif len(vertical) == 1:
        labels[vertical[0][0]] = "line"
    else:
        # Split vertical clusters at the biggest radius jump
        radii = [r for _, r, _ in vertical]
        ratios = [radii[k + 1] / radii[k] if radii[k] > 1e-9 else 1.0
                  for k in range(len(radii) - 1)]
        split = int(np.argmax(ratios))
        if ratios[split] < 1.3:
            for idx, r, a in vertical:
                labels[idx] = "line"
        else:
            for k, (idx, r, a) in enumerate(vertical):
                labels[idx] = "line" if k <= split else "block"

    # ── Fallbacks for unusual documents ───────────────────────────
    if not any(v == "word" for v in labels.values()) and cluster_info:
        labels[cluster_info[0][0]] = "word"
    if not any(v == "line" for v in labels.values()) and len(cluster_info) >= 2:
        for idx, _, _ in cluster_info:
            if labels.get(idx) != "word":
                labels[idx] = "line"
                break

    # ── Propagate labels to mirror partners ───────────────────────
    full_labels = {}
    for idx in active:
        mu = means[idx]
        best_match, best_dist = None, float("inf")
        for rep in unique_reps:
            d = min(np.linalg.norm(mu - means[rep]),
                    np.linalg.norm(mu + means[rep]))
            if d < best_dist:
                best_dist, best_match = d, rep
        full_labels[idx] = labels.get(best_match, "block")

    return full_labels


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Edge labelling (Bayes' MAP decision)
# ═══════════════════════════════════════════════════════════════════

def _label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping):
    """Label each edge with its semantic type via posterior probability.

    For each displacement vector x, the VBGMM computes:
        p(k | x) ∝ π_k · N(x | μ_k, Σ_k)
    and assigns x to the highest-posterior component (MAP rule).
    The edge inherits that component's semantic label.
    """
    if len(vectors_2d) == 0:
        return []

    predictions = model.predict(vectors_2d)
    labeled = []
    for edge_idx, (i_cc, j_cc) in enumerate(pruned_edges):
        # Forward vector is at even-indexed rows (due to symmetrisation)
        component = predictions[edge_idx * 2]
        label = cluster_labels.get(component, "block")
        labeled.append((i_cc, j_cc, label))
    return labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Hierarchical merging (3-pass union-find)
# ═══════════════════════════════════════════════════════════════════

def _union_bbox(bboxes: list[tuple]) -> tuple:
    """Smallest axis-aligned bbox enclosing all input bboxes."""
    xs = [b[0] for b in bboxes]
    ys = [b[1] for b in bboxes]
    x2s = [b[0] + b[2] for b in bboxes]
    y2s = [b[1] + b[3] for b in bboxes]
    x_min, y_min = min(xs), min(ys)
    return (x_min, y_min, max(x2s) - x_min, max(y2s) - y_min)


def _find(parent, x):
    """Union-find: find root with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    """Union-find: merge the sets containing a and b."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def _detect_words_in_line(binary_image, line_bbox, cc_bboxes,
                          min_word_gap_ratio=0.25):
    """Detect word boundaries within a text line via projection profiles.

    For each line, the column-wise ink-density profile is computed.
    Runs of zero-ink columns are classified as either:
        - kerning gaps (1–2 px, intra-character)  → ignored
        - word gaps (≥ 25% of median char height) → split point

    The threshold adapts per-line, handling justified text where word
    spacing varies.  For CJK text (no word-level whitespace), each
    character naturally becomes its own "word".

    Returns a list of word dicts with 'bbox' and 'components'.
    """
    lx, ly, lw, lh = line_bbox
    if lw <= 0 or lh <= 0 or len(cc_bboxes) == 0:
        return [{"bbox": line_bbox, "components": [{"bbox": b} for b in cc_bboxes]}]

    img_h, img_w = binary_image.shape[:2]
    y1, y2 = max(0, ly), min(img_h, ly + lh)
    x1, x2 = max(0, lx), min(img_w, lx + lw)
    if y2 <= y1 or x2 <= x1:
        return [{"bbox": line_bbox, "components": [{"bbox": b} for b in cc_bboxes]}]

    # Column-wise ink count (vertical projection profile)
    strip = binary_image[y1:y2, x1:x2]
    ink_profile = np.sum(strip == 0, axis=0)

    # Find runs of zero-ink columns
    is_gap = ink_profile == 0
    runs = []
    in_gap, gap_start = False, 0
    for col in range(len(is_gap)):
        if is_gap[col] and not in_gap:
            gap_start = col
            in_gap = True
        elif not is_gap[col] and in_gap:
            runs.append((gap_start, col, col - gap_start))
            in_gap = False

    if not runs:
        return [{"bbox": line_bbox, "components": [{"bbox": b} for b in cc_bboxes]}]

    # Adaptive threshold: word gaps ≥ 25% of median character height
    cc_heights = [b[3] for b in cc_bboxes]
    median_h = max(float(np.median(cc_heights)), 1.0)
    min_word_gap = max(3, int(min_word_gap_ratio * median_h))

    gap_widths = np.array([r[2] for r in runs])
    large_gaps = gap_widths[gap_widths >= min_word_gap]
    small_gaps = gap_widths[gap_widths < min_word_gap]

    # If bimodal, split at the midpoint between small/large gap distributions
    if len(large_gaps) > 0 and len(small_gaps) > 0:
        threshold = (small_gaps.max() + large_gaps.min()) / 2.0
    else:
        threshold = min_word_gap

    # Identify word-splitting gaps
    word_boundaries = []
    for gap_start_rel, gap_end_rel, gap_w in runs:
        if gap_w >= threshold:
            word_boundaries.append((x1 + gap_start_rel, x1 + gap_end_rel))

    if not word_boundaries:
        return [{"bbox": line_bbox, "components": [{"bbox": b} for b in cc_bboxes]}]

    # Build word slots between consecutive gaps
    slot_edges = [x1]
    for gl, gr in word_boundaries:
        slot_edges.append(gl)
        slot_edges.append(gr)
    slot_edges.append(x1 + lw)
    word_slots = [(slot_edges[s], slot_edges[s + 1])
                  for s in range(0, len(slot_edges) - 1, 2)]

    # Assign CCs to slots by center-x
    words = []
    for sl, sr in word_slots:
        slot_ccs = [b for b in cc_bboxes if sl <= b[0] + b[2] / 2.0 < sr]
        if slot_ccs:
            words.append({
                "bbox": _union_bbox(slot_ccs),
                "components": [{"bbox": b} for b in slot_ccs],
            })

    # Safety: any unassigned CCs become their own word
    assigned = {c["bbox"] for w in words for c in w["components"]}
    unassigned = [b for b in cc_bboxes if b not in assigned]
    if unassigned:
        words.append({
            "bbox": _union_bbox(unassigned),
            "components": [{"bbox": b} for b in unassigned],
        })

    words.sort(key=lambda w: w["bbox"][0])
    return words


def _split_line_group_into_lines(cc_bboxes, median_h):
    """Split a VBGMM line-group into individual text lines.

    A "line group" from the union-find may contain multiple text lines
    (since "line" edges connect vertically adjacent CCs).  This function
    clusters CCs by their vertical centre using the median character
    height as separation threshold.
    """
    if not cc_bboxes:
        return []

    centres_y = sorted([(b[1] + b[3] / 2.0, b) for b in cc_bboxes],
                       key=lambda x: x[0])
    gap_threshold = max(3, 0.7 * median_h)

    lines = [[centres_y[0][1]]]
    prev_y = centres_y[0][0]
    for cy, bbox in centres_y[1:]:
        if cy - prev_y > gap_threshold:
            lines.append([bbox])
        else:
            lines[-1].append(bbox)
        prev_y = cy
    return lines


def _hierarchical_merge(stats, centroids, labeled_edges, n_components,
                        binary_image=None):
    """Merge CCs into blocks → lines → words via 3-pass union-find.

    Pass 1: merge along "word" edges  → word groups
    Pass 2: merge word-groups along "line" edges  → line groups
    Pass 3: merge line-groups along "block" edges → final blocks

    Within each line, word detection uses projection profiles on the
    binary image (adapts to per-line spacing).

    Returns the full hierarchical structure (see module docstring).
    """
    n = n_components
    comp_dicts = []
    for i in range(n):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        comp_dicts.append({"bbox": (x, y, w, h),
                           "center": (float(centroids[i, 0]), float(centroids[i, 1]))})

    # ── Pass 1: word grouping ─────────────────────────────────────
    parent = list(range(n))
    for i_cc, j_cc, label in labeled_edges:
        if label == "word":
            _union(parent, i_cc, j_cc)

    word_groups_map = defaultdict(list)
    for idx in range(n):
        word_groups_map[_find(parent, idx)].append(idx)

    word_items = []
    word_gid = {}  # cc_index → word_group_index
    for gidx, (_, members) in enumerate(word_groups_map.items()):
        bboxes = [comp_dicts[m]["bbox"] for m in members]
        merged = _union_bbox(bboxes)
        word_items.append({
            "bbox": merged,
            "center": (merged[0] + merged[2] / 2, merged[1] + merged[3] / 2),
            "cc_members": members,
        })
        for m in members:
            word_gid[m] = gidx

    # ── Pass 2: line grouping ─────────────────────────────────────
    nw = len(word_items)
    parent = list(range(nw))
    for i_cc, j_cc, label in labeled_edges:
        if label == "line":
            a, b = word_gid.get(i_cc), word_gid.get(j_cc)
            if a is not None and b is not None and a != b:
                _union(parent, a, b)

    line_groups_map = defaultdict(list)
    for idx in range(nw):
        line_groups_map[_find(parent, idx)].append(idx)

    line_items = []
    line_gid = {}  # word_group_index → line_group_index
    for gidx, (_, wg_members) in enumerate(line_groups_map.items()):
        bboxes = [word_items[wg]["bbox"] for wg in wg_members]
        merged = _union_bbox(bboxes)
        all_ccs = []
        for wg in wg_members:
            all_ccs.extend(word_items[wg]["cc_members"])
        line_items.append({
            "bbox": merged,
            "center": (merged[0] + merged[2] / 2, merged[1] + merged[3] / 2),
            "cc_members": all_ccs,
            "wg_members": wg_members,
        })
        for wg in wg_members:
            line_gid[wg] = gidx

    # ── Pass 3: block grouping ────────────────────────────────────
    nl = len(line_items)
    parent = list(range(nl))
    for i_cc, j_cc, label in labeled_edges:
        if label == "block":
            wa, wb = word_gid.get(i_cc), word_gid.get(j_cc)
            if wa is not None and wb is not None:
                la, lb = line_gid.get(wa), line_gid.get(wb)
                if la is not None and lb is not None and la != lb:
                    _union(parent, la, lb)

    block_groups_map = defaultdict(list)
    for idx in range(nl):
        block_groups_map[_find(parent, idx)].append(idx)

    # ── Build hierarchical output ─────────────────────────────────
    all_heights = [comp_dicts[i]["bbox"][3] for i in range(n)]
    median_h = float(np.median(all_heights)) if all_heights else 10.0

    blocks = []
    for _, lg_members in block_groups_map.items():
        all_cc_in_block = []
        for lg in lg_members:
            all_cc_in_block.extend(line_items[lg]["cc_members"])
        all_cc_bboxes = [comp_dicts[cc]["bbox"] for cc in all_cc_in_block]
        block_bbox = _union_bbox(all_cc_bboxes)

        # Split into individual text lines by y-clustering
        text_lines = _split_line_group_into_lines(all_cc_bboxes, median_h)

        line_dicts = []
        for line_ccs in text_lines:
            line_bbox = _union_bbox(line_ccs)
            if binary_image is not None:
                words = _detect_words_in_line(binary_image, line_bbox, line_ccs)
            else:
                words = [{"bbox": b, "components": [{"bbox": b}]} for b in line_ccs]
                words.sort(key=lambda w: w["bbox"][0])
            line_dicts.append({"bbox": line_bbox, "words": words})

        line_dicts.sort(key=lambda l: l["bbox"][1])
        blocks.append({
            "bbox": block_bbox,
            "lines": line_dicts,
            "components": [{"bbox": b} for b in all_cc_bboxes],
        })

    return blocks


# ═══════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _validate_with_projection_profile(binary_image, blocks, min_gap_ratio=2.5):
    """Split blocks at large internal horizontal whitespace gaps.

    Only fires at gaps ≥ 2.5× median component height — far larger than
    normal inter-line whitespace (~1.2–1.5× char height).  Catches
    section breaks that the VBGMM may have missed.
    """
    median_heights = [c["bbox"][3] for b in blocks for c in b["components"]]
    if not median_heights:
        return blocks

    global_median_h = max(float(np.median(median_heights)), 1.0)
    min_gap = max(int(min_gap_ratio * global_median_h), 10)
    refined = []

    for block in blocks:
        bx, by, bw, bh = block["bbox"]
        if bw <= 0 or bh <= 0:
            refined.append(block)
            continue

        y1, y2 = max(0, by), min(binary_image.shape[0], by + bh)
        x1, x2 = max(0, bx), min(binary_image.shape[1], bx + bw)
        if y2 <= y1 or x2 <= x1:
            refined.append(block)
            continue

        region = binary_image[y1:y2, x1:x2]
        h_profile = np.sum(region == 0, axis=1)
        threshold = max(1, int(0.01 * bw))
        is_gap = h_profile <= threshold

        # Find large horizontal gaps
        gap_starts, gap_ends = [], []
        in_gap, gap_start = False, 0
        for row in range(len(is_gap)):
            if is_gap[row] and not in_gap:
                gap_start = row
                in_gap = True
            elif not is_gap[row] and in_gap:
                if row - gap_start >= min_gap:
                    gap_starts.append(gap_start)
                    gap_ends.append(row)
                in_gap = False
        if in_gap and len(is_gap) - gap_start >= min_gap:
            gap_starts.append(gap_start)
            gap_ends.append(len(is_gap))

        if not gap_starts:
            refined.append(block)
            continue

        # Split block at each large gap
        boundaries = [0]
        for gs, ge in zip(gap_starts, gap_ends):
            boundaries.extend([gs, ge])
        boundaries.append(bh)

        sub_blocks = []
        for s in range(0, len(boundaries) - 1, 2):
            sub_y1, sub_y2 = boundaries[s], boundaries[s + 1]
            if sub_y2 - sub_y1 < 2:
                continue
            sub_comps = [c for c in block["components"]
                         if by + sub_y1 <= c["bbox"][1] + c["bbox"][3] / 2.0 < by + sub_y2]
            if sub_comps:
                sub_blocks.append({
                    "bbox": _union_bbox([c["bbox"] for c in sub_comps]),
                    "components": sub_comps,
                })
        refined.extend(sub_blocks if sub_blocks else [block])

    return refined


def _filter_blocks(blocks, img_shape, min_components=4, min_area=100):
    """Remove degenerate blocks (too few CCs, too small, or page-sized)."""
    img_area = img_shape[0] * img_shape[1]
    return [
        b for b in blocks
        if (len(b["components"]) >= min_components
            and b["bbox"][2] * b["bbox"][3] >= min_area
            and not (b["bbox"][2] * b["bbox"][3] > 0.98 * img_area
                     and len(blocks) > 1))
    ]


# ═══════════════════════════════════════════════════════════════════
# REGION-LEVEL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def _segment_region(stats, centroids, n, binary_image=None):
    """Run the full VBGMM pipeline on a single XY-cut region."""

    # Trivial cases
    if n == 0:
        return []
    if n == 1:
        bbox = (int(stats[0, 0]), int(stats[0, 1]),
                int(stats[0, 2]), int(stats[0, 3]))
        return [{"bbox": bbox,
                 "lines": [{"bbox": bbox, "words": [{"bbox": bbox, "components": [{"bbox": bbox}]}]}],
                 "components": [{"bbox": bbox}]}]

    # Build neighbourhood graph
    if n == 2:
        edges = [(0, 1)]
    else:
        try:
            edges = _delaunay_edges(centroids)
        except QhullError:
            edges = _fallback_knn_edges(centroids)

    # Displacement vectors
    vectors_2d, pruned_edges, edge_mapping = _compute_displacement_vectors(
        edges, centroids, stats)

    if len(vectors_2d) < 10:
        # Too few edges for a meaningful mixture → single block
        all_bboxes = [(int(stats[i, 0]), int(stats[i, 1]),
                       int(stats[i, 2]), int(stats[i, 3])) for i in range(n)]
        merged = _union_bbox(all_bboxes)
        comps = [{"bbox": b} for b in all_bboxes]
        words = [{"bbox": b, "components": [{"bbox": b}]} for b in all_bboxes]
        return [{"bbox": merged, "lines": [{"bbox": merged, "words": words}],
                 "components": comps}]

    # Fit VBGMM
    model = _fit_vbgmm(vectors_2d)

    # Classify clusters
    cluster_labels = _classify_clusters(model)
    if not cluster_labels:
        all_bboxes = [(int(stats[i, 0]), int(stats[i, 1]),
                       int(stats[i, 2]), int(stats[i, 3])) for i in range(n)]
        merged = _union_bbox(all_bboxes)
        comps = [{"bbox": b} for b in all_bboxes]
        words = [{"bbox": b, "components": [{"bbox": b}]} for b in all_bboxes]
        return [{"bbox": merged, "lines": [{"bbox": merged, "words": words}],
                 "components": comps}]

    # Label edges and merge hierarchically
    labeled_edges = _label_edges(model, cluster_labels, vectors_2d,
                                 pruned_edges, edge_mapping)
    return _hierarchical_merge(stats, centroids, labeled_edges, n, binary_image)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def segment(binary_image: np.ndarray) -> list[dict]:
    """Segment a binary page image into a block → line → word hierarchy.

    This is the only function external code needs to call.

    Args:
        binary_image: H × W uint8 array, ink = 0, background = 255.
                      Typically the output of Sauvola binarisation.

    Returns:
        List of block dicts (see module docstring for full schema).
    """
    inverted = cv2.bitwise_not(binary_image)

    # Step 0: extract connected components
    stats, centroids = _extract_connected_components(inverted)
    stats = stats[stats[:, 4] >= 4]         # drop noise (< 4px area)
    centroids = centroids[:len(stats)]
    n = len(stats)
    if n == 0:
        return []
    if n == 1:
        bbox = (int(stats[0, 0]), int(stats[0, 1]),
                int(stats[0, 2]), int(stats[0, 3]))
        return [{"bbox": bbox, "components": [{"bbox": bbox}]}]

    # Step 1: merge sub-character fragments
    stats, centroids = _merge_subcharacter_components(stats, centroids)
    n = len(stats)
    if n <= 1:
        if n == 0:
            return []
        bbox = (int(stats[0, 0]), int(stats[0, 1]),
                int(stats[0, 2]), int(stats[0, 3]))
        return [{"bbox": bbox, "components": [{"bbox": bbox}]}]

    # Step 2: XY-cut pre-segmentation
    xy_regions = _xy_cut_presegment(binary_image)

    # Assign each CC to a region by centroid location
    cc_to_region = np.zeros(n, dtype=np.int32)
    for cc_idx in range(n):
        cx, cy = centroids[cc_idx]
        for r_idx, (rx, ry, rw, rh) in enumerate(xy_regions):
            if rx <= cx < rx + rw and ry <= cy < ry + rh:
                cc_to_region[cc_idx] = r_idx
                break

    # Step 3: process each region independently
    all_blocks = []
    for region_idx in range(len(xy_regions)):
        mask = cc_to_region == region_idx
        if not np.any(mask):
            continue
        r_stats = stats[mask]
        r_centroids = centroids[mask]
        all_blocks.extend(_segment_region(r_stats, r_centroids,
                                          len(r_stats), binary_image))

    # Step 4: post-processing
    all_blocks = _validate_with_projection_profile(binary_image, all_blocks)
    all_blocks = _filter_blocks(all_blocks, binary_image.shape)

    return all_blocks


# ═══════════════════════════════════════════════════════════════════════════════
#
# ALGORITHM DESCRIPTION AND DESIGN CHOICES
#
# ═══════════════════════════════════════════════════════════════════════════════
#
# ─── 1. OVERVIEW ─────────────────────────────────────────────────────────────
#
# This module implements a fully statistical, training-free approach to
# geometric document segmentation.  It takes a binarised page image and
# produces a hierarchical decomposition: blocks (paragraphs) → lines →
# words → connected components.  No deep-learning model is used — only
# classical statistical inference.
#
#
# ─── 2. PIPELINE STEPS ──────────────────────────────────────────────────────
#
# Step 0 — Connected Component Extraction
#     OpenCV's two-pass algorithm (He et al., 2017) labels every ink blob.
#     Noise blobs (< 4 pixels) are discarded.
#
# Step 1 — Sub-character Merging
#     Many glyphs (CJK radicals, Latin diacritics) are split into separate
#     CCs by binarisation.  A cKDTree spatial index finds nearby CC pairs;
#     they are merged via union-find if:
#         • edge-to-edge gap < 0.35 × median CC height
#         • merged aspect ratio ∈ [0.25, 4.0]
#         • merged area < 6 × median CC area
#     This prevents downstream algorithms from treating one character as
#     multiple words.
#
# Step 2 — XY-Cut Pre-segmentation
#     Different page zones (code, prose, headers, tables) have different
#     gap distributions.  A global VBGMM would average them, losing
#     discriminative power.  The XY-cut recursively splits the page at
#     large whitespace gaps (≥ 3× median CC height for horizontal gaps)
#     into homogeneous regions.  Each region then gets its own VBGMM.
#
# Step 3a — Delaunay Neighbourhood Graph
#     The Delaunay triangulation over CC centroids produces a planar graph
#     whose edges connect spatially adjacent components.  It is a superset
#     of the nearest-neighbour graph and the minimum spanning tree, adapts
#     to local density, and runs in O(N log N).  A k-NN fallback handles
#     degenerate cases (collinear points, < 3 CCs).
#
# Step 3b — 2D Displacement Vectors
#     For each Delaunay edge (i, j), we compute d = (c_j − c_i) / h_med,
#     where h_med is the median CC height (scale normalisation).
#     The dataset is symmetrised: both d and −d are included, so that each
#     gap type maps to one pair of mirror Gaussians regardless of edge
#     orientation.  Edges longer than 6× h_med are pruned (convex-hull
#     artefacts).
#
# Step 4 — Variational Bayesian Gaussian Mixture Model (VBGMM)
#     A standard GMM requires a fixed K (number of clusters).  Some pages
#     have 2 gap types, others 5+.  The VBGMM places a Dirichlet Process
#     prior (α₀ = 0.1) over mixing weights, which drives unused component
#     weights to zero during variational inference.  We start with K_init
#     up to 15 and let the data decide K*.  Full 2×2 covariance matrices
#     capture the elliptical shape of each gap cluster (word gaps spread
#     horizontally, line gaps vertically).
#
#     This is the core statistical contribution: instead of hand-tuning
#     distance thresholds, we let a Bayesian mixture model learn the
#     gap structure of each document region.
#
# Step 5 — Cluster Classification (Polar Decomposition)
#     Each VBGMM mean μ = (μx, μy) is converted to polar coordinates:
#         radius = ‖μ‖           (gap magnitude)
#         angle  = atan2(|μy|, |μx|)  (gap direction)
#     Classification:
#         • angle < 30°  → horizontal → "word"  (character/word spacing)
#         • angle ≥ 30°  → vertical   → "line" or "block"
#     Among vertical clusters (sorted by radius), the largest radius jump
#     separates line-level from block-level gaps.  Mirror pairs are
#     deduplicated, and labels are propagated.
#
# Step 6 — Edge Labelling (Bayes' MAP)
#     Each displacement vector is assigned to the VBGMM component with
#     the highest posterior probability:  k* = argmax_k π_k N(x|μ_k,Σ_k).
#     The edge inherits that component's semantic label.
#
# Step 7 — Hierarchical Merging (3-Pass Union-Find)
#     Three sequential union-find passes respect the nesting order:
#         Pass 1: merge along "word" edges   → word groups
#         Pass 2: merge word-groups along "line" edges  → line groups
#         Pass 3: merge line-groups along "block" edges → final blocks
#     Within each line, word detection uses vertical projection profiles
#     on the binary image (column-wise ink density).  The bimodal gap
#     distribution (kerning vs word spaces) is split adaptively per line,
#     handling justified text and mixed scripts.
#
# Post-processing
#     • Projection-profile validation splits blocks at large internal
#       horizontal gaps (≥ 2.5× median CC height) the VBGMM may have missed.
#     • Degenerate blocks (< 4 CCs, < 100 px² area, or page-covering) are
#       removed.
#
#
# ─── 3. DESIGN CHOICES ──────────────────────────────────────────────────────
#
# Why 2D vectors instead of scalar distances?
#     Scalar distances lose direction.  In tabular layouts, horizontal cell
#     gaps and vertical row gaps often have similar magnitudes but orthogonal
#     directions.  The 2D model distinguishes them.
#
# Why VBGMM instead of fixed-K GMM or KDE?
#     The number of gap types is document-dependent.  The Dirichlet Process
#     prior auto-selects K, avoiding the need for manual tuning.  KDE-based
#     valley detection (our earlier approach in segmentation v1) is fragile:
#     it works well for unimodal/bimodal distributions but breaks on complex
#     multi-type gap structures.
#
# Why XY-cut before VBGMM?
#     A single global VBGMM averages gap distributions across regions with
#     fundamentally different typography (code vs prose vs tables).
#     Per-region fitting is more accurate.  The XY-cut only fires at large
#     structural gaps, preserving enough data per region for the VBGMM.
#
# Why Delaunay (not k-NN)?
#     Delaunay adapts to local density without a fixed k.  It is the dual
#     of the Voronoi diagram (Kise et al., 1998) and produces a natural
#     neighbourhood structure for irregularly spaced CCs.
#
# Why projection profiles for word detection (not VBGMM)?
#     The VBGMM classifies edges between CC centroids, which is too coarse
#     for word boundaries: overlapping bounding boxes make edge-based gap
#     measurement unreliable at character level.  Projection profiles
#     operate directly on pixel columns and produce clean bimodal gap
#     distributions (kerning 1–2 px vs word spaces 6–13 px), with a clear
#     separation threshold.
#
# Why sub-character merging?
#     CJK characters like 讠, 亻, 氵 are composed of spatially disjoint
#     strokes that become separate CCs after binarisation.  Without
#     merging, the VBGMM would treat intra-character gaps as word-level
#     gaps, producing character-level "words" instead of actual words.
#
#
# ─── 4. REFERENCES ──────────────────────────────────────────────────────────
#
# [1] O'Gorman, L. (1993). The Document Spectrum for Page Layout Analysis.
#     IEEE TPAMI, 15(11), 1162–1173.
#     — Docstrum method: k-NN + angle histograms.  Our approach generalises
#       Docstrum by replacing hand-crafted histograms with a 2D Bayesian
#       mixture model.
#
# [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
#     Springer. Chapter 10: Approximate Inference.
#     — Variational Bayesian GMM: automatic relevance determination.
#
# [3] Blei, D. M. & Jordan, M. I. (2006). Variational inference for
#     Dirichlet process mixtures. Bayesian Analysis, 1(1), 121–144.
#     — Theoretical foundation for the Dirichlet Process prior.
#
# [4] Kise, K., Sato, A., & Iwata, M. (1998). Segmentation of Page
#     Images Using the Area Voronoi Diagram.  CVIU, 70(3), 370–382.
#     — Voronoi-based segmentation; our Delaunay graph is the dual.
#
# [5] He, L. et al. (2017). The connected-component labeling problem:
#     A review.  Pattern Recognition, 70, 25–43.
#     — CC labelling algorithms; OpenCV uses an optimised two-pass variant.
#
# [6] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in
#     Python.  JMLR, 12, 2825–2830.
#     — BayesianGaussianMixture implementation.
#
# ═══════════════════════════════════════════════════════════════════════════════
