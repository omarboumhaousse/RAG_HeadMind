

import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay, QhullError
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# ═══════════════════════════════════════════════════════════════════
# STEP 0 — Sub-character merging (handles split radicals in CJK)
# ═══════════════════════════════════════════════════════════════════

def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, rank, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1


def merge_subcharacter_components(stats, centroids):
    n = len(stats)
    if n <= 1:
        return stats, centroids

    median_h = max(float(np.median(stats[:, 3])), 1.0)
    median_w = max(float(np.median(stats[:, 2])), 1.0)
    median_area = median_h * median_w

    max_gap = 0.35 * median_h

    parent = list(range(n))
    rank = [0] * n

    bboxes = []
    for i in range(n):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        bboxes.append((x, y, x + w, y + h))

    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=max_gap + max(median_h, median_w))

    for (i, j) in pairs:
        ri, rj = _find(parent, i), _find(parent, j)
        if ri == rj:
            continue

        x1_i, y1_i, x2_i, y2_i = bboxes[i]
        x1_j, y1_j, x2_j, y2_j = bboxes[j]

        gap_x = max(0, max(x1_i, x1_j) - min(x2_i, x2_j))
        gap_y = max(0, max(y1_i, y1_j) - min(y2_i, y2_j))

        if gap_x > max_gap or gap_y > max_gap:
            continue

        merged_x1 = min(x1_i, x1_j)
        merged_y1 = min(y1_i, y1_j)
        merged_x2 = max(x2_i, x2_j)
        merged_y2 = max(y2_i, y2_j)
        merged_w = merged_x2 - merged_x1
        merged_h = merged_y2 - merged_y1

        if merged_w == 0 or merged_h == 0:
            continue

        aspect = merged_w / merged_h
        if aspect < 0.25 or aspect > 4.0:
            continue

        merged_area = merged_w * merged_h
        if merged_area > 6.0 * median_area:
            continue

        _union(parent, rank, i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[_find(parent, i)].append(i)

    new_stats_list = []
    new_centroids_list = []

    for root, members in groups.items():
        xs = [int(stats[m, 0]) for m in members]
        ys = [int(stats[m, 1]) for m in members]
        x2s = [int(stats[m, 0]) + int(stats[m, 2]) for m in members]
        y2s = [int(stats[m, 1]) + int(stats[m, 3]) for m in members]
        total_area = sum(int(stats[m, 4]) for m in members)

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(x2s), max(y2s)
        w = x_max - x_min
        h = y_max - y_min

        new_stats_list.append([x_min, y_min, w, h, total_area])
        new_centroids_list.append([x_min + w / 2.0, y_min + h / 2.0])

    new_stats = np.array(new_stats_list, dtype=np.int32)
    new_centroids = np.array(new_centroids_list, dtype=np.float64)

    return new_stats, new_centroids


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Connected component extraction
# ═══════════════════════════════════════════════════════════════════

def extract_connected_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    filtered_stats = stats[1:]
    filtered_centroids = centroids[1:]
    return filtered_stats, filtered_centroids


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Neighbourhood graph
# ═══════════════════════════════════════════════════════════════════

def generate_delaunay_edges(centroids):
    tri = Delaunay(centroids)
    unique_edges = set()
    for simplex in tri.simplices:
        for k in range(3):
            edge = tuple(sorted((simplex[k], simplex[(k + 1) % 3])))
            unique_edges.add(edge)
    return list(unique_edges)


def _fallback_knn_edges(centroids):
    n = len(centroids)
    k = min(6, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroids)
    _, indices = nbrs.kneighbors(centroids)
    unique_edges = set()
    for i in range(n):
        for j in indices[i]:
            if i != j:
                unique_edges.add(tuple(sorted((i, j))))
    return list(unique_edges)


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — 2D displacement vectors with adaptive pruning
# ═══════════════════════════════════════════════════════════════════

def detect_text_orientation(vectors_2d):
    if len(vectors_2d) < 4:
        return 0.0
    pca = PCA(n_components=2)
    pca.fit(vectors_2d)
    principal = pca.components_[0]
    angle = np.arctan2(principal[1], principal[0])
    return angle


def compute_displacement_vectors(edges, centroids, stats):
    median_h = max(float(np.median(stats[:, 3])), 1.0)

    adj = defaultdict(list)
    edge_lengths = {}
    for (i, j) in edges:
        d = np.linalg.norm(centroids[j] - centroids[i])
        edge_lengths[(i, j)] = d
        adj[i].append(d)
        adj[j].append(d)

    local_medians = {}
    for node, lengths in adj.items():
        local_medians[node] = np.median(lengths) if len(lengths) > 0 else 0.0

    pruned_edges = []
    vectors = []
    edge_mapping = []

    for idx, (i, j) in enumerate(edges):
        d = edge_lengths[(i, j)]
        local_thresh = 5.0 * max(local_medians.get(i, 0), local_medians.get(j, 0))
        global_thresh = 15.0 * median_h

        if d > min(local_thresh, global_thresh):
            continue
        if d < 1e-9:
            continue

        dx = (centroids[j][0] - centroids[i][0]) / median_h
        dy = (centroids[j][1] - centroids[i][1]) / median_h

        edge_idx = len(pruned_edges)
        pruned_edges.append((i, j))

        vectors.append([dx, dy])
        edge_mapping.append(edge_idx)
        vectors.append([-dx, -dy])
        edge_mapping.append(edge_idx)

    if len(vectors) == 0:
        return np.empty((0, 2)), [], []

    vectors_2d = np.array(vectors, dtype=np.float64)
    return vectors_2d, pruned_edges, edge_mapping


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — VBGMM fitting
# ═══════════════════════════════════════════════════════════════════

def fit_vbgmm(vectors_2d):
    model = BayesianGaussianMixture(
        n_components=15,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0,
        max_iter=500,
        n_init=5,
        random_state=42,
    )
    model.fit(vectors_2d)
    return model


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Adaptive cluster classification
# ═══════════════════════════════════════════════════════════════════

def classify_clusters(model, vectors_2d, weight_threshold=0.01):
    weights = model.weights_
    means = model.means_

    active_indices = np.where(weights > weight_threshold)[0]

    if len(active_indices) == 0:
        return {}

    # Deduplicate mirror pairs
    used = set()
    unique_representatives = []

    for idx in active_indices:
        if idx in used:
            continue
        mu = means[idx]
        for jdx in active_indices:
            if jdx <= idx or jdx in used:
                continue
            mu_j = means[jdx]
            sum_norm = np.linalg.norm(mu + mu_j)
            avg_radius = (np.linalg.norm(mu) + np.linalg.norm(mu_j)) / 2.0
            if avg_radius > 0 and sum_norm / avg_radius < 0.3:
                used.add(jdx)
                break
        used.add(idx)
        unique_representatives.append(idx)

    if len(unique_representatives) == 0:
        return {}

    # Detect orientation
    orientation = detect_text_orientation(vectors_2d)

    # Compute radius and angle for each unique representative
    cluster_info = []
    for idx in unique_representatives:
        mu = means[idx]
        r = np.linalg.norm(mu)
        angle = np.arctan2(abs(mu[1]), abs(mu[0]))

        # Rotate angle by detected orientation to normalize
        adjusted_angle = angle - abs(orientation)
        adjusted_angle = abs(adjusted_angle)

        cluster_info.append((idx, r, adjusted_angle))

    # Sort by radius
    cluster_info.sort(key=lambda x: x[1])

    # Data-driven classification using radius gaps
    labels = {}

    if len(cluster_info) == 1:
        labels[cluster_info[0][0]] = "line"
    elif len(cluster_info) == 2:
        labels[cluster_info[0][0]] = "line"
        labels[cluster_info[1][0]] = "block"
    else:
        radii = [c[1] for c in cluster_info]

        # Find ratio gaps between consecutive sorted radii
        ratios = []
        for k in range(len(radii) - 1):
            if radii[k] > 1e-9:
                ratios.append(radii[k + 1] / radii[k])
            else:
                ratios.append(1.0)

        # Find the largest gap(s) to define boundaries
        if len(ratios) >= 2:
            sorted_ratio_indices = sorted(range(len(ratios)),
                                          key=lambda x: ratios[x],
                                          reverse=True)
            gap1 = sorted_ratio_indices[0]
            gap2 = sorted_ratio_indices[1] if len(sorted_ratio_indices) > 1 else gap1

            if gap1 > gap2:
                gap1, gap2 = gap2, gap1

            # Use angle as secondary discriminator
            for k, (idx, r, angle_adj) in enumerate(cluster_info):
                if k <= gap1:
                    # Smallest radii cluster: use angle
                    angle_deg = np.degrees(angle_adj)
                    if angle_deg > 55:
                        labels[idx] = "line"
                    else:
                        labels[idx] = "word"
                elif k <= gap2:
                    labels[idx] = "line"
                else:
                    labels[idx] = "block"
        else:
            # Only one ratio
            if ratios[0] > 2.0:
                angle_deg_0 = np.degrees(cluster_info[0][2])
                if angle_deg_0 > 55:
                    labels[cluster_info[0][0]] = "line"
                else:
                    labels[cluster_info[0][0]] = "word"
                labels[cluster_info[1][0]] = "block"
            else:
                labels[cluster_info[0][0]] = "line"
                labels[cluster_info[1][0]] = "line"

    # Ensure at least one "line" label exists
    has_line = any(v == "line" for v in labels.values())
    if not has_line:
        for idx, r, angle in cluster_info:
            if labels.get(idx) == "word":
                labels[idx] = "line"
                break

    # Propagate labels to mirror partners
    full_labels = {}
    for idx in active_indices:
        mu = means[idx]
        best_match = None
        best_dist = float("inf")
        for rep_idx in unique_representatives:
            mu_rep = means[rep_idx]
            d_direct = np.linalg.norm(mu - mu_rep)
            d_mirror = np.linalg.norm(mu + mu_rep)
            d = min(d_direct, d_mirror)
            if d < best_dist:
                best_dist = d
                best_match = rep_idx
        if best_match is not None and best_match in labels:
            full_labels[idx] = labels[best_match]
        else:
            full_labels[idx] = "block"

    return full_labels


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Edge labelling
# ═══════════════════════════════════════════════════════════════════

def label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping):
    if len(vectors_2d) == 0:
        return []

    predictions = model.predict(vectors_2d)

    labeled = []
    for edge_idx, (i_cc, j_cc) in enumerate(pruned_edges):
        row_idx = edge_idx * 2
        predicted_component = predictions[row_idx]
        label = cluster_labels.get(predicted_component, "block")
        labeled.append((i_cc, j_cc, label))

    return labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Hierarchical merging (union-find)
# ═══════════════════════════════════════════════════════════════════

def _union_bbox(bboxes):
    xs = [b[0] for b in bboxes]
    ys = [b[1] for b in bboxes]
    x2s = [b[0] + b[2] for b in bboxes]
    y2s = [b[1] + b[3] for b in bboxes]
    x_min, y_min = min(xs), min(ys)
    return (x_min, y_min, max(x2s) - x_min, max(y2s) - y_min)


def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, rank, a, b):
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1


def hierarchical_merge(stats, centroids, labeled_edges, n_components):
    parent = list(range(n_components))
    rank = [0] * n_components

    # Pass 1: word edges
    for (i, j, label) in labeled_edges:
        if label == "word":
            _uf_union(parent, rank, i, j)

    # Build word groups
    word_groups = defaultdict(set)
    for i in range(n_components):
        word_groups[_uf_find(parent, i)].add(i)

    # Pass 2: line edges
    # Map each CC to its word group root
    wg_parent = {}
    wg_roots = list(word_groups.keys())
    wg_index = {r: idx for idx, r in enumerate(wg_roots)}
    line_parent = list(range(len(wg_roots)))
    line_rank = [0] * len(wg_roots)

    for (i, j, label) in labeled_edges:
        if label == "line":
            wg_i = _uf_find(parent, i)
            wg_j = _uf_find(parent, j)
            if wg_i != wg_j:
                idx_i = wg_index[wg_i]
                idx_j = wg_index[wg_j]
                _uf_union(line_parent, line_rank, idx_i, idx_j)

    # Build line groups
    line_groups = defaultdict(set)
    for wg_idx in range(len(wg_roots)):
        line_root = _uf_find(line_parent, wg_idx)
        for cc in word_groups[wg_roots[wg_idx]]:
            line_groups[line_root].add(cc)

    # Pass 3: block edges
    lg_roots = list(line_groups.keys())
    lg_index = {r: idx for idx, r in enumerate(lg_roots)}
    block_parent = list(range(len(lg_roots)))
    block_rank = [0] * len(lg_roots)

    # Map each CC to its line group root
    cc_to_lg = {}
    for lg_idx, lg_root in enumerate(lg_roots):
        for cc in line_groups[lg_root]:
            cc_to_lg[cc] = lg_idx

    for (i, j, label) in labeled_edges:
        if label == "block":
            lg_i = cc_to_lg.get(i)
            lg_j = cc_to_lg.get(j)
            if lg_i is not None and lg_j is not None and lg_i != lg_j:
                _uf_union(block_parent, block_rank, lg_i, lg_j)

    # Build final blocks
    final_groups = defaultdict(set)
    for lg_idx in range(len(lg_roots)):
        block_root = _uf_find(block_parent, lg_idx)
        for cc in line_groups[lg_roots[lg_idx]]:
            final_groups[block_root].add(cc)

    blocks = []
    for group_root, members in final_groups.items():
        member_bboxes = []
        component_list = []
        for cc_idx in members:
            x = int(stats[cc_idx, 0])
            y = int(stats[cc_idx, 1])
            w = int(stats[cc_idx, 2])
            h = int(stats[cc_idx, 3])
            bbox = (x, y, w, h)
            member_bboxes.append(bbox)
            component_list.append({"bbox": bbox})

        if len(member_bboxes) == 0:
            continue

        merged_bbox = _union_bbox(member_bboxes)
        blocks.append({
            "bbox": merged_bbox,
            "components": component_list,
        })

    return blocks


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — Projection profile validation
# ═══════════════════════════════════════════════════════════════════

def validate_with_projection_profile(binary_image, blocks, min_gap_ratio=0.3):
    median_heights = []
    for block in blocks:
        for comp in block["components"]:
            median_heights.append(comp["bbox"][3])

    if len(median_heights) == 0:
        return blocks

    global_median_h = max(float(np.median(median_heights)), 1.0)
    min_gap = int(min_gap_ratio * global_median_h)
    min_gap = max(min_gap, 2)

    refined_blocks = []

    for block in blocks:
        bx, by, bw, bh = block["bbox"]

        if bw <= 0 or bh <= 0:
            refined_blocks.append(block)
            continue

        # Extract region
        y1 = max(0, by)
        y2 = min(binary_image.shape[0], by + bh)
        x1 = max(0, bx)
        x2 = min(binary_image.shape[1], bx + bw)

        if y2 <= y1 or x2 <= x1:
            refined_blocks.append(block)
            continue

        region = binary_image[y1:y2, x1:x2]

        # Horizontal projection profile (ink = 0 in original, so count 0s)
        h_profile = np.sum(region == 0, axis=1)

        # Find gaps (rows with zero or near-zero ink)
        threshold = max(1, int(0.01 * bw))
        is_gap = h_profile <= threshold

        # Find gap runs longer than min_gap
        gap_starts = []
        gap_ends = []
        in_gap = False
        gap_start = 0

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

        if len(gap_starts) == 0:
            refined_blocks.append(block)
            continue

        # Split the block at gap positions
        split_boundaries = [0]
        for gs, ge in zip(gap_starts, gap_ends):
            split_boundaries.append(gs)
            split_boundaries.append(ge)
        split_boundaries.append(bh)

        sub_blocks = []
        for s in range(0, len(split_boundaries) - 1, 2):
            sub_y1 = split_boundaries[s]
            sub_y2 = split_boundaries[s + 1]

            if sub_y2 - sub_y1 < 2:
                continue

            sub_components = []
            for comp in block["components"]:
                cx, cy, cw, ch = comp["bbox"]
                comp_center_y = cy + ch / 2.0
                if by + sub_y1 <= comp_center_y < by + sub_y2:
                    sub_components.append(comp)

            if len(sub_components) > 0:
                sub_bboxes = [c["bbox"] for c in sub_components]
                sub_merged = _union_bbox(sub_bboxes)
                sub_blocks.append({
                    "bbox": sub_merged,
                    "components": sub_components,
                })

        if len(sub_blocks) > 0:
            refined_blocks.extend(sub_blocks)
        else:
            refined_blocks.append(block)

    return refined_blocks


# ═══════════════════════════════════════════════════════════════════
# STEP 9 — X-Y Cut for macro-level structure
# ═══════════════════════════════════════════════════════════════════

def xy_cut_presegment(binary_image, min_width=50, min_height=50, ink_threshold_ratio=0.005):
    h, w = binary_image.shape[:2]

    def _recursive_cut(region_x, region_y, region_w, region_h, depth=0):
        if region_w < min_width or region_h < min_height or depth > 20:
            return [(region_x, region_y, region_w, region_h)]

        y1 = max(0, region_y)
        y2 = min(h, region_y + region_h)
        x1 = max(0, region_x)
        x2 = min(w, region_x + region_w)

        if y2 <= y1 or x2 <= x1:
            return [(region_x, region_y, region_w, region_h)]

        region = binary_image[y1:y2, x1:x2]

        # Try vertical cut (split into left/right columns)
        v_profile = np.sum(region == 0, axis=0)
        v_threshold = max(1, int(ink_threshold_ratio * region_h))
        v_gap = v_profile <= v_threshold

        best_v_gap = _find_best_gap(v_gap, min_gap=max(10, region_w // 20))

        if best_v_gap is not None:
            gs, ge = best_v_gap
            left_results = _recursive_cut(region_x, region_y, gs, region_h, depth + 1)
            right_results = _recursive_cut(region_x + ge, region_y, region_w - ge, region_h, depth + 1)
            return left_results + right_results

        # Try horizontal cut (split into top/bottom)
        h_profile = np.sum(region == 0, axis=1)
        h_threshold = max(1, int(ink_threshold_ratio * region_w))
        h_gap = h_profile <= h_threshold

        best_h_gap = _find_best_gap(h_gap, min_gap=max(5, region_h // 30))

        if best_h_gap is not None:
            gs, ge = best_h_gap
            top_results = _recursive_cut(region_x, region_y, region_w, gs, depth + 1)
            bottom_results = _recursive_cut(region_x, region_y + ge, region_w, region_h - ge, depth + 1)
            return top_results + bottom_results

        return [(region_x, region_y, region_w, region_h)]

    return _recursive_cut(0, 0, w, h)


def _find_best_gap(gap_mask, min_gap=10):
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

    if len(gaps) == 0:
        return None

    # Prefer gaps near the center and wide
    center = len(gap_mask) / 2.0
    best = None
    best_score = -1
    for gs, ge, gw in gaps:
        gap_center = (gs + ge) / 2.0
        distance_to_center = abs(gap_center - center)
        # Score: prefer wide gaps that are somewhat central
        # Avoid gaps at the very edges
        edge_margin = len(gap_mask) * 0.05
        if gs < edge_margin or ge > len(gap_mask) - edge_margin:
            continue
        score = gw - 0.5 * distance_to_center / len(gap_mask) * gw
        if score > best_score:
            best_score = score
            best = (gs, ge)

    return best


# ═══════════════════════════════════════════════════════════════════
# STEP 10 — Filter and clean up blocks
# ═══════════════════════════════════════════════════════════════════

def filter_blocks(blocks, image_shape, min_area=20):
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    filtered = []
    for block in blocks:
        bx, by, bw, bh = block["bbox"]
        area = bw * bh

        if area < min_area:
            continue

        if area > 0.98 * img_area and len(blocks) > 1:
            continue

        if len(block["components"]) == 0:
            continue

        filtered.append(block)

    return filtered


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def segment(binary_image):
    inverted = cv2.bitwise_not(binary_image)

    # Step 1: Extract connected components
    stats, centroids = extract_connected_components(inverted)

    # Filter noise
    areas = stats[:, 4]
    valid_mask = areas >= 3
    stats = stats[valid_mask]
    centroids = centroids[valid_mask]

    n = len(stats)

    if n == 0:
        return []

    if n == 1:
        x, y, w, h = int(stats[0, 0]), int(stats[0, 1]), int(stats[0, 2]), int(stats[0, 3])
        return [{"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]}]

    # Step 0: Sub-character merging for CJK
    stats, centroids = merge_subcharacter_components(stats, centroids)
    n = len(stats)

    if n == 0:
        return []

    if n == 1:
        x, y, w, h = int(stats[0, 0]), int(stats[0, 1]), int(stats[0, 2]), int(stats[0, 3])
        return [{"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]}]

    # Step 9 (early): X-Y cut pre-segmentation for macro structure
    xy_regions = xy_cut_presegment(binary_image)

    # Assign each CC to an X-Y region
    cc_to_region = np.zeros(n, dtype=np.int32)
    for cc_idx in range(n):
        cx = centroids[cc_idx, 0]
        cy = centroids[cc_idx, 1]
        best_region = 0
        for r_idx, (rx, ry, rw, rh) in enumerate(xy_regions):
            if rx <= cx < rx + rw and ry <= cy < ry + rh:
                best_region = r_idx
                break
        cc_to_region[cc_idx] = best_region

    # Process each region independently with the VBGMM pipeline
    all_blocks = []

    for region_idx in range(len(xy_regions)):
        region_mask = cc_to_region == region_idx
        region_cc_indices = np.where(region_mask)[0]

        if len(region_cc_indices) == 0:
            continue

        r_stats = stats[region_cc_indices]
        r_centroids = centroids[region_cc_indices]
        r_n = len(r_stats)

        if r_n == 1:
            x, y, w, h = int(r_stats[0, 0]), int(r_stats[0, 1]), int(r_stats[0, 2]), int(r_stats[0, 3])
            all_blocks.append({"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]})
            continue

        # Step 2: neighbourhood graph
        if r_n == 2:
            edges = [(0, 1)]
        else:
            try:
                edges = generate_delaunay_edges(r_centroids)
            except QhullError:
                edges = _fallback_knn_edges(r_centroids)

        # Step 3: displacement vectors
        vectors_2d, pruned_edges, edge_mapping = compute_displacement_vectors(
            edges, r_centroids, r_stats
        )

        if len(vectors_2d) < 6:
            all_bboxes = [(int(r_stats[i, 0]), int(r_stats[i, 1]),
                           int(r_stats[i, 2]), int(r_stats[i, 3])) for i in range(r_n)]
            merged = _union_bbox(all_bboxes)
            all_blocks.append({"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]})
            continue

        # Step 4: VBGMM
        model = fit_vbgmm(vectors_2d)

        # Step 5: classify
        cluster_labels = classify_clusters(model, vectors_2d)

        # Step 6: label edges
        labeled_edges = label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping)

        # Step 7: hierarchical merge
        region_blocks = hierarchical_merge(r_stats, r_centroids, labeled_edges, r_n)

        all_blocks.extend(region_blocks)

    # Step 8: projection profile validation
    all_blocks = validate_with_projection_profile(binary_image, all_blocks)

    # Step 10: filter
    all_blocks = filter_blocks(all_blocks, binary_image.shape)

    return all_blocks