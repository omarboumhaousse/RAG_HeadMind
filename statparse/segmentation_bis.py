import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from scipy.spatial import Delaunay, QhullError
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = "/home/onyxia/work/RAG_Statap_ENSAE_2025_Headminds/granular_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_PATH = "/home/onyxia/work/RAG_Statap_ENSAE_2025_Headminds/pdfs/book_en_搬书匠-3473-Reactive Programming with RxJS-2015-英文版_page_021.pdf"


# ═══════════════════════════════════════════════════════════════════
# PDF TO IMAGE
# ═══════════════════════════════════════════════════════════════════

def pdf_to_image(pdf_path, dpi=300):
    """Convert first page of PDF to grayscale image."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        doc.close()
        return img
    except ImportError:
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
            img = np.array(images[0])
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img
        except ImportError:
            raise ImportError("Install PyMuPDF (fitz) or pdf2image to convert PDFs")


# ═══════════════════════════════════════════════════════════════════
# BINARIZATION (Sauvola)
# ═══════════════════════════════════════════════════════════════════

def binarize_sauvola(gray_image, window_size=51, k=0.2):
    """Sauvola binarization."""
    if window_size % 2 == 0:
        window_size += 1
    
    gray = gray_image.astype(np.float64)
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray * gray, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    
    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))
    
    binary = np.zeros_like(gray_image, dtype=np.uint8)
    binary[gray_image >= threshold] = 255
    binary[gray_image < threshold] = 0
    
    return binary


# ═══════════════════════════════════════════════════════════════════
# ALL PIPELINE FUNCTIONS (from the segmentation module)
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

    return np.array(new_stats_list, dtype=np.int32), np.array(new_centroids_list, dtype=np.float64)


def extract_connected_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return stats[1:], centroids[1:]


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


def detect_text_orientation(vectors_2d):
    if len(vectors_2d) < 4:
        return 0.0
    pca = PCA(n_components=2)
    pca.fit(vectors_2d)
    principal = pca.components_[0]
    return np.arctan2(principal[1], principal[0])


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
    return np.array(vectors, dtype=np.float64), pruned_edges, edge_mapping


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


def classify_clusters(model, vectors_2d, weight_threshold=0.01):
    weights = model.weights_
    means = model.means_
    active_indices = np.where(weights > weight_threshold)[0]

    if len(active_indices) == 0:
        return {}

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

    orientation = detect_text_orientation(vectors_2d)
    cluster_info = []
    for idx in unique_representatives:
        mu = means[idx]
        r = np.linalg.norm(mu)
        angle = np.arctan2(abs(mu[1]), abs(mu[0]))
        adjusted_angle = abs(angle - abs(orientation))
        cluster_info.append((idx, r, adjusted_angle))

    cluster_info.sort(key=lambda x: x[1])
    labels = {}

    if len(cluster_info) == 1:
        labels[cluster_info[0][0]] = "line"
    elif len(cluster_info) == 2:
        labels[cluster_info[0][0]] = "line"
        labels[cluster_info[1][0]] = "block"
    else:
        radii = [c[1] for c in cluster_info]
        ratios = []
        for k in range(len(radii) - 1):
            ratios.append(radii[k + 1] / radii[k] if radii[k] > 1e-9 else 1.0)

        if len(ratios) >= 2:
            sorted_ratio_indices = sorted(range(len(ratios)), key=lambda x: ratios[x], reverse=True)
            gap1 = sorted_ratio_indices[0]
            gap2 = sorted_ratio_indices[1] if len(sorted_ratio_indices) > 1 else gap1
            if gap1 > gap2:
                gap1, gap2 = gap2, gap1
            for k, (idx, r, angle_adj) in enumerate(cluster_info):
                if k <= gap1:
                    angle_deg = np.degrees(angle_adj)
                    labels[idx] = "line" if angle_deg > 55 else "word"
                elif k <= gap2:
                    labels[idx] = "line"
                else:
                    labels[idx] = "block"
        else:
            if ratios[0] > 2.0:
                angle_deg_0 = np.degrees(cluster_info[0][2])
                labels[cluster_info[0][0]] = "line" if angle_deg_0 > 55 else "word"
                labels[cluster_info[1][0]] = "block"
            else:
                labels[cluster_info[0][0]] = "line"
                labels[cluster_info[1][0]] = "line"

    has_line = any(v == "line" for v in labels.values())
    if not has_line:
        for idx, r, angle in cluster_info:
            if labels.get(idx) == "word":
                labels[idx] = "line"
                break

    full_labels = {}
    for idx in active_indices:
        mu = means[idx]
        best_match = None
        best_dist = float("inf")
        for rep_idx in unique_representatives:
            mu_rep = means[rep_idx]
            d = min(np.linalg.norm(mu - mu_rep), np.linalg.norm(mu + mu_rep))
            if d < best_dist:
                best_dist = d
                best_match = rep_idx
        full_labels[idx] = labels.get(best_match, "block") if best_match is not None else "block"

    return full_labels


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

    for (i, j, label) in labeled_edges:
        if label == "word":
            _uf_union(parent, rank, i, j)

    word_groups = defaultdict(set)
    for i in range(n_components):
        word_groups[_uf_find(parent, i)].add(i)

    wg_roots = list(word_groups.keys())
    wg_index = {r: idx for idx, r in enumerate(wg_roots)}
    line_parent = list(range(len(wg_roots)))
    line_rank = [0] * len(wg_roots)

    for (i, j, label) in labeled_edges:
        if label == "line":
            wg_i = _uf_find(parent, i)
            wg_j = _uf_find(parent, j)
            if wg_i != wg_j:
                _uf_union(line_parent, line_rank, wg_index[wg_i], wg_index[wg_j])

    line_groups = defaultdict(set)
    for wg_idx in range(len(wg_roots)):
        line_root = _uf_find(line_parent, wg_idx)
        for cc in word_groups[wg_roots[wg_idx]]:
            line_groups[line_root].add(cc)

    lg_roots = list(line_groups.keys())
    block_parent = list(range(len(lg_roots)))
    block_rank = [0] * len(lg_roots)

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
        blocks.append({"bbox": merged_bbox, "components": component_list})
    return blocks


def validate_with_projection_profile(binary_image, blocks, min_gap_ratio=0.3):
    median_heights = []
    for block in blocks:
        for comp in block["components"]:
            median_heights.append(comp["bbox"][3])
    if len(median_heights) == 0:
        return blocks

    global_median_h = max(float(np.median(median_heights)), 1.0)
    min_gap = max(int(min_gap_ratio * global_median_h), 2)
    refined_blocks = []

    for block in blocks:
        bx, by, bw, bh = block["bbox"]
        if bw <= 0 or bh <= 0:
            refined_blocks.append(block)
            continue

        y1 = max(0, by)
        y2 = min(binary_image.shape[0], by + bh)
        x1 = max(0, bx)
        x2 = min(binary_image.shape[1], bx + bw)
        if y2 <= y1 or x2 <= x1:
            refined_blocks.append(block)
            continue

        region = binary_image[y1:y2, x1:x2]
        h_profile = np.sum(region == 0, axis=1)
        threshold = max(1, int(0.01 * bw))
        is_gap = h_profile <= threshold

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
                sub_blocks.append({"bbox": sub_merged, "components": sub_components})

        if len(sub_blocks) > 0:
            refined_blocks.extend(sub_blocks)
        else:
            refined_blocks.append(block)

    return refined_blocks


def xy_cut_presegment(binary_image, min_width=50, min_height=50, ink_threshold_ratio=0.005):
    h, w = binary_image.shape[:2]

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
        center = len(gap_mask) / 2.0
        best = None
        best_score = -1
        edge_margin = len(gap_mask) * 0.05
        for gs, ge, gw in gaps:
            if gs < edge_margin or ge > len(gap_mask) - edge_margin:
                continue
            gap_center = (gs + ge) / 2.0
            distance_to_center = abs(gap_center - center)
            score = gw - 0.5 * distance_to_center / len(gap_mask) * gw
            if score > best_score:
                best_score = score
                best = (gs, ge)
        return best

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
        v_profile = np.sum(region == 0, axis=0)
        v_threshold = max(1, int(ink_threshold_ratio * region_h))
        v_gap = v_profile <= v_threshold
        best_v_gap = _find_best_gap(v_gap, min_gap=max(10, region_w // 20))
        if best_v_gap is not None:
            gs, ge = best_v_gap
            left = _recursive_cut(region_x, region_y, gs, region_h, depth + 1)
            right = _recursive_cut(region_x + ge, region_y, region_w - ge, region_h, depth + 1)
            return left + right
        h_profile = np.sum(region == 0, axis=1)
        h_threshold = max(1, int(ink_threshold_ratio * region_w))
        h_gap = h_profile <= h_threshold
        best_h_gap = _find_best_gap(h_gap, min_gap=max(5, region_h // 30))
        if best_h_gap is not None:
            gs, ge = best_h_gap
            top = _recursive_cut(region_x, region_y, region_w, gs, depth + 1)
            bottom = _recursive_cut(region_x, region_y + ge, region_w, region_h - ge, depth + 1)
            return top + bottom
        return [(region_x, region_y, region_w, region_h)]

    return _recursive_cut(0, 0, w, h)


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


def segment(binary_image: np.ndarray) -> list[dict]:
    """
    Full segmentation pipeline: from binary image to structured blocks.

    Input:  binary_image (H x W, uint8, ink = 0, background = 255)
            This is the output of the preprocessing stage (Sauvola binarization).
    Output: list of block dicts compatible with classification.py:
            [{"bbox": (x, y, w, h), "components": [{"bbox": ...}, ...]}, ...]
    """
    # ── Invert: ink (0) -> foreground (255) for OpenCV CC extraction ──
    inverted = cv2.bitwise_not(binary_image)

    # ── Step 1: Extract connected components ──────────────────────
    stats, centroids = extract_connected_components(inverted)

    # Filter noise: tiny CCs
    areas = stats[:, 4]
    valid_mask = areas >= 3
    stats = stats[valid_mask]
    centroids = centroids[valid_mask]
    n = len(stats)

    # ── Edge cases ────────────────────────────────────────────────
    if n == 0:
        return []

    if n == 1:
        x, y, w, h = int(stats[0, 0]), int(stats[0, 1]), int(stats[0, 2]), int(stats[0, 3])
        return [{"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]}]

    # ── Step 0: Sub-character merging (handles split radicals) ────
    stats, centroids = merge_subcharacter_components(stats, centroids)
    n = len(stats)

    if n == 0:
        return []
    if n == 1:
        x, y, w, h = int(stats[0, 0]), int(stats[0, 1]), int(stats[0, 2]), int(stats[0, 3])
        return [{"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]}]

    # ── Step 4 (pre): X-Y Cut pre-segmentation ───────────────────
    xy_regions = xy_cut_presegment(binary_image)

    # Assign each CC to a region based on its centroid
    cc_to_region = np.zeros(n, dtype=np.int32)
    for cc_idx in range(n):
        cx = centroids[cc_idx, 0]
        cy = centroids[cc_idx, 1]
        for r_idx, (rx, ry, rw, rh) in enumerate(xy_regions):
            if rx <= cx < rx + rw and ry <= cy < ry + rh:
                cc_to_region[cc_idx] = r_idx
                break

    # ── Process each region independently ─────────────────────────
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

        # Step 2: Neighbourhood graph
        if r_n == 2:
            edges = [(0, 1)]
        else:
            try:
                edges = generate_delaunay_edges(r_centroids)
            except QhullError:
                edges = _fallback_knn_edges(r_centroids)

        # Step 3: Displacement vectors
        vectors_2d, pruned_edges, edge_mapping = compute_displacement_vectors(
            edges, r_centroids, r_stats
        )

        if len(vectors_2d) < 6:
            all_bboxes = [(int(r_stats[i, 0]), int(r_stats[i, 1]),
                           int(r_stats[i, 2]), int(r_stats[i, 3])) for i in range(r_n)]
            merged = _union_bbox(all_bboxes)
            all_blocks.append({"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]})
            continue

        # Step 4: Fit VBGMM
        model = fit_vbgmm(vectors_2d)

        # Step 5: Classify clusters
        cluster_labels = classify_clusters(model, vectors_2d)

        if not cluster_labels:
            all_bboxes = [(int(r_stats[i, 0]), int(r_stats[i, 1]),
                           int(r_stats[i, 2]), int(r_stats[i, 3])) for i in range(r_n)]
            merged = _union_bbox(all_bboxes)
            all_blocks.append({"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]})
            continue

        # Step 6: Label edges
        labeled_edges = label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping)

        # Step 7: Hierarchical merge
        region_blocks = hierarchical_merge(r_stats, r_centroids, labeled_edges, r_n)
        all_blocks.extend(region_blocks)

    # ── Post-processing ───────────────────────────────────────────
    # Projection profile validation (split blocks with large internal gaps)
    all_blocks = validate_with_projection_profile(binary_image, all_blocks)

    # Filter degenerate blocks
    all_blocks = filter_blocks(all_blocks, binary_image.shape)

    return all_blocks


# # ═══════════════════════════════════════════════════════════════════
# # VISUALIZATION HELPERS
# # ═══════════════════════════════════════════════════════════════════

# def save_fig(fig, name):
#     path = os.path.join(OUTPUT_DIR, name)
#     fig.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  -> Saved: {path}")


# def draw_bboxes_on_image(image, stats, color=(0, 255, 0), thickness=1):
#     """Draw bounding boxes from stats array onto a color image."""
#     vis = image.copy()
#     if len(vis.shape) == 2:
#         vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
#     for i in range(len(stats)):
#         x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
#         cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
#     return vis


# # ═══════════════════════════════════════════════════════════════════
# # MAIN GRANULAR ANALYSIS
# # ═══════════════════════════════════════════════════════════════════

# def run_granular_analysis():
#     print("=" * 70)
#     print("GRANULAR ANALYSIS OF SEGMENTATION PIPELINE")
#     print("=" * 70)

#     # ─── Load PDF ─────────────────────────────────────────────────
#     print("\n[0] Loading PDF...")
#     gray = pdf_to_image(PDF_PATH)
#     print(f"    Image shape: {gray.shape}")

#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(gray, cmap='gray')
#     ax.set_title("Step 0: Original grayscale image")
#     ax.axis('off')
#     save_fig(fig, "00_original_grayscale.png")

#     # ─── Binarize ─────────────────────────────────────────────────
#     print("\n[1] Binarizing (Sauvola)...")
#     binary = binarize_sauvola(gray)
#     print(f"    Binary image: {binary.shape}, unique values: {np.unique(binary)}")
#     ink_pixels = np.sum(binary == 0)
#     total_pixels = binary.shape[0] * binary.shape[1]
#     print(f"    Ink pixels: {ink_pixels} ({100*ink_pixels/total_pixels:.2f}%)")

#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(binary, cmap='gray')
#     ax.set_title(f"Step 1: Binarized (Sauvola) — {ink_pixels} ink pixels ({100*ink_pixels/total_pixels:.2f}%)")
#     ax.axis('off')
#     save_fig(fig, "01_binarized.png")

#     # ─── Invert & Extract CCs ────────────────────────────────────
#     print("\n[2] Extracting connected components...")
#     inverted = cv2.bitwise_not(binary)
#     stats, centroids = extract_connected_components(inverted)
#     print(f"    Total CCs (before filtering): {len(stats)}")

#     # Filter noise
#     areas = stats[:, 4]
#     valid_mask = areas >= 3
#     stats_filtered = stats[valid_mask]
#     centroids_filtered = centroids[valid_mask]
#     print(f"    CCs after area>=3 filter: {len(stats_filtered)}")

#     # Stats about CC sizes
#     heights = stats_filtered[:, 3]
#     widths = stats_filtered[:, 2]
#     cc_areas = stats_filtered[:, 4]
#     print(f"    Height — min: {heights.min()}, median: {np.median(heights):.1f}, max: {heights.max()}")
#     print(f"    Width  — min: {widths.min()}, median: {np.median(widths):.1f}, max: {widths.max()}")
#     print(f"    Area   — min: {cc_areas.min()}, median: {np.median(cc_areas):.1f}, max: {cc_areas.max()}")

#     # Visualize all CCs
#     vis_cc = draw_bboxes_on_image(gray, stats_filtered, color=(0, 255, 0), thickness=1)
#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(cv2.cvtColor(vis_cc, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"Step 2: Connected Components — {len(stats_filtered)} CCs")
#     ax.axis('off')
#     save_fig(fig, "02_connected_components.png")

#     # Histogram of CC sizes
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     axes[0].hist(heights, bins=50, color='steelblue', edgecolor='black')
#     axes[0].set_title(f"CC Heights (median={np.median(heights):.1f})")
#     axes[0].axvline(np.median(heights), color='red', linestyle='--', label='median')
#     axes[0].legend()
#     axes[1].hist(widths, bins=50, color='coral', edgecolor='black')
#     axes[1].set_title(f"CC Widths (median={np.median(widths):.1f})")
#     axes[1].axvline(np.median(widths), color='red', linestyle='--', label='median')
#     axes[1].legend()
#     axes[2].hist(np.log10(cc_areas + 1), bins=50, color='green', edgecolor='black')
#     axes[2].set_title(f"CC log10(Area) (median={np.median(cc_areas):.1f})")
#     axes[2].axvline(np.log10(np.median(cc_areas) + 1), color='red', linestyle='--', label='median')
#     axes[2].legend()
#     plt.suptitle("Step 2b: CC Size Distributions")
#     plt.tight_layout()
#     save_fig(fig, "02b_cc_size_histograms.png")

#     # ─── Sub-character merging ────────────────────────────────────
#     print("\n[3] Sub-character merging...")
#     stats_merged, centroids_merged = merge_subcharacter_components(stats_filtered, centroids_filtered)
#     print(f"    CCs after merging: {len(stats_merged)} (was {len(stats_filtered)})")
#     print(f"    Merged {len(stats_filtered) - len(stats_merged)} components")

#     vis_merged = draw_bboxes_on_image(gray, stats_merged, color=(255, 0, 0), thickness=1)
#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(cv2.cvtColor(vis_merged, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"Step 3: After Sub-character Merging — {len(stats_merged)} CCs")
#     ax.axis('off')
#     save_fig(fig, "03_subchar_merged.png")

#     stats = stats_merged
#     centroids = centroids_merged
#     n = len(stats)

#     # ─── X-Y Cut Pre-segmentation ────────────────────────────────
#     print("\n[4] X-Y Cut pre-segmentation...")
#     xy_regions = xy_cut_presegment(binary)
#     print(f"    Found {len(xy_regions)} regions")
#     for i, (rx, ry, rw, rh) in enumerate(xy_regions):
#         print(f"      Region {i}: x={rx}, y={ry}, w={rw}, h={rh}")

#     vis_xy = gray.copy()
#     if len(vis_xy.shape) == 2:
#         vis_xy = cv2.cvtColor(vis_xy, cv2.COLOR_GRAY2BGR)
#     colors_xy = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
#                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
#     for i, (rx, ry, rw, rh) in enumerate(xy_regions):
#         color = colors_xy[i % len(colors_xy)]
#         cv2.rectangle(vis_xy, (rx, ry), (rx + rw, ry + rh), color, 3)
#         cv2.putText(vis_xy, f"R{i}", (rx + 10, ry + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(cv2.cvtColor(vis_xy, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"Step 4: X-Y Cut Regions — {len(xy_regions)} regions")
#     ax.axis('off')
#     save_fig(fig, "04_xy_cut_regions.png")

#     # ─── Assign CCs to regions ────────────────────────────────────
#     print("\n[5] Assigning CCs to X-Y regions...")
#     cc_to_region = np.zeros(n, dtype=np.int32)
#     for cc_idx in range(n):
#         cx = centroids[cc_idx, 0]
#         cy = centroids[cc_idx, 1]
#         for r_idx, (rx, ry, rw, rh) in enumerate(xy_regions):
#             if rx <= cx < rx + rw and ry <= cy < ry + rh:
#                 cc_to_region[cc_idx] = r_idx
#                 break

#     for r_idx in range(len(xy_regions)):
#         count = np.sum(cc_to_region == r_idx)
#         print(f"    Region {r_idx}: {count} CCs")

#     # ─── Process each region ──────────────────────────────────────
#     all_blocks = []

#     for region_idx in range(len(xy_regions)):
#         region_mask = cc_to_region == region_idx
#         region_cc_indices = np.where(region_mask)[0]

#         if len(region_cc_indices) == 0:
#             continue

#         r_stats = stats[region_cc_indices]
#         r_centroids = centroids[region_cc_indices]
#         r_n = len(r_stats)

#         print(f"\n[6] Processing Region {region_idx} ({r_n} CCs)...")

#         if r_n == 1:
#             x, y, w, h = int(r_stats[0, 0]), int(r_stats[0, 1]), int(r_stats[0, 2]), int(r_stats[0, 3])
#             all_blocks.append({"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]})
#             continue

#         # Step 2: Delaunay
#         print(f"    [6a] Building Delaunay triangulation...")
#         if r_n == 2:
#             edges = [(0, 1)]
#         else:
#             try:
#                 edges = generate_delaunay_edges(r_centroids)
#             except QhullError:
#                 edges = _fallback_knn_edges(r_centroids)
#         print(f"         {len(edges)} edges")

#         # Visualize Delaunay edges for this region
#         fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#         ax.imshow(gray, cmap='gray')
#         for (i, j) in edges:
#             p1 = r_centroids[i]
#             p2 = r_centroids[j]
#             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=0.3, alpha=0.5)
#         ax.scatter(r_centroids[:, 0], r_centroids[:, 1], c='red', s=2, zorder=5)
#         ax.set_title(f"Step 6a: Delaunay Edges — Region {region_idx} ({len(edges)} edges)")
#         ax.axis('off')
#         save_fig(fig, f"06a_delaunay_region_{region_idx}.png")

#         # Step 3: Displacement vectors
#         print(f"    [6b] Computing displacement vectors...")
#         vectors_2d, pruned_edges, edge_mapping = compute_displacement_vectors(edges, r_centroids, r_stats)
#         print(f"         {len(pruned_edges)} pruned edges, {len(vectors_2d)} vectors")

#         if len(vectors_2d) < 6:
#             all_bboxes = [(int(r_stats[i, 0]), int(r_stats[i, 1]),
#                            int(r_stats[i, 2]), int(r_stats[i, 3])) for i in range(r_n)]
#             merged = _union_bbox(all_bboxes)
#             all_blocks.append({"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]})
#             print(f"         Too few vectors, merging all into one block")
#             continue

#         # Visualize pruned edges
#         fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#         ax.imshow(gray, cmap='gray')
#         for (i, j) in pruned_edges:
#             p1 = r_centroids[i]
#             p2 = r_centroids[j]
#             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=0.3, alpha=0.5)
#         ax.set_title(f"Step 6b: Pruned Edges — Region {region_idx} ({len(pruned_edges)} edges)")
#         ax.axis('off')
#         save_fig(fig, f"06b_pruned_edges_region_{region_idx}.png")

#         # Visualize 2D displacement cloud
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=1, alpha=0.3, c='steelblue')
#         ax.set_xlabel("dx / median_h")
#         ax.set_ylabel("dy / median_h")
#         ax.set_title(f"Step 6b: 2D Displacement Cloud — Region {region_idx} ({len(vectors_2d)} vectors)")
#         ax.axhline(0, color='gray', linewidth=0.5)
#         ax.axvline(0, color='gray', linewidth=0.5)
#         ax.set_aspect('equal')
#         ax.grid(True, alpha=0.3)
#         save_fig(fig, f"06b_displacement_cloud_region_{region_idx}.png")

#         # Step 4: VBGMM
#         print(f"    [6c] Fitting VBGMM...")
#         model = fit_vbgmm(vectors_2d)
#         weights = model.weights_
#         means = model.means_
#         active = np.where(weights > 0.01)[0]
#         print(f"         {len(active)} active components (out of 15)")
#         for comp_idx in active:
#             print(f"           Component {comp_idx}: weight={weights[comp_idx]:.4f}, "
#                   f"mean=({means[comp_idx, 0]:.2f}, {means[comp_idx, 1]:.2f})")

#         # Visualize VBGMM clusters
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         predictions = model.predict(vectors_2d)
#         scatter_colors = plt.cm.tab20(predictions / max(predictions.max(), 1))
#         ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=scatter_colors, s=2, alpha=0.5)
#         for comp_idx in active:
#             mu = means[comp_idx]
#             ax.plot(mu[0], mu[1], 'k*', markersize=15, markeredgecolor='white', markeredgewidth=0.5)
#             ax.annotate(f"C{comp_idx}\nw={weights[comp_idx]:.3f}",
#                        (mu[0], mu[1]), fontsize=8, ha='center', va='bottom',
#                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
#         ax.set_xlabel("dx / median_h")
#         ax.set_ylabel("dy / median_h")
#         ax.set_title(f"Step 6c: VBGMM Clusters — Region {region_idx}")
#         ax.axhline(0, color='gray', linewidth=0.5)
#         ax.axvline(0, color='gray', linewidth=0.5)
#         ax.set_aspect('equal')
#         ax.grid(True, alpha=0.3)
#         save_fig(fig, f"06c_vbgmm_clusters_region_{region_idx}.png")

#         # Step 5: Classify
#         print(f"    [6d] Classifying clusters...")
#         cluster_labels = classify_clusters(model, vectors_2d)
#         for comp_idx, label in cluster_labels.items():
#             print(f"           Component {comp_idx} -> {label}")

#         # Step 6: Label edges
#         print(f"    [6e] Labelling edges...")
#         labeled_edges = label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping)

#         word_count = sum(1 for _, _, l in labeled_edges if l == "word")
#         line_count = sum(1 for _, _, l in labeled_edges if l == "line")
#         block_count = sum(1 for _, _, l in labeled_edges if l == "block")
#         print(f"         word: {word_count}, line: {line_count}, block: {block_count}")

#         # Visualize labeled edges
#         fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#         ax.imshow(gray, cmap='gray')
#         edge_colors = {"word": "green", "line": "blue", "block": "red"}
#         for (i, j, label) in labeled_edges:
#             p1 = r_centroids[i]
#             p2 = r_centroids[j]
#             ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
#                     color=edge_colors.get(label, 'gray'), linewidth=0.5, alpha=0.6)

#         # Legend
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], color='green', linewidth=2, label=f'word ({word_count})'),
#             Line2D([0], [0], color='blue', linewidth=2, label=f'line ({line_count})'),
#             Line2D([0], [0], color='red', linewidth=2, label=f'block ({block_count})'),
#         ]
#         ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
#         ax.set_title(f"Step 6e: Labeled Edges — Region {region_idx}")
#         ax.axis('off')
#         save_fig(fig, f"06e_labeled_edges_region_{region_idx}.png")

#         # Step 7: Hierarchical merge
#         print(f"    [6f] Hierarchical merging...")
#         region_blocks = hierarchical_merge(r_stats, r_centroids, labeled_edges, r_n)
#         print(f"         Produced {len(region_blocks)} blocks")

#         # Visualize region blocks
#         fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#         ax.imshow(gray, cmap='gray')
#         block_colors_list = plt.cm.Set1(np.linspace(0, 1, max(len(region_blocks), 1)))
#         for b_idx, block in enumerate(region_blocks):
#             bx, by, bw, bh = block["bbox"]
#             color = block_colors_list[b_idx % len(block_colors_list)]
#             rect = Rectangle((bx, by), bw, bh, linewidth=2, edgecolor=color, facecolor=(*color[:3], 0.1))
#             ax.add_patch(rect)
#             ax.text(bx + 5, by + 15, f"B{b_idx} ({len(block['components'])} CCs)",
#                    fontsize=7, color='white', bbox=dict(facecolor=color[:3], alpha=0.7))
#             # Draw component bboxes
#             for comp in block["components"]:
#                 cx, cy, cw, ch = comp["bbox"]
#                 rect_c = Rectangle((cx, cy), cw, ch, linewidth=0.5, edgecolor=color, facecolor='none')
#                 ax.add_patch(rect_c)
#         ax.set_title(f"Step 6f: Blocks from Region {region_idx} — {len(region_blocks)} blocks")
#         ax.axis('off')
#         save_fig(fig, f"06f_blocks_region_{region_idx}.png")

#         all_blocks.extend(region_blocks)

#     # ─── Post-processing ──────────────────────────────────────────
#     print(f"\n[7] Post-processing...")
#     print(f"    Total blocks before validation: {len(all_blocks)}")

#     # Projection profile validation
#     all_blocks = validate_with_projection_profile(binary, all_blocks)
#     print(f"    After projection profile validation: {len(all_blocks)}")

#     # Filter
#     all_blocks = filter_blocks(all_blocks, binary.shape)
#     print(f"    After filtering: {len(all_blocks)}")

#     # ─── Final visualization ──────────────────────────────────────
#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(gray, cmap='gray')
#     final_colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_blocks), 1)))
#     for b_idx, block in enumerate(all_blocks):
#         bx, by, bw, bh = block["bbox"]
#         color = final_colors[b_idx % len(final_colors)]
#         rect = Rectangle((bx, by), bw, bh, linewidth=2, edgecolor=color, facecolor=(*color[:3], 0.15))
#         ax.add_patch(rect)
#         ax.text(bx + 3, by + 12, f"Block {b_idx}\n{len(block['components'])} CCs",
#                fontsize=6, color='white', bbox=dict(facecolor=color[:3], alpha=0.8))
#     ax.set_title(f"FINAL: {len(all_blocks)} blocks detected")
#     ax.axis('off')
#     save_fig(fig, "07_final_result.png")

#     # Also: a version with ALL component bboxes visible
#     fig, ax = plt.subplots(1, 1, figsize=(12, 16))
#     ax.imshow(gray, cmap='gray')
#     for b_idx, block in enumerate(all_blocks):
#         bx, by, bw, bh = block["bbox"]
#         color = final_colors[b_idx % len(final_colors)]
#         rect = Rectangle((bx, by), bw, bh, linewidth=2, edgecolor=color, facecolor=(*color[:3], 0.05))
#         ax.add_patch(rect)
#         for comp in block["components"]:
#             cx, cy, cw, ch = comp["bbox"]
#             rect_c = Rectangle((cx, cy), cw, ch, linewidth=0.5, edgecolor=color, facecolor='none')
#             ax.add_patch(rect_c)
#     ax.set_title(f"FINAL (with components): {len(all_blocks)} blocks, "
#                  f"{sum(len(b['components']) for b in all_blocks)} components")
#     ax.axis('off')
#     save_fig(fig, "07_final_with_components.png")

#     # Summary
#     print("\n" + "=" * 70)
#     print("SUMMARY")
#     print("=" * 70)
#     print(f"  Total blocks: {len(all_blocks)}")
#     for b_idx, block in enumerate(all_blocks):
#         bx, by, bw, bh = block["bbox"]
#         print(f"  Block {b_idx}: bbox=({bx},{by},{bw},{bh}), "
#               f"{len(block['components'])} components")
#     print(f"\n  All outputs saved to: {OUTPUT_DIR}")
#     print("=" * 70)


# if __name__ == "__main__":
#     run_granular_analysis()
