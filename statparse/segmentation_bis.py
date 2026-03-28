
"""
Geometric segmentation of a binarized document page using a 2D Variational
Bayesian Gaussian Mixture Model (VBGMM) on inter component displacement vectors.

Input:  preprocessed binary image (H x W, ink = 0, background = 255)
Output: list of block dicts, each with:
            "bbox"       — (x, y, w, h) tight bounding box
            "components" — list of child dicts with "bbox" keys

Method overview:
    1. Extract connected components (ink blobs) from the page.
    2. Build a Delaunay neighbourhood graph over component centroids.
    3. Compute 2D displacement vectors (dx, dy) for every edge, normalized
       by the median component height for scale invariance.
    4. Fit a Variational Bayesian GMM with K_init = 15 components and a
       Dirichlet process prior => the model automatically prunes unused
       clusters, yielding the optimal number K* of gap types.
    5. Classify each surviving Gaussian into a semantic level (word / line /
       block) using the polar decomposition of its mean vector (angle for
       direction, radius for magnitude).
    6. Label every edge in the graph with its semantic level via Bayes'
       decision rule (maximum a posteriori assignment).
    7. Hierarchically merge components using three union-find passes:
       words (merge along "word" edges), lines (merge word-groups along
       "line" edges), blocks (merge line-groups along "block" edges).
"""

import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay, QhullError
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Connected component extraction
# ═══════════════════════════════════════════════════════════════════

def extract_connected_components(binary_image: np.ndarray):
    """
    Extracts ink blobs from the page.
    Input: binary_image (Shape: Height x Width, Values: 0 or 255)
    """

    # algo OpenCV in C++ optimized
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8
    )
    #num_labels is the total number of connected components = N
    #labels is a matrix of the same size as the binary image, but with each connected component labeled with a unique integer 1,2,3,...
    #stats: matrix Nx5 with the 5 columns being:[Top-Left X, Top-Left Y, Width, Height, Pixel Area]
    #centroids: matrix Nx2 with the 2 columns being:[Centroid X, Centroid Y]

    # WHY WE SLICE [1:]:
    # Index 0 in 'stats' and 'centroids' represents the white background of the page.
    # If we don't remove it, the algorithm will draw lines from the center of the page
    # to every single character.
    # By doing [1:], we drop row 0 and keep rows 1 to the end.

    filtered_stats = stats[1:]        # Shape becomes (N, 5)
    filtered_centroids = centroids[1:] # Shape becomes (N, 2)

    return filtered_stats, filtered_centroids


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Neighbourhood graph via Delaunay triangulation
# ═══════════════════════════════════════════════════════════════════

def generate_delaunay_edges(centroids: np.ndarray) -> list[tuple]:
    """
    Connects neighboring centroids using triangles.
    Input: centroids array of shape (N, 2) e.g., [[126., 59.], [140., 59.], ...]
    Output: list of tuples: [(0, 1), (1, 2), (5, 12), of all connected components that are neighbours
    """
    tri = Delaunay(centroids) #creates the delaunay triangulation of the centroids
    unique_edges = set() # Using a set automatically prevents duplicates

    # Loop through every triangle (simplex) in the mesh
    for simplex in tri.simplices: #tri.simplices is an array of shape (Num_Triangles, 3), each row is a triangle formed by the indices of the centroids
        # Example: if simplex is [5, 12, 45], meaning centroid[5], centroid[12], and centroid[45] are connected our edges are (5,12), (12,45), (45,5)
        edge1 = (simplex[0], simplex[1])
        edge2 = (simplex[1], simplex[2])
        edge3 = (simplex[2], simplex[0])

        for edge in [edge1, edge2, edge3]:
            # tuple(sorted()) ensures that edge (12, 5) becomes (5, 12).
            # This allows the set() to realize (5, 12) is already in the list and ignore it.
            sorted_edge = tuple(sorted(edge))
            unique_edges.add(sorted_edge)

    # Output is a list of tuples: [(0, 1), (1, 2), (5, 12), of all connected components that are neighbours
    return list(unique_edges)


def _fallback_knn_edges(centroids: np.ndarray) -> list[tuple]:
    """
    Fallback when Delaunay fails (collinear points, < 3 points).
    Uses k-nearest-neighbors to build the neighbourhood graph instead.
    """
    n = len(centroids)
    k = min(5, n - 1)  # at most 5 neighbours, but cannot exceed N-1
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroids)  # +1 because the point itself is included
    _, indices = nbrs.kneighbors(centroids)

    unique_edges = set()
    for i in range(n):
        for j in indices[i]:
            if i != j:
                unique_edges.add(tuple(sorted((i, j))))

    return list(unique_edges)


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — 2D displacement vectors
# ═══════════════════════════════════════════════════════════════════

def compute_displacement_vectors(
    edges: list[tuple],
    centroids: np.ndarray,
    stats: np.ndarray,
    max_distance_ratio: float = 6.0,
) -> tuple[np.ndarray, list[tuple], list[int]]:
    """
    Computes normalized 2D displacement vectors (dx, dy) for each edge.

    WHY 2D INSTEAD OF SCALAR DISTANCE:
    A scalar distance loses directional information. Two gaps of 50 pixels
    are treated identically whether they are horizontal (word spacing) or
    vertical (line spacing). By keeping the full (dx, dy) vector, the
    downstream VBGMM can discover clusters that encode BOTH magnitude and
    direction => e.g. a horizontal Gaussian for word gaps and a vertical
    Gaussian for line gaps.

    WHY NORMALIZATION BY MEDIAN HEIGHT:
    Different documents have different font sizes and scanning DPIs.
    Dividing by the median connected-component height makes all
    displacement vectors scale-invariant: a word gap is always ~1–2
    median heights wide regardless of whether the document was scanned
    at 150 or 300 DPI.

    WHY SYMMETRIZATION:
    Each undirected edge (i, j) has an arbitrary orientation. If we only
    emit (dx, dy), the VBGMM would see an asymmetric cloud and might
    fit two separate Gaussians for the same gap type (one for left-to-right,
    one for right-to-left). By emitting BOTH (dx, dy) and (-dx, -dy),
    the point cloud becomes symmetric around the origin, and each true
    gap type produces a single pair of mirror Gaussians that are easy
    to deduplicate.

    WHY PRUNING LONG EDGES:
    Delaunay triangulation connects ALL points on the convex hull with
    very long edges that span the entire page. These hull artifacts would
    dominate the VBGMM and create spurious clusters. We discard any edge
    longer than max_distance_ratio * median_height.

    Returns:
        vectors_2d    — (M, 2) array of normalized, symmetrized displacement vectors
        pruned_edges  — list of (i, j) tuples that survived pruning
        edge_mapping  — for each row in vectors_2d, the index into pruned_edges
                        (two consecutive rows map to the same edge: forward and backward)
    """
    # Median component height (our scale reference)
    # stats[:, 3] is the height column. Clamp to >= 1 to avoid division by zero.
    median_h = max(float(np.median(stats[:, 3])), 1.0)

    pruned_edges = []
    vectors = []       # will hold (dx, dy) pairs
    edge_mapping = []  # which edge index does each vector row belong to

    for i_cc, j_cc in edges:
        dx = centroids[j_cc][0] - centroids[i_cc][0]
        dy = centroids[j_cc][1] - centroids[i_cc][1]
        dist = np.sqrt(dx * dx + dy * dy)

        # Prune edges that are unreasonably long (Delaunay hull artifacts)
        if dist / median_h > max_distance_ratio:
            continue

        # Normalize by median height for scale invariance
        dx_norm = dx / median_h
        dy_norm = dy / median_h

        edge_idx = len(pruned_edges)
        pruned_edges.append((i_cc, j_cc))

        # Symmetrize: add both the forward vector and its mirror
        vectors.append([dx_norm, dy_norm])
        edge_mapping.append(edge_idx)

        vectors.append([-dx_norm, -dy_norm])
        edge_mapping.append(edge_idx)

    vectors_2d = np.array(vectors) if vectors else np.empty((0, 2))
    return vectors_2d, pruned_edges, edge_mapping


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Variational Bayesian Gaussian Mixture Model
# ═══════════════════════════════════════════════════════════════════

def fit_vbgmm(vectors_2d: np.ndarray) -> BayesianGaussianMixture:
    """
    Fits a Variational Bayesian GMM with K_init = 15 on the 2D vectors.

    WHY VARIATIONAL BAYESIAN (not standard EM-GMM):
    A standard GMM requires the user to specify K (the number of clusters)
    in advance. Choosing K = 4 assumes every document has exactly 4 gap
    types, which is wrong — some documents have 2 (characters + lines),
    others have 5+ (characters, words, lines, columns, paragraphs).
    The Variational Bayesian approach places a Dirichlet Process prior
    over the mixing weights. During inference, the model drives the
    weights of unnecessary components toward zero, effectively "pruning"
    them. We start with K_init = 15 (a generous upper bound) and let the
    data decide how many clusters actually exist.

    WHY weight_concentration_prior = 0.1:
    This is the alpha_0 parameter of the Dirichlet prior. A small value
    (alpha_0 << 1) favors sparse solutions => the model prefers using fewer
    components. A large value (alpha_0 >> 1) would keep all 15 alive.
    We set alpha_0 = 0.1 to encourage aggressive pruning while still
    allowing up to ~6 clusters for complex documents.

    WHY covariance_type = 'full':
    Each 2D Gaussian needs a full 2x2 covariance matrix to capture
    elliptical clusters (e.g., word gaps spread more horizontally than
    vertically). A 'diag' or 'spherical' covariance would force circular
    clusters, losing the directional discrimination that is the whole
    point of the 2D approach.
    """
    model = BayesianGaussianMixture(
        n_components=15,                             # K_init: generous upper bound
        covariance_type="full",                      # full 2x2 covariance per component
        weight_concentration_prior_type="dirichlet_process",  # enables auto-pruning
        weight_concentration_prior=0.1,              # small alpha_0 → sparse solution
        max_iter=300,                                # enough iterations for convergence
        random_state=42,                             # reproducibility
    )
    model.fit(vectors_2d)
    return model


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Cluster classification (word / line / block)
# ═══════════════════════════════════════════════════════════════════

def classify_clusters(
    model: BayesianGaussianMixture,
    weight_threshold: float = 0.01,
) -> dict[int, str]:
    """
    Assigns a semantic label ("word", "line", or "block") to each
    surviving VBGMM component.

    WHY POLAR DECOMPOSITION:
    Each 2D Gaussian has a mean vector (mu_x, mu_y). Converting to polar
    coordinates (radius, angle) gives us two interpretable quantities:
      - radius = ||mu|| = typical gap SIZE for that cluster
      - angle  = atan2(|mu_y|, |mu_x|) = typical gap DIRECTION

    In a standard document:
      - Word gaps are small and nearly horizontal => small radius, angle ≈ 0°
      - Line gaps are medium and nearly vertical => medium radius, angle ≈ 90°
      - Block gaps are large (any direction) => large radius

    WHY DEDUPLICATION OF MIRROR PAIRS:
    Because we symmetrized the input data (Step 3), every true gap type
    produces TWO Gaussians: one at (mu_x, mu_y) and one at (-mu_x, -mu_y).
    We detect and merge these mirror pairs to avoid double-counting.

    CLASSIFICATION RULES (fixed angle thresholds):
      - angle < 30°   => horizontal cluster => label "word"
      - angle > 50°   => vertical cluster   => label "line"
      - remaining     => label "block"
    Among clusters of the same direction, we sort by radius and assign
    the smallest as "word" (or "line"), with larger ones as "block".

    Returns:
        A dictionary mapping each component index to its semantic label ("word", "line", or "block").
    """
    weights = model.weights_
    means = model.means_  # shape (K_init, 2)

    # --- Filter active components (those not pruned by the Dirichlet prior) ---
    active_indices = np.where(weights > weight_threshold)[0]

    if len(active_indices) == 0:
        # Degenerate case: no active component => everything is one block
        return {}

    # --- Deduplicate mirror pairs ---
    # Two components are mirrors if their means are approximately negatives
    # of each other => mu_a ≈ -mu_b (within 20% of their radius).
    used = set()
    unique_representatives = []  # list of component indices (one per mirror pair)

    for idx in active_indices:
        if idx in used:
            continue
        mu = means[idx]
        # Search for a mirror partner among remaining active components
        found_mirror = False
        for jdx in active_indices:
            if jdx <= idx or jdx in used:
                continue
            mu_j = means[jdx]
            # Check if mu_j ≈ -mu (their sum should be near zero)
            sum_norm = np.linalg.norm(mu + mu_j)
            avg_radius = (np.linalg.norm(mu) + np.linalg.norm(mu_j)) / 2.0
            if avg_radius > 0 and sum_norm / avg_radius < 0.3:
                # They are mirrors — keep the one with positive mu_x
                # (or positive mu_y if mu_x ≈ 0)
                used.add(jdx)
                found_mirror = True
                break
        used.add(idx)
        unique_representatives.append(idx)

    # --- Compute polar coordinates for each unique representative ---
    cluster_info = []  # list of (component_index, radius, angle_deg)
    for idx in unique_representatives:
        mu = means[idx]
        radius = np.linalg.norm(mu)
        # Use absolute angle so that left-pointing and right-pointing are treated the same
        angle_deg = np.degrees(np.arctan2(abs(mu[1]), abs(mu[0])))
        cluster_info.append((idx, radius, angle_deg))

    # Sort by radius (ascending) to process small gaps first
    cluster_info.sort(key=lambda x: x[1])

    # --- Assign semantic labels ---
    labels = {}          # component_index → "word" | "line" | "block"
    found_word = False
    found_line = False

    for idx, radius, angle_deg in cluster_info:
        if angle_deg < 30.0 and not found_word:
            # Small, horizontal gap => word spacing
            labels[idx] = "word"
            found_word = True
        elif angle_deg > 50.0 and not found_line:
            # Medium, vertical gap => line spacing
            labels[idx] = "line"
            found_line = True
        else:
            labels[idx] = "block"

    # --- Fallbacks for unusual documents ---
    if not found_word and cluster_info:
        # No horizontal cluster found => assign the smallest-radius cluster as "word"
        smallest_idx = cluster_info[0][0]
        labels[smallest_idx] = "word"
        found_word = True

    if not found_line and len(cluster_info) >= 2:
        # No vertical cluster found => assign the second-smallest as "line"
        for idx, radius, angle_deg in cluster_info:
            if labels.get(idx) != "word":
                labels[idx] = "line"
                found_line = True
                break

    # --- Propagate labels to mirror partners ---
    # Every active component that is a mirror of a classified component
    # gets the same label.
    full_labels = {}
    for idx in active_indices:
        mu = means[idx]
        # Find which unique representative this component belongs to
        best_match = None
        best_dist = float("inf")
        for rep_idx in unique_representatives:
            mu_rep = means[rep_idx]
            # Check direct match
            d_direct = np.linalg.norm(mu - mu_rep)
            # Check mirror match
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
# STEP 6 — Edge labelling via Bayes' decision rule
# ═══════════════════════════════════════════════════════════════════

def label_edges(
    model: BayesianGaussianMixture,
    cluster_labels: dict[int, str],
    vectors_2d: np.ndarray,
    pruned_edges: list[tuple],
    edge_mapping: list[int],
) -> list[tuple[int, int, str]]:
    """
    Labels each edge in the neighbourhood graph with its semantic type.

    HOW IT WORKS:
    The VBGMM's predict() method applies Bayes' decision rule: for each
    2D vector x, it computes the posterior probability of each component k:

        p(k | x) ∝ π_k · N(x | μ_k, Σ_k)

    and assigns x to the component with the highest posterior. This is the
    Maximum A-Posteriori (MAP) assignment.

    We then look up the semantic label of the assigned component to get
    the edge's type ("word", "line", or "block").

    WHY WE USE THE FORWARD VECTOR ONLY:
    Each edge has two rows in vectors_2d (forward and backward due to
    symmetrization). Both should predict mirror components with the same
    semantic label. We use the forward vector (even-indexed rows) for
    classification.
    """
    if len(vectors_2d) == 0:
        return []

    # Predict the VBGMM component for every vector (both forward and backward)
    predictions = model.predict(vectors_2d)

    # For each original edge, read the prediction from its forward vector
    labeled = []
    for edge_idx, (i_cc, j_cc) in enumerate(pruned_edges):
        # The forward vector is at row 2 * edge_idx (even rows)
        row_idx = edge_idx * 2
        predicted_component = predictions[row_idx]

        # Look up the semantic label; default to "block" if component not classified
        label = cluster_labels.get(predicted_component, "block")
        labeled.append((i_cc, j_cc, label))

    return labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Hierarchical merging (union-find)
# ═══════════════════════════════════════════════════════════════════

def _union_bbox(bboxes: list[tuple]) -> tuple:
    """
    Computes the smallest axis-aligned bounding box that encloses all
    input bounding boxes.
    Input:  list of (x, y, w, h)
    Output: single (x, y, w, h)
    """
    xs = [b[0] for b in bboxes]
    ys = [b[1] for b in bboxes]
    x2s = [b[0] + b[2] for b in bboxes]
    y2s = [b[1] + b[3] for b in bboxes]
    x_min, y_min = min(xs), min(ys)
    return (x_min, y_min, max(x2s) - x_min, max(y2s) - y_min)


def _find(parent: list[int], x: int) -> int:
    """Union-find: find root with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving for amortized near-O(1)
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    """Union-find: merge the sets containing a and b."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def hierarchical_merge(
    stats: np.ndarray,
    centroids: np.ndarray,
    labeled_edges: list[tuple[int, int, str]],
    n_components: int,
) -> list[dict]:
    """
    Merges connected components into blocks via three hierarchical passes.

    WHY THREE PASSES (not one):
    Document structure is inherently hierarchical:
        characters => words => lines => blocks
    A single flat clustering would lose this hierarchy. By merging in order
    of gap type (smallest first), we respect the nesting: characters are
    first grouped into words, then words into lines, then lines into blocks.

    PASS 1 — WORD GROUPING:
    Merge all CCs connected by "word" edges (small, horizontal gaps).
    Each resulting group is a word (or part of a word for CJK).

    PASS 2 — LINE GROUPING:
    Two word-groups are neighbours if any of their constituent CCs
    were connected by a "line" edge. Merge neighbouring word-groups
    => each resulting group is a line.

    PASS 3 — BLOCK GROUPING:
    Same logic, using "block" edges to merge line-groups.
    Each resulting group is a text block (paragraph, table, figure...).

    The output matches segmentation.py's contract:
        [{"bbox": (x,y,w,h), "components": [{"bbox": (x,y,w,h)}, ...]}, ...]
    """

    # Build component dicts from stats
    comp_dicts = []
    for i in range(n_components):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        comp_dicts.append({"bbox": (x, y, w, h), "center": (cx, cy)})

    # ── Pass 1: word grouping ──────────────────────────────────────
    parent = list(range(n_components))
    for i_cc, j_cc, label in labeled_edges:
        if label == "word":
            _union(parent, i_cc, j_cc)

    # Collect word-groups
    word_groups_map = defaultdict(list)
    for idx in range(n_components):
        word_groups_map[_find(parent, idx)].append(idx)

    word_items = []
    word_group_id_for_cc = {}  # maps each CC index → its word-group index
    for group_idx, (_, members) in enumerate(word_groups_map.items()):
        bboxes = [comp_dicts[m]["bbox"] for m in members]
        merged_bbox = _union_bbox(bboxes)
        cx = merged_bbox[0] + merged_bbox[2] / 2
        cy = merged_bbox[1] + merged_bbox[3] / 2
        word_items.append({
            "bbox": merged_bbox,
            "center": (cx, cy),
            "cc_members": members,  # track which CCs belong to this word-group
        })
        for m in members:
            word_group_id_for_cc[m] = group_idx

    # ── Pass 2: line grouping ─────────────────────────────────────
    n_word_groups = len(word_items)
    parent = list(range(n_word_groups))
    for i_cc, j_cc, label in labeled_edges:
        if label == "line":
            # Find which word-groups these CCs belong to
            wg_a = word_group_id_for_cc.get(i_cc)
            wg_b = word_group_id_for_cc.get(j_cc)
            if wg_a is not None and wg_b is not None and wg_a != wg_b:
                _union(parent, wg_a, wg_b)

    # Collect line-groups
    line_groups_map = defaultdict(list)
    for idx in range(n_word_groups):
        line_groups_map[_find(parent, idx)].append(idx)

    line_items = []
    line_group_id_for_wg = {}  # maps each word-group index → its line-group index
    for group_idx, (_, wg_members) in enumerate(line_groups_map.items()):
        bboxes = [word_items[wg]["bbox"] for wg in wg_members]
        merged_bbox = _union_bbox(bboxes)
        cx = merged_bbox[0] + merged_bbox[2] / 2
        cy = merged_bbox[1] + merged_bbox[3] / 2
        # Collect all CCs from all word-groups in this line
        all_cc_members = []
        for wg in wg_members:
            all_cc_members.extend(word_items[wg]["cc_members"])
        line_items.append({
            "bbox": merged_bbox,
            "center": (cx, cy),
            "cc_members": all_cc_members,
            "wg_members": wg_members,
        })
        for wg in wg_members:
            line_group_id_for_wg[wg] = group_idx

    # ── Pass 3: block grouping ────────────────────────────────────
    n_line_groups = len(line_items)
    parent = list(range(n_line_groups))
    for i_cc, j_cc, label in labeled_edges:
        if label == "block":
            wg_a = word_group_id_for_cc.get(i_cc)
            wg_b = word_group_id_for_cc.get(j_cc)
            if wg_a is not None and wg_b is not None:
                lg_a = line_group_id_for_wg.get(wg_a)
                lg_b = line_group_id_for_wg.get(wg_b)
                if lg_a is not None and lg_b is not None and lg_a != lg_b:
                    _union(parent, lg_a, lg_b)

    # Collect final blocks
    block_groups_map = defaultdict(list)
    for idx in range(n_line_groups):
        block_groups_map[_find(parent, idx)].append(idx)

    blocks = []
    for _, lg_members in block_groups_map.items():
        bboxes = [line_items[lg]["bbox"] for lg in lg_members]
        merged_bbox = _union_bbox(bboxes)
        # Collect all child CC bboxes for the "components" field
        all_cc_members = []
        for lg in lg_members:
            all_cc_members.extend(line_items[lg]["cc_members"])
        child_components = [{"bbox": comp_dicts[cc]["bbox"]} for cc in all_cc_members]

        blocks.append({
            "bbox": merged_bbox,
            "components": child_components,
        })

    return blocks


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def segment(binary_image: np.ndarray) -> list[dict]:
    """
    Full segmentation pipeline: from binary image to structured blocks.

    Input:  binary_image (H x W, uint8, ink = 0, background = 255)
            This is the output of the preprocessing stage (Sauvola binarization).
    Output: list of block dicts compatible with classification.py:
            [{"bbox": (x, y, w, h), "components": [{"bbox": ...}, ...]}, ...]
    """
    # ── Invert the image ──────────────────────────────────────────
    # The preprocessing stage outputs ink as 0 (black) and background as 255.
    # OpenCV connectedComponentsWithStats treats non-zero pixels as foreground.
    # We invert so that ink pixels become 255 (foreground for OpenCV).
    inverted = cv2.bitwise_not(binary_image)

    # ── Step 1: Extract connected components ──────────────────────
    stats, centroids = extract_connected_components(inverted)

    # Filter out noise: tiny CCs with fewer than 4 pixels are likely
    # scanner artefacts, not real characters.
    areas = stats[:, 4]  # column 4 = pixel area
    valid_mask = areas >= 4
    stats = stats[valid_mask]
    centroids = centroids[valid_mask]
    n = len(stats)

    # ── Edge cases ────────────────────────────────────────────────
    if n == 0:
        return []

    if n == 1:
        x, y, w, h = int(stats[0, 0]), int(stats[0, 1]), int(stats[0, 2]), int(stats[0, 3])
        return [{"bbox": (x, y, w, h), "components": [{"bbox": (x, y, w, h)}]}]

    # ── Step 2: Build neighbourhood graph ─────────────────────────
    if n == 2:
        # Delaunay requires >= 3 non-collinear points; for 2 CCs, build the single edge directly
        edges = [(0, 1)]
    else:
        try:
            edges = generate_delaunay_edges(centroids)
        except QhullError:
            # Collinear or degenerate point set — fall back to k-NN
            edges = _fallback_knn_edges(centroids)

    # ── Step 3: 2D displacement vectors ───────────────────────────
    vectors_2d, pruned_edges, edge_mapping = compute_displacement_vectors(
        edges, centroids, stats
    )

    if len(vectors_2d) < 6:
        # Too few edges to fit a mixture model — return everything as one block
        all_bboxes = [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])) for i in range(n)]
        merged = _union_bbox(all_bboxes)
        return [{"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]}]

    # ── Step 4: Fit VBGMM ────────────────────────────────────────
    model = fit_vbgmm(vectors_2d)

    # ── Step 5: Classify clusters ────────────────────────────────
    cluster_labels = classify_clusters(model)

    if not cluster_labels:
        # No active components — degenerate case
        all_bboxes = [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])) for i in range(n)]
        merged = _union_bbox(all_bboxes)
        return [{"bbox": merged, "components": [{"bbox": b} for b in all_bboxes]}]

    # ── Step 6: Label edges ──────────────────────────────────────
    labeled_edges = label_edges(model, cluster_labels, vectors_2d, pruned_edges, edge_mapping)

    # ── Step 7: Hierarchical merge ───────────────────────────────
    blocks = hierarchical_merge(stats, centroids, labeled_edges, n)

    return blocks





#
# ─── 1. ALGORITHM OVERVIEW ───────────────────────────────────────────────────
#
# This module implements a fully statistical, training-free approach to
# geometric document segmentation. The algorithm takes a binarized page
# image and produces a set of axis-aligned bounding boxes, each enclosing
# a coherent text block (paragraph, table, heading, etc.). No deep
# learning model is used — only classical statistical inference.
#
# The pipeline has seven steps:
#
#   Step 1.  Extract connected components (CCs) via OpenCV's two-pass
#            algorithm (He et al., 2017). Each CC is an ink blob (a
#            character, ligature, diacritical mark, or graphical element).
#
#   Step 2.  Build a neighbourhood graph over CC centroids using the
#            Delaunay triangulation. This yields a planar graph whose
#            edges connect spatially adjacent components without requiring
#            a fixed number of neighbours.
#
#   Step 3.  For each edge (i, j), compute the 2D displacement vector
#            d_{ij} = (Δx, Δy) = centroid_j − centroid_i, normalized by
#            the median CC height h_med for scale invariance.  Symmetrize
#            by including both d_{ij} and −d_{ij}.
#
#   Step 4.  Fit a Variational Bayesian Gaussian Mixture Model (VBGMM)
#            with K_init = 15 components and a Dirichlet Process prior to
#            the symmetrized 2D point cloud.  The variational inference
#            automatically prunes superfluous components, yielding K*
#            active clusters.
#
#   Step 5.  Classify each active cluster into a semantic level — "word",
#            "line", or "block" — by converting its mean vector to polar
#            coordinates (radius, angle) and applying fixed thresholds:
#                angle < 30°   →  horizontal  →  "word"
#                angle > 50°   →  vertical    →  "line"
#                otherwise     →               "block"
#
#   Step 6.  Label every edge using Bayes' MAP decision rule: assign each
#            displacement vector to the VBGMM component with the highest
#            posterior probability, then look up the semantic label.
#
#   Step 7.  Hierarchically merge CCs via three union-find passes:
#            (i) merge CCs along "word" edges → word groups,
#            (ii) merge word-groups along "line" edges → line groups,
#            (iii) merge line-groups along "block" edges → final blocks.
#
#
# ─── 2. RATIONALE ────────────────────────────────────────────────────────────
#
# 2.1  Why 2D displacement vectors instead of scalar distances?
#
#      Scalar distances discard directional information. In a tabular
#      layout, horizontal cell gaps and vertical row gaps often have
#      similar magnitudes but orthogonal directions. A 1D model cannot
#      distinguish them; a 2D model can, because they occupy different
#      regions of the (Δx, Δy) plane.
#
# 2.2  Why VBGMM instead of a fixed-K GMM?
#
#      The number of distinct gap types varies across documents: a
#      single-column letter may have 2 types (intra-word, inter-line),
#      while a multi-column newspaper may have 5+. A fixed K forces an
#      incorrect model on most documents. The VBGMM with a Dirichlet
#      Process prior adapts K to the data.
#
# 2.3  Why Delaunay triangulation?
#
#      Delaunay edges are a superset of the nearest-neighbour graph and
#      the minimum spanning tree. They adapt to local density without a
#      fixed k, and computation is O(N log N). This makes them a robust
#      and efficient neighbourhood structure for irregularly spaced CCs.
#
# 2.4  Why symmetrization?
#
#      Undirected edges have arbitrary orientation. Without symmetrization,
#      a left-to-right word gap and a right-to-left one would be mapped
#      to opposite quadrants of the (Δx, Δy) plane, splitting a single
#      logical cluster into two. Symmetrization ensures each gap type
#      produces a single pair of mirror Gaussians, simplifiable by
#      deduplication.
#
# 2.5  Why hierarchical merging?
#
#      Document layout is inherently nested: characters form words, words
#      form lines, lines form blocks. A flat clustering would lose this
#      hierarchy and could merge a line-gap CC pair across a word boundary.
#      Three sequential union-find passes respect the nesting order.
#
#
# ─── 3. MATHEMATICAL FORMULATION ────────────────────────────────────────────
#
# 3.1  Displacement vectors
#
#      Given N connected components with centroids c_1, ..., c_N ∈ R^2
#      and a set of edges E ⊆ {1,...,N}^2 from the Delaunay triangulation,
#      compute:
#
#          d_{ij} = (c_j - c_i) / h_{med}     for each (i, j) ∈ E
#
#      where h_{med} = median({h_k}_{k=1}^{N}) is the median bounding-box
#      height.  The symmetrized dataset is:
#
#          D = { d_{ij} : (i,j) ∈ E } ∪ { -d_{ij} : (i,j) ∈ E }
#
#
# 3.2  Variational Bayesian Gaussian Mixture Model
#
#      We model D as drawn from a mixture of K Gaussians:
#
#          p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
#
#      with a Dirichlet Process prior on the mixing weights:
#
#          \pi \sim \text{GEM}(\alpha_0)
#
#      where GEM is the stick-breaking construction with concentration
#      parameter \alpha_0 = 0.1. The variational posterior q(\pi, \mu, \Sigma)
#      is computed by maximizing the Evidence Lower Bound (ELBO):
#
#          \mathcal{L}(q) = E_q[\log p(D, \pi, \mu, \Sigma)]
#                         - E_q[\log q(\pi, \mu, \Sigma)]
#
#      During optimization, components with negligible responsibility are
#      pruned: their weight \pi_k → 0. The effective number of components
#      K* = |{k : \pi_k > \tau}| where \tau = 0.01.
#
#      References:
#      - Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
#        Chapter 10.2: Variational Mixture of Gaussians.
#      - Blei, D. M. & Jordan, M. I. (2006). Variational inference for
#        Dirichlet process mixtures. Bayesian Analysis, 1(1), 121–144.
#
#
# 3.3  Cluster classification via polar decomposition
#
#      For each active component k with mean \mu_k = (\mu_x, \mu_y):
#
#          r_k     = \|\mu_k\|_2                         (radius / gap magnitude)
#          \theta_k = \arctan(|\mu_y| / |\mu_x|)         (angle / gap direction)
#
#      Classification rule:
#
#          label_k = \begin{cases}
#              \text{"word"}  & \text{if } \theta_k < 30° \text{ and smallest } r_k \\
#              \text{"line"}  & \text{if } \theta_k > 50° \text{ and next } r_k \\
#              \text{"block"} & \text{otherwise}
#          \end{cases}
#
#
# 3.4  Edge labelling via Bayes' decision rule
#
#      For each edge (i, j) with displacement vector x = d_{ij}:
#
#          k^* = \arg\max_k \; \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
#
#      The semantic label of edge (i, j) is label_{k^*}.
#
#
# 3.5  Hierarchical union-find merging
#
#      Define the label function L: E → {"word", "line", "block"}.
#
#      Pass 1:  E_w = {(i,j) ∈ E : L(i,j) = "word"}.
#               Run union-find on E_w → partition CCs into word-groups W.
#
#      Pass 2:  Construct inter-word-group edges:
#               E_l = {(W_a, W_b) : ∃ (i,j) ∈ E, L(i,j) = "line",
#                       i ∈ W_a, j ∈ W_b, W_a ≠ W_b}.
#               Run union-find on E_l → partition W into line-groups.
#
#      Pass 3:  Analogous construction using "block" edges → final blocks.
#
#      Each group's bounding box is the axis-aligned minimum enclosing
#      rectangle of all its constituents' bounding boxes.
#
#
# ─── 4. REFERENCES ──────────────────────────────────────────────────────────
#
# [1] O'Gorman, L. (1993). The Document Spectrum for Page Layout Analysis.
#     IEEE Transactions on Pattern Analysis and Machine Intelligence,
#     15(11), 1162–1173. doi:10.1109/34.244677
#     — Introduced the Docstrum method: k-NN + angle histograms for
#       document structure analysis. Our approach generalizes Docstrum by
#       replacing the handcrafted angle/distance histograms with a
#       principled 2D Bayesian mixture model.
#
# [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
#     Springer. Chapter 10: Approximate Inference.
#     — Derives the variational Bayesian treatment of Gaussian mixtures,
#       including the automatic relevance determination (pruning) property.
#
# [3] Blei, D. M. & Jordan, M. I. (2006). Variational inference for
#     Dirichlet process mixtures. Bayesian Analysis, 1(1), 121–144.
#     — Provides the theoretical foundation for the Dirichlet process
#       prior used in our VBGMM to auto-select the number of components.
#
# [4] Kise, K., Sato, A., & Iwata, M. (1998). Segmentation of Page
#     Images Using the Area Voronoi Diagram. Computer Vision and Image
#     Understanding, 70(3), 370–382.
#     — Uses Voronoi diagrams for page segmentation; our Delaunay-based
#       neighbourhood graph is the dual of the Voronoi diagram.
#
# [5] He, L., Ren, X., Gao, Q., Zhao, X., Yao, B., & Chao, Y. (2017).
#     The connected-component labeling problem: A review of state-of-the-art
#     algorithms. Pattern Recognition, 70, 25–43.
#     — Survey of connected component labelling algorithms; OpenCV uses an
#       optimized two-pass variant.
#
# [6] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in
#     Python. Journal of Machine Learning Research, 12, 2825–2830.
#     — Implementation of BayesianGaussianMixture used in this module.
#
# ═══════════════════════════════════════════════════════════════════════════════
