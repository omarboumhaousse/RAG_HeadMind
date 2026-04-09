"""Step 3b: Semantic classification — label each block by type.

OmniDocBench-compatible taxonomy.

Key fixes vs previous version:
    1. Micro-block merging (pre-classification pass):
       When segmentation over-segments (lines instead of paragraphs),
       we merge vertically adjacent small blocks before classifying.
       Threshold: blocks with rh < 0.025 are candidates for merging.

    2. Lateral table caption detection:
       In two-column layouts (table body rw≈0.84 + caption column rw≈0.15),
       the caption column sits BESIDE the table, not above/below.
       We detect this via x-overlap with a table block.

    3. Calibrated table detection:
       Wide + many lines + many words = table (observed: rw=0.84, lines=14)

    4. Better title detection without relying on font_ratio alone.

Labels:
    'title', 'text_block', 'figure', 'figure_caption',
    'table', 'table_caption', 'table_footnote',
    'equation_isolated', 'header', 'footer', 'page_number', 'abandon'
"""

import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def classify(blocks: list[dict], image_shape: tuple) -> list[dict]:
    """Assign an OmniDocBench label to each block.

    Args:
        blocks:      list of dicts from segment() with 'bbox', 'components',
                     'lines' keys.
        image_shape: (H, W) or (H, W, C).

    Returns:
        Same list, each block enriched with a 'label' key.
    """
    if not blocks:
        return blocks

    page_h, page_w = image_shape[:2]

    # ── Pass 0: merge micro-blocks ──────────────────────────────────
    # When segmentation over-segments (many tiny line-level blocks),
    # merge vertically adjacent ones before classifying.
    blocks = _merge_micro_blocks(blocks, page_h, page_w)

    # ── Feature extraction ──────────────────────────────────────────
    features = [_extract_features(b, page_h, page_w) for b in blocks]

    # Reference font from real blocks
    real_heights = []
    for feat in features:
        if not feat["is_synthetic"]:
            real_heights.extend(feat["comp_heights_raw"])
    valid = [h for h in real_heights if 2 < h < page_h * 0.12]
    median_font = float(np.median(valid)) if valid else 10.0

    # ── Pass 1: classify all blocks ─────────────────────────────────
    for block, feat in zip(blocks, features):
        block["label"] = _classify_block(feat, median_font, page_h, page_w)

    # ── Pass 2: refine captions ─────────────────────────────────────
    _refine_captions(blocks, features, page_h, page_w)

    return blocks


# ═══════════════════════════════════════════════════════════════════
# PASS 0: MICRO-BLOCK MERGING
# ═══════════════════════════════════════════════════════════════════

def _merge_micro_blocks(blocks: list[dict],
                        page_h: int, page_w: int) -> list[dict]:
    """Merge vertically adjacent micro-blocks into paragraph-sized blocks.

    A micro-block is one whose height < 2.5% of page height AND whose
    width is consistent with its neighbours (same column / same x-range).

    Strategy:
        1. Sort blocks by y-position.
        2. For consecutive blocks, if both are small (rh < THRESH) and
           their x-ranges overlap significantly → merge.
        3. Repeat until no more merges possible (convergent).

    This corrects the over-segmentation caused by VBGMM convergence
    failure on large / complex pages.
    """
    MICRO_RH  = 0.030   # blocks shorter than 3% of page height
    V_GAP_MAX = 0.015   # max vertical gap between consecutive micro-blocks (1.5% page)
    X_OVERLAP = 0.50    # min x-overlap fraction to consider same column

    def x_overlap_frac(b1, b2):
        x1, _, w1, _ = b1["bbox"]
        x2, _, w2, _ = b2["bbox"]
        lo = max(x1, x2)
        hi = min(x1 + w1, x2 + w2)
        overlap = max(0, hi - lo)
        shorter_w = min(w1, w2)
        return overlap / shorter_w if shorter_w > 0 else 0.0

    def merge_two(b1, b2):
        """Merge two blocks into one."""
        x1, y1, w1, h1 = b1["bbox"]
        x2, y2, w2, h2 = b2["bbox"]
        nx = min(x1, x2)
        ny = min(y1, y2)
        nw = max(x1 + w1, x2 + w2) - nx
        nh = max(y1 + h1, y2 + h2) - ny
        merged = {
            "bbox":       (nx, ny, nw, nh),
            "components": b1.get("components", []) + b2.get("components", []),
            "lines":      b1.get("lines", []) + b2.get("lines", []),
        }
        return merged

    changed = True
    while changed:
        changed = False
        # Sort by top-y then left-x
        blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        new_blocks = []
        used = [False] * len(blocks)

        for i in range(len(blocks)):
            if used[i]:
                continue
            current = blocks[i]
            cx, cy, cw, ch = current["bbox"]
            rh_i = ch / page_h

            if rh_i >= MICRO_RH:
                new_blocks.append(current)
                continue

            # Try to merge with the next eligible block
            merged = False
            for j in range(i + 1, len(blocks)):
                if used[j]:
                    continue
                nx, ny, nw, nh = blocks[j]["bbox"]
                rh_j = nh / page_h

                # Vertical gap
                v_gap = (ny - (cy + ch)) / page_h
                if v_gap > V_GAP_MAX:
                    break   # sorted by y → no point looking further

                # Must also be micro or similar width
                if rh_j >= MICRO_RH * 2:
                    continue

                # X-overlap check (same column)
                if x_overlap_frac(current, blocks[j]) >= X_OVERLAP:
                    current = merge_two(current, blocks[j])
                    cx, cy, cw, ch = current["bbox"]
                    used[j] = True
                    merged = True
                    changed = True

            new_blocks.append(current)

        blocks = new_blocks

    return blocks


# ═══════════════════════════════════════════════════════════════════
# SYNTHETIC DETECTION
# ═══════════════════════════════════════════════════════════════════

def _is_synthetic(components: list[dict], bbox: tuple) -> bool:
    if len(components) < 4:
        return False
    x, y, w, h = bbox
    bbox_area = max(w * h, 1)
    comp_area = sum(c["bbox"][2] * c["bbox"][3] for c in components)
    if comp_area / bbox_area < 0.85:
        return False
    ws = np.array([c["bbox"][2] for c in components], dtype=float)
    hs = np.array([c["bbox"][3] for c in components], dtype=float)
    cv_w = np.std(ws) / (np.mean(ws) + 1e-6)
    cv_h = np.std(hs) / (np.mean(hs) + 1e-6)
    return cv_w < 0.08 and cv_h < 0.08


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _extract_features(block: dict, page_h: int, page_w: int) -> dict:
    x, y, w, h  = block["bbox"]
    components   = block.get("components", [])
    is_synth     = _is_synthetic(components, block["bbox"])

    lines   = block.get("lines", [])
    n_lines = len(lines)
    n_words = sum(len(line.get("words", [])) for line in lines)
    wpl     = [len(line.get("words", [])) for line in lines]
    avg_wpl = float(np.mean(wpl)) if wpl else 0.0
    std_wpl = float(np.std(wpl))  if wpl else 0.0

    comp_heights = [c["bbox"][3] for c in components if c["bbox"][3] > 0]

    if is_synth:
        median_comp_h    = 0.0
        density          = -1.0
        h_align          = -1.0
        grid_score       = -1.0
        comp_heights_raw = []
    else:
        median_comp_h    = float(np.median(comp_heights)) if comp_heights else 0.0
        raw_density      = sum(c["bbox"][2] * c["bbox"][3]
                               for c in components) / max(w * h, 1)
        density          = min(raw_density, 1.5)
        h_align          = _h_align(components, median_comp_h)
        grid_score       = _grid(components) if len(components) >= 6 else 0.0
        comp_heights_raw = comp_heights

    rel_y_top    = y / page_h
    rel_y_bot    = (y + h) / page_h
    rel_x_center = (x + w / 2) / page_w
    rel_width    = w / page_w
    rel_height   = h / page_h
    centering    = 1.0 - 2.0 * abs(rel_x_center - 0.5)

    return {
        "is_synthetic":      is_synth,
        "comp_heights_raw":  comp_heights_raw,
        "n_components":      len(components),
        "n_lines":           n_lines,
        "n_words":           n_words,
        "avg_wpl":           avg_wpl,
        "std_wpl":           std_wpl,
        "median_comp_h":     median_comp_h,
        "density":           density,
        "h_align":           h_align,
        "grid_score":        grid_score,
        "rel_y_top":         rel_y_top,
        "rel_y_bot":         rel_y_bot,
        "rel_x_center":      rel_x_center,
        "rel_width":         rel_width,
        "rel_height":        rel_height,
        "centering":         centering,
        "bbox_x":            x,
        "bbox_y":            y,
        "bbox_w":            w,
        "bbox_h":            h,
    }


def _h_align(components, median_comp_h):
    if len(components) < 2:
        return 0.5
    thresh   = max(median_comp_h * 0.7, 3.0)
    y_cen    = np.array([c["bbox"][1] + c["bbox"][3] / 2 for c in components])
    sorted_y = np.sort(y_cen)
    lines, cur = [], [sorted_y[0]]
    for yc in sorted_y[1:]:
        if yc - cur[-1] <= thresh:
            cur.append(yc)
        else:
            lines.append(cur); cur = [yc]
    lines.append(cur)
    in_row = sum(len(l) for l in lines if len(l) >= 2)
    return in_row / len(components)


def _grid(components):
    if len(components) < 4:
        return 0.0
    xs = np.array([c["bbox"][0] + c["bbox"][2] / 2 for c in components])
    ys = np.array([c["bbox"][1] + c["bbox"][3] / 2 for c in components])

    def reg(coords):
        s = np.sort(coords)
        d = np.diff(s)
        d = d[d > 1]
        if len(d) == 0:
            return 0.0
        return float(np.clip(1 - np.std(d) / (np.mean(d) + 1e-6), 0, 1))

    return (reg(xs) + reg(ys)) / 2.0


# ═══════════════════════════════════════════════════════════════════
# DECISION TREE
# ═══════════════════════════════════════════════════════════════════

def _classify_block(feat, median_font, page_h, page_w):
    mh         = feat["median_comp_h"]
    font_ratio = mh / median_font if median_font > 0 and mh > 0 else 1.0

    # ── Page number ──────────────────────────────────────────────────
    # Observed: scihub B02 (ry=0.048, rh=0.013, rw=0.061, words=2)
    if (
        feat["n_words"] <= 4
        and feat["rel_height"] < 0.025
        and (feat["rel_y_top"] < 0.06 or feat["rel_y_bot"] > 0.94)
        and feat["rel_width"] < 0.35
    ):
        return "page_number"

    # ── Header ───────────────────────────────────────────────────────
    # Sufficient condition: bottom of page + thin strip
    if feat["rel_y_top"] < 0.10 and feat["rel_height"] < 0.05:
        return "header"

    # ── Footer ───────────────────────────────────────────────────────
    # Sufficient condition: bottom of page + thin strip
    if feat["rel_y_bot"] > 0.92 and feat["rel_height"] < 0.05:
        return "footer"

    # ── Synthetic path ───────────────────────────────────────────────
    if feat["is_synthetic"]:
        return _classify_synthetic(feat)

    return _classify_real(feat, font_ratio, page_h, page_w)


def _classify_synthetic(feat):
    rh = feat["rel_height"]
    rw = feat["rel_width"]
    ry = feat["rel_y_top"]
    if rh > 0.15 and rw > 0.20:
        return "figure"
    if rh < 0.10 and rw > 0.25 and ry < 0.55:
        return "title"
    if rh < 0.06 and rw < 0.55 and feat["centering"] > 0.5:
        return "equation_isolated"
    return "text_block"


def _classify_real(feat, font_ratio, page_h, page_w):

    # ── MÉGA-BLOC MAL SEGMENTÉ ───────────────────────────────────────
    # VBGMM failure → giant block, lines=0, many components, high density.
    # Observed: eastmoney B02 (lines=0, comps=1315, density=0.605)
    # NOT code: code also has lines=0, comps>80 but different density profile
    if feat["n_lines"] == 0 and feat["n_components"] > 80 and feat["density"] > 0.25:
        return "text_block"

    # ── CODE ─────────────────────────────────────────────────────────
    # Code blocks: lines=0 (monospace chars confuse VBGMM line detection),
    # many components, moderate-high density, wide block.
    # Observed RxJS code blocks: lines=0, comps=109-256, density=0.17-0.47,
    # rw=0.53-0.67, ry=0.084-0.884 (not at page edges).
    # Distinct from figures: code has density > 0.15 (lots of ink).
    # Distinct from mega-blocs: code has rh < 0.20 per block.
    if (
        feat["n_lines"] == 0
        and feat["n_components"] > 80
        and 0.15 < feat["density"] < 0.50
        and feat["rel_width"] > 0.40
        and feat["rel_height"] < 0.20
        and feat["h_align"] > 0.80   # code is well-aligned horizontally
        and feat["rel_y_top"] > 0.06  # not a header
        and feat["rel_y_bot"] < 0.94  # not a footer
    ):
        return "code_txt"

    # ── TABLE ────────────────────────────────────────────────────────
    # Primary: wide + many lines + many words
    if (
        feat["rel_width"] > 0.55
        and feat["n_lines"] >= 5
        and feat["n_words"] >= 25
        and feat["density"] > 0.09
        and feat["rel_height"] > 0.07
    ):
        return "table"

    # Secondary: grid structure
    if (
        feat["grid_score"] > 0.35
        and feat["rel_width"] > 0.28
        and feat["n_lines"] >= 3
        and feat["density"] > 0.06
        and feat["rel_height"] > 0.04
    ):
        return "table"

    # ── FIGURE ───────────────────────────────────────────────────────
    # Figures: photos, charts, drawings, curves.
    # Key: lines=0 AND few components (real graphical element)
    #      OR very low density (sparse ink = image region)
    # NOT code (already handled above with density > 0.15)

    # lines=0 with few components → real figure/image
    if feat["n_lines"] == 0 and feat["rel_height"] > 0.05:
        if feat["n_components"] <= 80:
            if feat["rel_height"] > 0.06 or feat["rel_width"] > 0.20:
                return "figure"

    # Very low density → figure (sparse image region)
    # Threshold 0.05 to avoid capturing sparse text blocks (density ≈ 0.08-0.15)
    if (
        feat["density"] < 0.05
        and feat["rel_height"] > 0.06
        and feat["rel_width"] > 0.12
    ):
        return "figure"

    # Low density + not text-aligned → figure with embedded labels
    if (
        feat["density"] < 0.10
        and feat["h_align"] < 0.35
        and feat["rel_height"] > 0.10
        and feat["rel_width"] > 0.15
    ):
        return "figure"

    # ── EQUATION (isolated) ──────────────────────────────────────────
    if feat["n_lines"] == 0 and feat["rel_height"] <= 0.06:
        if (
            feat["centering"] > 0.25
            and feat["rel_width"] < 0.80
            and feat["n_components"] <= 15
        ):
            return "equation_isolated"

    if (
        feat["n_lines"] == 1
        and feat["n_words"] <= 3
        and feat["rel_height"] < 0.04
        and feat["centering"] > 0.35
        and feat["rel_width"] < 0.60
        and feat["n_components"] <= 20
    ):
        return "equation_isolated"

    # ── CAPTION candidate (refined in pass 2) ────────────────────────
    if (
        feat["n_lines"] <= 2
        and feat["rel_height"] < 0.040
        and feat["n_words"] >= 3
        and font_ratio <= 1.10
        and feat["rel_y_top"] > 0.08
    ):
        return "_caption_candidate"

    # ── TITLE ────────────────────────────────────────────────────────
    # Strong font signal
    if (
        font_ratio > 1.50
        and feat["n_lines"] <= 4
        and feat["rel_height"] < 0.20
        and feat["rel_width"] > 0.10
    ):
        return "title"

    # Moderate font, short block
    if (
        font_ratio > 1.20
        and feat["n_lines"] <= 4
        and feat["rel_height"] < 0.12
        and 0.10 < feat["rel_width"] < 0.95
        and feat["density"] > 0.04
    ):
        return "title"

    # Position-based: top third, short
    if (
        feat["rel_y_top"] < 0.35
        and feat["n_lines"] <= 3
        and feat["rel_height"] < 0.08
        and feat["rel_width"] > 0.15
        and feat["density"] > 0.05
        and feat["rel_y_top"] > 0.08
    ):
        return "title"

    # Single line, few words, section headings
    if (
        feat["n_lines"] == 1
        and feat["n_words"] <= 10
        and feat["rel_height"] < 0.05
        and feat["rel_width"] > 0.20
        and font_ratio > 1.10
    ):
        return "title"

    # ── TEXT BLOCK (default) ─────────────────────────────────────────
    return "text_block"


# ═══════════════════════════════════════════════════════════════════
# PASS 2: CAPTION / FOOTNOTE REFINEMENT
# ═══════════════════════════════════════════════════════════════════

def _refine_captions(blocks, features, page_h, page_w):
    """Resolve _caption_candidate labels using spatial proximity.

    Two cases:
    A) Vertical proximity: caption directly above/below a table or figure.
    B) Lateral proximity: caption column beside a table (two-column layout).
       Observed: table rw≈0.84 centered, caption column rw≈0.15 on the left.
       Both blocks have overlapping y-ranges.
    """
    v_gap = page_h * 0.06
    h_gap = page_w * 0.05   # horizontal gap for lateral detection

    table_info  = []  # (y0, y1, x0, x1, idx)
    figure_info = []

    for i, (b, f) in enumerate(zip(blocks, features)):
        lbl = b.get("label", "")
        y0  = f["bbox_y"];           y1 = f["bbox_y"] + f["bbox_h"]
        x0  = f["bbox_x"];           x1 = f["bbox_x"] + f["bbox_w"]
        if lbl == "table":
            table_info.append((y0, y1, x0, x1, i))
        elif lbl == "figure":
            figure_info.append((y0, y1, x0, x1, i))

    for block, feat in zip(blocks, features):
        if block.get("label") != "_caption_candidate":
            continue

        by0 = feat["bbox_y"];  by1 = feat["bbox_y"] + feat["bbox_h"]
        bx0 = feat["bbox_x"];  bx1 = feat["bbox_x"] + feat["bbox_w"]
        done = False

        # ── Check tables ─────────────────────────────────────────────
        for t0, t1, tx0, tx1, _ in table_info:
            # Vertical: caption above table
            if 0 <= t0 - by1 <= v_gap:
                block["label"] = "table_caption"; done = True; break
            # Vertical: caption below table (footnote)
            if 0 <= by0 - t1 <= v_gap:
                block["label"] = "table_footnote"; done = True; break
            # Lateral: caption beside table (overlapping y-range)
            y_overlap = min(by1, t1) - max(by0, t0)
            x_gap     = max(0, max(tx0, bx0) - min(tx1, bx1))
            if y_overlap > 0 and x_gap <= h_gap:
                block["label"] = "table_caption"; done = True; break
        if done:
            continue

        # ── Check figures ─────────────────────────────────────────────
        for f0, f1, fx0, fx1, _ in figure_info:
            if 0 <= f0 - by1 <= v_gap or 0 <= by0 - f1 <= v_gap:
                block["label"] = "figure_caption"; done = True; break
            y_overlap = min(by1, f1) - max(by0, f0)
            x_gap     = max(0, max(fx0, bx0) - min(fx1, bx1))
            if y_overlap > 0 and x_gap <= h_gap:
                block["label"] = "figure_caption"; done = True; break
        if done:
            continue

        block["label"] = "text_block"