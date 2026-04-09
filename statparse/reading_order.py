"""Step 4: Reading order — sort blocks into the logical reading sequence.

Input:  list of classified block dicts (output of classify()), each with:
            'bbox':  (x, y, w, h)
            'label': str
            'lines': [{'bbox':..., 'words':[...]}]

Output: same list, reordered and enriched with an 'order' key (1-indexed).

Algorithm
---------
Document reading order is not simply top-to-bottom, left-to-right.
Multi-column layouts, side captions, and floating elements require
a column-aware approach.

Pipeline:
    1. Separate structural elements (header, footer, page_number) from
       content blocks. Headers always come first, footers last.

    2. Detect column layout via X-projection gap analysis:
       - Project all content blocks onto the X-axis.
       - Find large whitespace gaps (> 5% page width with no block overlap).
       - Each gap defines a column boundary.

    3. Assign each block to its column based on its horizontal center.

    4. Within each column, sort top-to-bottom by block top-y.

    5. Read columns left-to-right.

    6. Handle special cases:
       - Full-width blocks (spanning > 80% of page width) interrupt
         the column flow and are placed at their natural vertical position
         relative to the surrounding column content.
       - Captions are placed immediately after their associated
         figure/table when possible.

References
----------
Meunier (2005): "Efficient Text Extraction from PDF Documents"
Aiello et al. (2002): "Document Understanding for a Broad Class of Documents"
"""

import numpy as np
from collections import defaultdict


# ── Labels that always go first / last ────────────────────────────────
FIRST_LABELS = {"header"}
LAST_LABELS  = {"footer", "page_number", "page_footnote"}

# ── Labels considered "full-width" anchors ────────────────────────────
FULL_WIDTH_LABELS = {"title", "table", "figure", "equation_isolated"}


def order_blocks(blocks: list[dict], image_shape: tuple = None) -> list[dict]:
    """Sort blocks into reading order and add an 'order' key.

    Args:
        blocks:      classified blocks from classify()
        image_shape: (H, W) or (H, W, C) — inferred from block bboxes if None

    Returns:
        Same blocks reordered, each with 'order' key (1-indexed integer).
    """
    if not blocks:
        return blocks

    # Infer page dimensions from block bboxes if not provided
    if image_shape is not None:
        page_h, page_w = image_shape[:2]
    else:
        all_x2 = [b["bbox"][0] + b["bbox"][2] for b in blocks]
        all_y2 = [b["bbox"][1] + b["bbox"][3] for b in blocks]
        page_w  = max(all_x2) if all_x2 else 1000
        page_h  = max(all_y2) if all_y2 else 1000

    # ── Step 1: Separate fixed-position elements ───────────────────────
    first_blocks   = [b for b in blocks if b.get("label") in FIRST_LABELS]
    last_blocks    = [b for b in blocks if b.get("label") in LAST_LABELS]
    content_blocks = [b for b in blocks
                      if b.get("label") not in FIRST_LABELS
                      and b.get("label") not in LAST_LABELS]

    # Sort first/last by vertical position
    first_blocks.sort(key=lambda b: b["bbox"][1])
    last_blocks.sort(key=lambda b: b["bbox"][1])

    if not content_blocks:
        ordered = first_blocks + last_blocks
        for i, b in enumerate(ordered):
            b["order"] = i + 1
        return ordered

    # ── Step 2: Detect column layout ──────────────────────────────────
    column_assignments = _assign_columns(content_blocks, page_w)

    # ── Step 3: Sort within columns ───────────────────────────────────
    n_cols = max(column_assignments.values()) + 1 if column_assignments else 1

    # Group blocks by column
    col_groups = defaultdict(list)
    for block, col_idx in zip(content_blocks, column_assignments.values()):
        col_groups[col_idx].append(block)

    # Sort each column top-to-bottom
    for col_idx in col_groups:
        col_groups[col_idx].sort(key=lambda b: b["bbox"][1])

    # ── Step 4: Interleave full-width blocks ──────────────────────────
    # Full-width blocks interrupt columns at their vertical position.
    ordered_content = _interleave_fullwidth(col_groups, n_cols, page_w)

    # ── Step 5: Assemble final order ──────────────────────────────────
    ordered = first_blocks + ordered_content + last_blocks

    for i, b in enumerate(ordered):
        b["order"] = i + 1

    return ordered


# ═══════════════════════════════════════════════════════════════════
# COLUMN DETECTION
# ═══════════════════════════════════════════════════════════════════

def _assign_columns(blocks: list[dict], page_w: int) -> dict:
    """Assign each block to a column index (0-based, left to right).

    Method: find X-axis gaps where no block projects ink, then use
    those gaps as column boundaries.

    Returns a dict mapping block index → column index.
    """
    if not blocks:
        return {}

    # Build X-projection: for each pixel column, is there any block?
    # Use a coarser grid (1% of page width per cell) for efficiency.
    grid_w = max(page_w // 100, 1)
    n_cells = page_w // grid_w + 1
    coverage = np.zeros(n_cells, dtype=np.int32)

    for b in blocks:
        x, y, w, h = b["bbox"]
        x1 = x // grid_w
        x2 = (x + w) // grid_w
        coverage[x1:x2 + 1] += 1

    # Find runs of zero coverage → candidate column gaps
    MIN_GAP_CELLS = max(3, int(0.04 * page_w / grid_w))  # ≥ 4% page width

    gaps = []
    in_gap, gap_start = False, 0
    for i in range(n_cells):
        if coverage[i] == 0 and not in_gap:
            gap_start = i
            in_gap = True
        elif coverage[i] > 0 and in_gap:
            gap_w = i - gap_start
            if gap_w >= MIN_GAP_CELLS:
                gaps.append((gap_start * grid_w, i * grid_w))
            in_gap = False
    if in_gap:
        gap_w = n_cells - gap_start
        if gap_w >= MIN_GAP_CELLS:
            gaps.append((gap_start * grid_w, n_cells * grid_w))

    # Column boundaries from gaps
    col_boundaries = [0]
    for gap_start_px, gap_end_px in gaps:
        col_boundaries.append((gap_start_px + gap_end_px) // 2)
    col_boundaries.append(page_w)

    # Assign each block to the column whose center it falls in
    assignments = {}
    for i, b in enumerate(blocks):
        x, y, w, h = b["bbox"]
        cx = x + w / 2
        col_idx = 0
        for j in range(len(col_boundaries) - 1):
            if col_boundaries[j] <= cx < col_boundaries[j + 1]:
                col_idx = j
                break
        assignments[i] = col_idx

    return assignments


# ═══════════════════════════════════════════════════════════════════
# FULL-WIDTH BLOCK INTERLEAVING
# ═══════════════════════════════════════════════════════════════════

def _interleave_fullwidth(col_groups: dict,
                          n_cols: int,
                          page_w: int) -> list[dict]:
    """Merge column streams, inserting full-width blocks at their y-position.

    Full-width blocks (spanning > 80% of page width) interrupt the normal
    left-to-right column reading flow at their vertical position.

    Algorithm:
        - Separate full-width blocks from column-specific blocks.
        - Use a merge-sort-like approach: at each step, pick the next block
          from any column or full-width queue based on top-y.
    """
    # Identify full-width blocks (already sorted within their column)
    full_width  = []
    col_content = defaultdict(list)

    for col_idx, col_blocks in col_groups.items():
        for b in col_blocks:
            x, y, w, h = b["bbox"]
            # Full-width: block covers > 75% of page width
            if w / page_w > 0.75:
                full_width.append(b)
            else:
                col_content[col_idx].append(b)

    # Sort full-width blocks by top-y
    full_width.sort(key=lambda b: b["bbox"][1])

    # Column pointers
    col_pointers = {col: 0 for col in range(n_cols)}

    result = []
    fw_ptr = 0

    # Read columns left-to-right, inserting full-width blocks when their
    # y-position is reached
    while True:
        # Find next block across all columns (minimum top-y)
        candidates = []
        for col in range(n_cols):
            ptr = col_pointers.get(col, 0)
            col_list = col_content.get(col, [])
            if ptr < len(col_list):
                candidates.append((col_list[ptr]["bbox"][1], col, ptr))

        if not candidates and fw_ptr >= len(full_width):
            break  # all done

        # Insert any full-width block that comes before all column candidates
        next_col_y = min(c[0] for c in candidates) if candidates else float("inf")
        while fw_ptr < len(full_width):
            fw_y = full_width[fw_ptr]["bbox"][1]
            if fw_y <= next_col_y:
                result.append(full_width[fw_ptr])
                fw_ptr += 1
            else:
                break

        if not candidates:
            # Flush remaining full-width blocks
            while fw_ptr < len(full_width):
                result.append(full_width[fw_ptr])
                fw_ptr += 1
            break

        # Read one block from each column in left-to-right order,
        # up to the next full-width block's y-position
        next_fw_y = full_width[fw_ptr]["bbox"][1] if fw_ptr < len(full_width) else float("inf")

        for col in range(n_cols):
            ptr = col_pointers.get(col, 0)
            col_list = col_content.get(col, [])
            while ptr < len(col_list):
                b = col_list[ptr]
                if b["bbox"][1] < next_fw_y:
                    result.append(b)
                    ptr += 1
                else:
                    break
            col_pointers[col] = ptr

        # Check if we made progress (avoid infinite loop)
        new_candidates = []
        for col in range(n_cols):
            ptr = col_pointers.get(col, 0)
            col_list = col_content.get(col, [])
            if ptr < len(col_list):
                new_candidates.append(col_list[ptr])
        if len(new_candidates) == len(candidates) and fw_ptr == fw_ptr:
            # No progress: flush everything remaining
            for col in range(n_cols):
                ptr = col_pointers.get(col, 0)
                col_list = col_content.get(col, [])
                result.extend(col_list[ptr:])
                col_pointers[col] = len(col_list)
            while fw_ptr < len(full_width):
                result.append(full_width[fw_ptr])
                fw_ptr += 1
            break

    return result