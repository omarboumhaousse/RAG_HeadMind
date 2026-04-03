"""
Detailed segmentation test — visualises each intermediate step of the
segmentation pipeline on up to 5 random PDFs from pdfs/.

For each PDF, generates:
    1. step0_connected_components.png  — raw CCs (bounding boxes)
    2. step1_subchar_merge.png         — CCs after sub-character merging
    3. step2_xy_cut_regions.png        — XY-cut pre-segmentation regions
    4. step3_delaunay_graph.png        — Delaunay triangulation edges
    5. step4_vbgmm_clusters.png        — displacement vectors coloured by cluster
    6. step5_edge_labels.png           — edges coloured by semantic label
    7. step6_hierarchy.png             — final blocks → lines → words
"""

import sys
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess
from statparse.segmentation import (
    _extract_connected_components,
    _merge_subcharacter_components,
    _xy_cut_presegment,
    _delaunay_edges,
    _fallback_knn_edges,
    _compute_displacement_vectors,
    _fit_vbgmm,
    _classify_clusters,
    _label_edges,
    _hierarchical_merge,
    _union_bbox,
    segment,
)
from scipy.spatial import QhullError

# ── Configuration ─────────────────────────────────────────────────
PDF_DIR = Path("./pdfs")
OUTPUT_DIR = Path("./segmentation_detailed_output")
DPI = 150
N_SAMPLES = 5

# Colours
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
ORANGE = (200, 100, 0)


def pdf_to_rgb(pdf_path, dpi):
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    except ImportError:
        from pdf2image import convert_from_path
        return np.array(convert_from_path(str(pdf_path), dpi=dpi)[0])


def save_img(rgb, path):
    Image.fromarray(rgb).save(path)


# ═══════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def viz_step0_connected_components(page_rgb, binary):
    """Draw all raw connected components as orange bounding boxes."""
    inverted = cv2.bitwise_not(binary)
    stats, centroids = _extract_connected_components(inverted)
    mask = stats[:, 4] >= 4
    stats, centroids = stats[mask], centroids[mask]

    vis = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    for i in range(len(stats)):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        cv2.rectangle(vis, (x, y), (x + w, y + h), ORANGE, 1)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    n = len(stats)
    # Add text overlay with count
    cv2.putText(vis, f"{n} connected components", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"{n} connected components", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)
    return vis, stats, centroids


def viz_step1_subchar_merge(page_rgb, stats, centroids):
    """Draw CCs after sub-character merging."""
    merged_stats, merged_centroids = _merge_subcharacter_components(stats, centroids)

    vis = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    for i in range(len(merged_stats)):
        x, y, w, h = int(merged_stats[i, 0]), int(merged_stats[i, 1]), \
                      int(merged_stats[i, 2]), int(merged_stats[i, 3])
        cv2.rectangle(vis, (x, y), (x + w, y + h), ORANGE, 1)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    n_before, n_after = len(stats), len(merged_stats)
    label = f"{n_after} components (merged from {n_before})"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)
    return vis, merged_stats, merged_centroids


def viz_step2_xy_cut(page_rgb, binary):
    """Draw XY-cut region boundaries."""
    regions = _xy_cut_presegment(binary)

    vis = page_rgb.copy()
    colours = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (200, 0, 200),
               (0, 200, 200), (200, 200, 0), (128, 0, 255), (255, 128, 0)]
    for i, (rx, ry, rw, rh) in enumerate(regions):
        c = colours[i % len(colours)]
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), c, 2)
        cv2.putText(vis, f"R{i+1}", (rx + 5, ry + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)

    label = f"{len(regions)} XY-cut regions"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    return vis, regions


def viz_step3_delaunay(page_rgb, stats, centroids):
    """Draw the Delaunay triangulation edges over the page."""
    n = len(stats)
    if n < 2:
        return page_rgb.copy(), []
    if n == 2:
        edges = [(0, 1)]
    else:
        try:
            edges = _delaunay_edges(centroids)
        except QhullError:
            edges = _fallback_knn_edges(centroids)

    vis = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    for i, j in edges:
        pt1 = (int(centroids[i, 0]), int(centroids[i, 1]))
        pt2 = (int(centroids[j, 0]), int(centroids[j, 1]))
        cv2.line(vis, pt1, pt2, (180, 180, 180), 1, cv2.LINE_AA)
    for i in range(n):
        cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    label = f"{len(edges)} Delaunay edges, {n} nodes"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2, cv2.LINE_AA)
    return vis, edges


def viz_step4_vbgmm_clusters(vectors_2d, model):
    """Scatter plot of 2D displacement vectors coloured by VBGMM cluster."""
    if len(vectors_2d) == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("No displacement vectors")
        fig.savefig("/tmp/_tmp_vbgmm.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return np.array(Image.open("/tmp/_tmp_vbgmm.png").convert("RGB"))

    labels = model.predict(vectors_2d)
    weights = model.weights_
    active = np.where(weights > 0.01)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                         c=labels, cmap="tab10", s=3, alpha=0.5)
    ax.set_xlabel("dx / median_h")
    ax.set_ylabel("dy / median_h")
    ax.set_title(f"VBGMM clusters ({len(active)} active / {len(weights)} total)")
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    # Mark cluster centres
    for k in active:
        mu = model.means_[k]
        ax.plot(mu[0], mu[1], "kx", markersize=8, markeredgewidth=2)
        r = np.linalg.norm(mu)
        a = np.degrees(np.arctan2(abs(mu[1]), abs(mu[0])))
        ax.annotate(f"k={k}\nr={r:.1f} θ={a:.0f}°",
                    (mu[0], mu[1]), fontsize=7, color="black",
                    textcoords="offset points", xytext=(5, 5))

    fig.savefig("/tmp/_tmp_vbgmm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return np.array(Image.open("/tmp/_tmp_vbgmm.png").convert("RGB"))


def viz_step5_edge_labels(page_rgb, centroids, labeled_edges):
    """Draw edges coloured by semantic label: word=blue, line=red, block=green."""
    label_colours = {"word": (255, 100, 0), "line": (0, 0, 200), "block": (0, 180, 0)}

    vis = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    counts = defaultdict(int)
    for i, j, label in labeled_edges:
        c = label_colours.get(label, (128, 128, 128))
        pt1 = (int(centroids[i, 0]), int(centroids[i, 1]))
        pt2 = (int(centroids[j, 0]), int(centroids[j, 1]))
        cv2.line(vis, pt1, pt2, c, 1, cv2.LINE_AA)
        counts[label] += 1

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    summary = ", ".join(f"{l}: {c}" for l, c in sorted(counts.items()))
    cv2.putText(vis, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

    # Legend
    y0 = vis.shape[0] - 60
    for i, (lbl, col) in enumerate(label_colours.items()):
        cv2.rectangle(vis, (10, y0 + i * 18), (25, y0 + i * 18 + 12), col, -1)
        cv2.putText(vis, lbl, (30, y0 + i * 18 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return vis


def viz_step6_hierarchy(page_rgb, blocks):
    """Final hierarchy: blocks (green), lines (red), words (blue)."""
    vis = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)

    for i, block in enumerate(blocks):
        bx, by, bw, bh = block["bbox"]
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), GREEN, 2)
        cv2.putText(vis, f"B{i+1}", (bx + 4, max(by + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)

        for line in block.get("lines", []):
            lx, ly, lw, lh = line["bbox"]
            cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), RED, 1)
            for word in line.get("words", []):
                wx, wy, ww, wh = word["bbox"]
                cv2.rectangle(vis, (wx, wy), (wx + ww, wy + wh), BLUE, 1)

    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    n_blocks = len(blocks)
    n_lines = sum(len(b.get("lines", [])) for b in blocks)
    n_words = sum(len(w) for b in blocks for l in b.get("lines", [])
                  for w in [l.get("words", [])])
    label = f"{n_blocks} blocks, {n_lines} lines, {n_words} words"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2, cv2.LINE_AA)
    return vis


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def process_pdf(pdf_path, out_dir):
    """Run and visualise every pipeline step for one PDF."""
    print(f"\n{'='*60}")
    print(f"  {pdf_path.name}")
    print(f"{'='*60}")

    page_rgb = pdf_to_rgb(pdf_path, DPI)
    binary = preprocess(page_rgb)
    stem = pdf_path.stem

    # ── Step 0: Connected components ──────────────────────────────
    print("  Step 0: Extracting connected components...")
    vis0, stats, centroids = viz_step0_connected_components(page_rgb, binary)
    save_img(vis0, out_dir / f"{stem}_step0_connected_components.png")
    print(f"    {len(stats)} CCs extracted")

    # ── Step 1: Sub-character merging ─────────────────────────────
    print("  Step 1: Merging sub-character fragments...")
    vis1, m_stats, m_centroids = viz_step1_subchar_merge(page_rgb, stats, centroids)
    save_img(vis1, out_dir / f"{stem}_step1_subchar_merge.png")
    print(f"    {len(stats)} → {len(m_stats)} components")

    # ── Step 2: XY-cut pre-segmentation ───────────────────────────
    print("  Step 2: XY-cut pre-segmentation...")
    vis2, regions = viz_step2_xy_cut(page_rgb, binary)
    save_img(vis2, out_dir / f"{stem}_step2_xy_cut_regions.png")
    print(f"    {len(regions)} regions")

    # ── Steps 3–5: Per-region VBGMM (use first non-trivial region) ─
    # Assign CCs to regions
    n = len(m_stats)
    cc_to_region = np.zeros(n, dtype=np.int32)
    for cc_idx in range(n):
        cx, cy = m_centroids[cc_idx]
        for r_idx, (rx, ry, rw, rh) in enumerate(regions):
            if rx <= cx < rx + rw and ry <= cy < ry + rh:
                cc_to_region[cc_idx] = r_idx
                break

    # Pick the largest region for detailed visualisation
    region_sizes = [(cc_to_region == r).sum() for r in range(len(regions))]
    best_region = int(np.argmax(region_sizes))
    r_mask = cc_to_region == best_region
    r_stats = m_stats[r_mask]
    r_centroids = m_centroids[r_mask]
    r_n = len(r_stats)
    print(f"  Detailed viz on region {best_region+1} ({r_n} CCs)")

    # Step 3: Delaunay graph
    print("  Step 3: Building Delaunay graph...")
    vis3, edges = viz_step3_delaunay(page_rgb, r_stats, r_centroids)
    save_img(vis3, out_dir / f"{stem}_step3_delaunay_graph.png")

    if r_n >= 2 and edges:
        # Step 4: Displacement vectors + VBGMM
        print("  Step 4: Fitting VBGMM...")
        vectors_2d, pruned_edges, edge_mapping = _compute_displacement_vectors(
            edges, r_centroids, r_stats)

        if len(vectors_2d) >= 10:
            model = _fit_vbgmm(vectors_2d)
            vis4 = viz_step4_vbgmm_clusters(vectors_2d, model)
            save_img(vis4, out_dir / f"{stem}_step4_vbgmm_clusters.png")

            # Step 5: Classify + label edges
            print("  Step 5: Classifying clusters and labelling edges...")
            cluster_labels = _classify_clusters(model)
            labeled_edges = _label_edges(model, cluster_labels, vectors_2d,
                                         pruned_edges, edge_mapping)
            vis5 = viz_step5_edge_labels(page_rgb, r_centroids, labeled_edges)
            save_img(vis5, out_dir / f"{stem}_step5_edge_labels.png")
        else:
            print("    Too few edges for VBGMM, skipping steps 4-5")
    else:
        print("    Too few CCs for Delaunay, skipping steps 3-5")

    # ── Step 6: Full pipeline result ──────────────────────────────
    print("  Step 6: Running full pipeline...")
    blocks = segment(binary)
    vis6 = viz_step6_hierarchy(page_rgb, blocks)
    save_img(vis6, out_dir / f"{stem}_step6_hierarchy.png")

    n_lines = sum(len(b.get("lines", [])) for b in blocks)
    n_words = sum(len(w) for b in blocks for l in b.get("lines", [])
                  for w in [l.get("words", [])])
    print(f"    Final: {len(blocks)} blocks, {n_lines} lines, {n_words} words")
    print(f"    Saved to {out_dir}/")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    sample = random.sample(all_pdfs, min(N_SAMPLES, len(all_pdfs)))
    print(f"Detailed segmentation test on {len(sample)} PDFs\n")

    for pdf_path in sample:
        try:
            out_dir = OUTPUT_DIR / pdf_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            process_pdf(pdf_path, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. All results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
