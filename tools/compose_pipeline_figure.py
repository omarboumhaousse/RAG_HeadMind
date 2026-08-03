"""Compose the pipeline walk-through figure from the per-step renders.

The step images carry a title bar drawn by visualize_pipeline.py whose
text overflows and collides with the "ETAPE n/6" prefix, and it is in
French. Crop it off and lay the pages out with titles drawn here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "pipeline_visualization"
OUT = REPO / "tools" / "fig9_pipeline_page.png"

BAR_H = 215          # measured height of the broken title bar
TARGET_W = 1340      # per-panel width after downscaling
COLS = 2             # two panels per row: at three the pages render too
                     # small to read once GitHub scales the figure down

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

STEPS = [
    ("step1_original.png", "1. Original page", "PDF rendered at 150 DPI"),
    # step2 carries an in-panel "Original / Binarise" caption row in French;
    # crop it away, the panel subtitle already says what the stage does
    ("step2_binarisation.png", "2. Preprocessing", "Sauvola binarisation + deskew", 62),
    ("step3_segmentation.png", "3. Segmentation", "VBGMM on Delaunay displacement vectors"),
    ("step4_classification (1).png", "4. Classification", "Geometric rules, OmniDocBench labels"),
    ("step5_reading_order (1).png", "5. Reading order", "Column detection, then sort"),
    ("step6_ocr.png", "6. OCR", "PaddleOCR PP-OCRv5 + PPStructureV3"),
]


def load(name, extra_top=0):
    im = Image.open(SRC / name)
    # drop the broken bar, plus any in-panel label row we replace ourselves
    im = im.crop((0, BAR_H + extra_top, im.width, im.height))
    ratio = TARGET_W / im.width
    return im.resize((TARGET_W, int(im.height * ratio)), Image.LANCZOS)


def main():
    imgs = [load(s[0], s[3] if len(s) > 3 else 0) for s in STEPS]
    rows = -(-len(STEPS) // COLS)
    aspect = imgs[0].height / imgs[0].width
    panel_in = 7.4                                   # inches per panel
    fig_w = COLS * panel_in + 0.6
    fig_h = rows * panel_in * aspect + 1.5
    fig, axes = plt.subplots(rows, COLS, figsize=(fig_w, fig_h), facecolor=SURFACE)
    for ax, im, step in zip(axes.ravel(), imgs, STEPS):
        title, sub = step[1], step[2]
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#d8d7d0")
            s.set_linewidth(1)
        ax.set_title(title, fontsize=19, fontweight="bold", color=INK, pad=14, loc="left")
        ax.text(0, -0.016, sub, transform=ax.transAxes, fontsize=13,
                color=MUTED, va="top", ha="left")
    fig.suptitle("StatParse pipeline, stage by stage on one page",
                 fontsize=25, fontweight="bold", color=INK, y=0.988)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.945, bottom=0.028,
                        wspace=0.07, hspace=0.10)
    fig.savefig(OUT, dpi=100, facecolor=SURFACE)

    # The panels are scanned pages on a near-white ground, so a 256-colour
    # palette is visually identical at 1:1 and cuts the file from ~2.4 MB
    # to ~0.5 MB. Worth it for a figure this large in a README.
    im = Image.open(OUT).convert("RGB")
    im.convert("P", palette=Image.ADAPTIVE, colors=256).save(OUT, optimize=True)
    print("wrote", OUT, Image.open(OUT).size,
          "%.2f MB" % (OUT.stat().st_size / 1048576))


if __name__ == "__main__":
    main()
