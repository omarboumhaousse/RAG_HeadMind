"""Regenerate every StatParse result figure in English, straight from the
committed OmniDocBench score files. No number is typed by hand.

Palette: dataviz categorical slots 1 (blue) and 2 (orange). Validated -
blue/orange keeps tritan separation at dE 32.7 where the previous
blue/green pair sat at 8.2, right on the floor.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tools"

STAT = json.loads((REPO / "result" / "statparse_quick_match_metric_result.json").read_text(encoding="utf-8"))
DOCL = json.loads((REPO / "result" / "docling_quick_match_metric_result.json").read_text(encoding="utf-8"))

C_STAT = "#2a78d6"   # slot 1 blue
C_DOCL = "#eb6834"   # slot 2 orange
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7",
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def style(ax, xlabel=None, ylabel=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def grouped_barh(ax, labels, stat_vals, docl_vals, xlabel, xmax=1.0):
    """Horizontal grouped bars. Long category names stay readable."""
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, stat_vals, h, label="StatParse", color=C_STAT)
    ax.barh(y - h / 2, docl_vals, h, label="Docling", color=C_DOCL)
    for yi, v in zip(y + h / 2, stat_vals):
        ax.text(v + xmax * 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    for yi, v in zip(y - h / 2, docl_vals):
        ax.text(v + xmax * 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style(ax, xlabel=xlabel)


# ══════════════════════════════════════════════════════════════════
# fig1 - overall comparison
# ══════════════════════════════════════════════════════════════════
def fig1():
    panels = [
        ("Text blocks\nEdit distance", "lower is better",
         STAT["text_block"]["page"]["Edit_dist"]["ALL"], DOCL["text_block"]["page"]["Edit_dist"]["ALL"]),
        ("Reading order\nEdit distance", "lower is better",
         STAT["reading_order"]["page"]["Edit_dist"]["ALL"], DOCL["reading_order"]["page"]["Edit_dist"]["ALL"]),
        ("Tables\nTEDS", "higher is better",
         STAT["table"]["page"]["TEDS"]["ALL"], DOCL["table"]["page"]["TEDS"]["ALL"]),
        ("Display formulas\nEdit distance", "lower is better",
         STAT["display_formula"]["page"]["Edit_dist"]["ALL"], DOCL["display_formula"]["page"]["Edit_dist"]["ALL"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, (title, direction, sv, dv) in zip(axes, panels):
        bars = ax.bar(["StatParse", "Docling"], [sv, dv], color=[C_STAT, C_DOCL], width=0.55)
        for b, v in zip(bars, [sv, dv]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}",
                    ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
        ax.text(0.5, -0.16, direction, transform=ax.transAxes, ha="center",
                fontsize=9, color=MUTED, style="italic")
        ax.tick_params(labelsize=10)
        ax.xaxis.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("StatParse vs Docling on OmniDocBench - overall comparison",
                 fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / "fig1_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig2 - performance profile (radar). Fixes the stray cartesian axis.
# ══════════════════════════════════════════════════════════════════
def fig2():
    axes_labels = ["Text\n(1 - edit)", "Reading order\n(1 - edit)", "Tables\n(TEDS)", "Tables\n(TEDS structure)"]
    s = [1 - STAT["text_block"]["page"]["Edit_dist"]["ALL"],
         1 - STAT["reading_order"]["page"]["Edit_dist"]["ALL"],
         STAT["table"]["page"]["TEDS"]["ALL"],
         STAT["table"]["page"]["TEDS_structure_only"]["ALL"]]
    d = [1 - DOCL["text_block"]["page"]["Edit_dist"]["ALL"],
         1 - DOCL["reading_order"]["page"]["Edit_dist"]["ALL"],
         DOCL["table"]["page"]["TEDS"]["ALL"],
         DOCL["table"]["page"]["TEDS_structure_only"]["ALL"]]

    ang = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(7.5, 6.4))
    ax = fig.add_subplot(111, polar=True)          # the bug was a non-polar subplot here
    for vals, color, name in ((s, C_STAT, "StatParse"), (d, C_DOCL, "Docling")):
        v = vals + vals[:1]
        ax.plot(ang, v, color=color, linewidth=2, marker="o", markersize=7, label=name)
        ax.fill(ang, v, color=color, alpha=0.13)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10, color=INK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=MUTED)
    ax.set_title("Performance profile - all axes higher is better",
                 fontsize=13, fontweight="bold", color=INK, pad=26)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.13), frameon=False, fontsize=10)
    ax.grid(color=GRID)
    fig.subplots_adjust(left=0.16, right=0.84, top=0.86, bottom=0.10)
    fig.savefig(OUT / "fig2_global.png", dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig3 - by document type
# ══════════════════════════════════════════════════════════════════
DOC_TYPES = [
    ("research_report", "Research report"),
    ("book", "Book"),
    ("academic_literature", "Academic literature"),
    ("magazine", "Magazine"),
    ("colorful_textbook", "Colour textbook"),
    ("PPT2PDF", "Slides (PPT2PDF)"),
    ("exam_paper", "Exam paper"),
    ("newspaper", "Newspaper"),
    ("note", "Handwritten note"),
]


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, elem, title in ((axes[0], "text_block", "Text blocks"),
                            (axes[1], "reading_order", "Reading order")):
        keys = [k for k, _ in DOC_TYPES if "data_source: " + k in STAT[elem]["page"]["Edit_dist"]]
        labels = [lbl for k, lbl in DOC_TYPES if k in keys]
        sv = [STAT[elem]["page"]["Edit_dist"]["data_source: " + k] for k in keys]
        dv = [DOCL[elem]["page"]["Edit_dist"]["data_source: " + k] for k in keys]
        grouped_barh(ax, labels, sv, dv, "Edit distance (lower is better)", xmax=0.78)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Performance by document type", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig3_by_doctype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig4 - table TEDS by attribute
# ══════════════════════════════════════════════════════════════════
TABLE_ATTRS = [
    ("language: table_en", "English"),
    ("language: table_simplified_chinese", "Simplified Chinese"),
    ("language: table_en_ch_mixed", "Mixed EN/CH"),
    ("line: full_line", "Fully ruled"),
    ("line: less_line", "Sparsely ruled"),
    ("line: fewer_line", "Few rules"),
    ("line: wireless_line", "Borderless"),
    ("with_span: True", "With merged cells"),
    ("with_span: False", "No merged cells"),
]


def fig4():
    labels = [lbl for k, lbl in TABLE_ATTRS]
    sv = [STAT["table"]["group"]["TEDS"][k] for k, _ in TABLE_ATTRS]
    dv = [DOCL["table"]["group"]["TEDS"][k] for k, _ in TABLE_ATTRS]
    n = [STAT["table"]["group"]["sample_count"][k] for k, _ in TABLE_ATTRS]
    labels = [f"{lbl}  (n={c})" for lbl, c in zip(labels, n)]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    grouped_barh(ax, labels, sv, dv, "TEDS (higher is better)", xmax=0.92)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title("Table structure recovery (TEDS) by attribute",
                 fontsize=13, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_table_teds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig5 - language and layout
# ══════════════════════════════════════════════════════════════════
LANGS = [("text_language: text_english", "English"),
         ("text_language: text_simplified_chinese", "Simplified Chinese"),
         ("text_language: text_en_ch_mixed", "Mixed EN/CH")]
LAYOUTS = [("layout: single_column", "Single column"),
           ("layout: double_column", "Double column"),
           ("layout: three_column", "Three column"),
           ("layout: other_layout", "Other layout")]


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sv = [STAT["text_block"]["group"]["Edit_dist"][k] for k, _ in LANGS]
    dv = [DOCL["text_block"]["group"]["Edit_dist"][k] for k, _ in LANGS]
    grouped_barh(axes[0], [l for _, l in LANGS], sv, dv,
                 "Edit distance (lower is better)", xmax=0.82)
    axes[0].set_title("Text blocks by language", fontsize=12, fontweight="bold", color=INK, pad=10)

    sv = [STAT["text_block"]["page"]["Edit_dist"][k] for k, _ in LAYOUTS]
    dv = [DOCL["text_block"]["page"]["Edit_dist"][k] for k, _ in LAYOUTS]
    grouped_barh(axes[1], [l for _, l in LAYOUTS], sv, dv,
                 "Edit distance (lower is better)", xmax=0.62)
    axes[1].set_title("Text blocks by page layout", fontsize=12, fontweight="bold", color=INK, pad=10)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Performance by language and page layout", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig5_lang_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig6 - pipeline schematic, English, no emoji (the old one rendered
# emoji as tofu boxes)
# ══════════════════════════════════════════════════════════════════
def fig6():
    steps = [
        ("PDF\ninput", ""),
        ("Preprocessing\nBinarise + deskew", "preprocessing.py"),
        ("Segmentation\nVBGMM + Delaunay", "segmentation.py"),
        ("Classification\nGeometric rules", "classification.py"),
        ("Reading order\nColumn detection", "reading_order.py"),
        ("OCR\nPaddleOCR", "ocr.py"),
        ("Markdown\nserialisation", "serialization.py"),
    ]
    shades = ["#0d366b", "#184f95", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"]
    fig, ax = plt.subplots(figsize=(15, 3.4))
    ax.set_xlim(0, len(steps) * 2.2)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, ((label, module), col) in enumerate(zip(steps, shades)):
        x = i * 2.2 + 0.12
        ax.add_patch(plt.Rectangle((x, 0.85), 1.85, 1.7, facecolor=col,
                                   edgecolor="none", zorder=2))
        ax.text(x + 0.925, 1.7, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", zorder=3, linespacing=1.5)
        if module:
            ax.text(x + 0.925, 0.6, module, ha="center", va="center",
                    fontsize=8.5, color=MUTED, style="italic", zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.12, 1.7), xytext=(x + 1.9, 1.7),
                        arrowprops=dict(arrowstyle="-|>", color="#898781", lw=1.8))
    ax.set_title("StatParse pipeline architecture", fontsize=14, fontweight="bold",
                 color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig7 - summary table. Only metrics the harness actually produces;
# the old figure reported an undefined composite "score global".
# ══════════════════════════════════════════════════════════════════
def fig7():
    rows = [
        ("Text blocks", "Edit distance", "lower",
         STAT["text_block"]["page"]["Edit_dist"]["ALL"], DOCL["text_block"]["page"]["Edit_dist"]["ALL"]),
        ("Reading order", "Edit distance", "lower",
         STAT["reading_order"]["page"]["Edit_dist"]["ALL"], DOCL["reading_order"]["page"]["Edit_dist"]["ALL"]),
        ("Tables", "Edit distance", "lower",
         STAT["table"]["page"]["Edit_dist"]["ALL"], DOCL["table"]["page"]["Edit_dist"]["ALL"]),
        ("Tables", "TEDS", "higher",
         STAT["table"]["page"]["TEDS"]["ALL"], DOCL["table"]["page"]["TEDS"]["ALL"]),
        ("Tables", "TEDS (structure)", "higher",
         STAT["table"]["page"]["TEDS_structure_only"]["ALL"], DOCL["table"]["page"]["TEDS_structure_only"]["ALL"]),
        ("Display formulas", "Edit distance", "lower",
         STAT["display_formula"]["page"]["Edit_dist"]["ALL"], DOCL["display_formula"]["page"]["Edit_dist"]["ALL"]),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off")
    cells, colors = [], []
    for elem, metric, direction, sv, dv in rows:
        arrow = "lower is better" if direction == "lower" else "higher is better"
        gap = (dv - sv) if direction == "lower" else (sv - dv)
        cells.append([elem, metric, arrow, f"{sv:.3f}", f"{dv:.3f}", f"{abs(gap):.3f}"])
        colors.append(["#ffffff"] * 3 + ["#eaf2fd", "#fdeee7", "#ffffff"])
    t = ax.table(cellText=cells,
                 colLabels=["Element", "Metric", "Direction", "StatParse", "Docling", "Gap"],
                 cellColours=colors, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.75)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        if r == 0:
            cell.set_facecolor("#0d366b")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Score summary - StatParse vs Docling on OmniDocBench",
                 fontsize=13, fontweight="bold", color=INK, pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# fig8 - strengths and weaknesses. Every number recomputed from the
# JSON; the old figure quoted stale values.
# ══════════════════════════════════════════════════════════════════
def fig8():
    tb = STAT["text_block"]["page"]["Edit_dist"]
    ro = STAT["reading_order"]["page"]["Edit_dist"]
    strengths = [
        f"Research reports: text edit {tb['data_source: research_report']:.3f} (best document type)",
        f"Three-column layout: text edit {tb['layout: three_column']:.3f}",
        f"Books: text edit {tb['data_source: book']:.3f}",
        f"Reading order on research reports: {ro['data_source: research_report']:.3f}",
        "Layout analysis is training-free and runs on CPU only",
        "Handles both Latin and CJK scripts",
    ]
    weaknesses = [
        f"Tables: TEDS {STAT['table']['page']['TEDS']['ALL']:.3f} vs {DOCL['table']['page']['TEDS']['ALL']:.3f} for Docling",
        f"Display formulas: edit {STAT['display_formula']['page']['Edit_dist']['ALL']:.3f}, no LaTeX recognition",
        f"Handwritten notes: text edit {tb['data_source: note']:.3f}",
        f"Newspapers: text edit {tb['data_source: newspaper']:.3f}",
        f"Mixed EN/CH pages: text edit {tb['language: en_ch_mixed']:.3f}",
        "Captions are still confused with body text",
    ]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # transAxes keeps every string inside the canvas; drawing in data
    # coordinates let long lines escape the axes and blew the saved
    # figure up to 16000px wide.
    tx = ax.transAxes
    ax.text(0.015, 0.95, "Strengths", fontsize=13, fontweight="bold",
            color="#0d366b", transform=tx, va="top")
    ax.text(0.525, 0.95, "Areas for improvement", fontsize=13, fontweight="bold",
            color="#a8391a", transform=tx, va="top")
    for i, s in enumerate(strengths):
        ax.text(0.015, 0.79 - i * 0.135, s, fontsize=10, color=INK2,
                transform=tx, va="top")
    for i, w in enumerate(weaknesses):
        ax.text(0.525, 0.79 - i * 0.135, w, fontsize=10, color=INK2,
                transform=tx, va="top")
    ax.plot([0.505, 0.505], [0.02, 0.98], color=GRID, lw=1.2, transform=tx)
    ax.set_title("StatParse - strengths and areas for improvement",
                 fontsize=14, fontweight="bold", color=INK, pad=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.03)
    fig.savefig(OUT / "fig8_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
        print("wrote", f.__name__)
