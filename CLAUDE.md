# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**StatParse** — a statistical PDF parsing pipeline that converts PDFs to Markdown for RAG systems, using classical computer vision (no deep learning for layout analysis). It is benchmarked against Docling on OmniDocBench.

There are two completely independent systems in this repo:
1. **`statparse/`** — the parser itself (the research contribution)
2. **`dataset/`, `metrics/`, `task/`, `registry/`, `utils/`, `tools/`** — the OmniDocBench evaluation harness, taken as-is from https://github.com/opendatalab/OmniDocBench. StatParse does not depend on these at runtime.

## Common commands

All commands run from the **repo root**.

```bash
# Run the full pipeline on a folder of PDFs → produces Markdown in result/statparse/
python scripts/run_statparse.py --pdf-dir ./pdfs --out-dir ./result/statparse

# Run on a single PDF
python scripts/run_statparse.py --pdf ./pdfs/foo.pdf

# Evaluate StatParse output against OmniDocBench ground truth
python scripts/pdf_validation.py --config ./configs/end2end_statparse.yaml

# Evaluate Docling baseline
python scripts/pdf_validation.py --config ./configs/end2end.yaml

# Visualize all pipeline stages on one PDF (outputs to viz/pipeline/)
python scripts/visualize_pipeline.py --pdf ./pdfs/foo.pdf

# Compare GT vs pipeline side-by-side (outputs to viz/compare/)
python scripts/compare_classification.py

# Download PaddleOCR models — run once after installing dependencies
python scripts/download_models.py
```

## Environment setup (Onyxia)

Open `setup_environment.ipynb` and run cells in order. Key constraints:

- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` and `PADDLE_PDX_MODEL_SOURCE=BOS` **must be set before any paddleocr/paddlex import**, including at module load time — `statparse/ocr.py` sets them at the top of the file for this reason.
- PaddleOCR versions must be installed in strict order: `paddlepaddle==3.0.0` → `paddleocr==3.0.3` → `paddlex==3.0.3`. Installing paddleocr first pulls paddlex==3.5.0 which is incompatible.
- Use `opencv-python-headless` instead of `opencv-python` on Onyxia (no display server).

## Pipeline architecture (`statparse/`)

Six sequential steps, each an independent module with a clear function signature:

```
render_pdf()        pdf_to_image.py    PDF → list of RGB numpy arrays
preprocess()        preprocessing.py   RGB → binary image (CLAHE + Sauvola + deskew)
segment()           segmentation.py    binary → list of block dicts with bbox/lines/words
classify()          classification.py  blocks → blocks with 'label' key
order_blocks()      reading_order.py   labeled blocks → reordered blocks with 'order' key
recognize_text()    ocr.py             ordered blocks + original RGB → blocks with 'text' key
to_markdown()       serialization.py   text blocks → Markdown string
```

The `Pipeline` class in `pipeline.py` chains all six steps. `__init__.py` exports only `Pipeline`.

**Segmentation** is the core innovation: Delaunay triangulation on connected components → 2D displacement vectors → VBGMM (auto-selects cluster count) → clusters classified by polar angle into word/line/block gaps → 3-pass union-find hierarchical merge. k-NN is only a fallback when Delaunay fails (collinear points).

**Classification** is a pure rule-based if/else decision tree in `_classify_block()` / `_classify_real()`. No trained model. Features are geometric: relative height/width, density, line count, component height ratio.

**OCR** uses lazy-loaded singletons (`_OCR_ENGINE`, `_TABLE_ENGINE`) so models load only once per process. Tables go through PPStructureV3 → Markdown; all other blocks go through PaddleOCR plain text.

## OmniDocBench evaluation harness

Entry point is `scripts/pdf_validation.py` which reads a YAML config and drives everything through registries:

- `DATASET_REGISTRY["end2end_dataset"]` → `dataset/end2end_dataset.py` loads OmniDocBench.json + prediction Markdown files, extracts and matches elements
- `EVAL_TASK_REGISTRY["end2end_eval"]` → `task/end2end_run_eval.py` orchestrates metric computation
- `METRIC_REGISTRY["Edit_dist" / "TEDS" / "CDM_plain"]` → `metrics/cal_metric.py` / `table_metric.py` / `cdm_metric.py`

The two configs differ only in prediction path: `end2end_statparse.yaml` → `result/statparse/`, `end2end.yaml` → `result/docling/`.

`utils/extract.py::md_tex_filter()` is the central parsing function used by datasets — it splits a Markdown string into text, equations, HTML tables, and LaTeX tables.

## Key facts

- **DPI**: `Pipeline` defaults to 150 DPI. `pdf_to_image.py::render_pdf()` defaults to 300 DPI. Scripts use 150 DPI via PyMuPDF (`fitz`) directly, not via `render_pdf()`.
- **Output dirs** (`viz/`, `result/`, `pdfs/`) are gitignored. The `viz/` folder has two subfolders in use: `viz/compare/` and `viz/pipeline/`.
- **Known bug** in `reading_order.py` line 283: `fw_ptr == fw_ptr` is always `True` (should compare to a saved value) — stall detection in `_interleave_fullwidth` fires prematurely.
- **Dead code** in `preprocessing.py`: `morph()` is defined but never called.
- `statparse/pipeline.py` imports `numpy` but never uses it.
- `serialization.py` always uses heading level 1 (`block.get("heading_level", 1)`) — no heading level inference is implemented.
