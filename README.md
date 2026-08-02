# StatParse: A Statistical Document Parsing Pipeline for RAG Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**StatParse** is a lightweight, fully statistical document parsing pipeline that converts PDF documents into structured Markdown — without relying on deep learning or GPU resources.

The core idea: at every level of document structure (characters → words → lines → blocks → columns), spatial relationships follow **statistical distributions**. By modeling these distributions explicitly (using Variational Bayesian Gaussian Mixture Models and adaptive thresholding), we can segment and classify document elements with **zero training data** and **zero GPU compute** for the layout analysis stages.

StatParse is designed to plug into Retrieval-Augmented Generation (RAG) systems as the parsing front-end.

## Motivation

Modern document parsing methods (LayoutParser, Docling, DiT) achieve high accuracy but depend on large pretrained models. We explore how far **classical statistics and computer vision** can go, producing an interpretable, reproducible, and resource-free alternative suitable for constrained environments.

This is a research project. We benchmark StatParse against [Docling](https://github.com/DS4SD/docling) (our baseline) on standard document layout analysis benchmarks, then integrate both into a RAG system for end-to-end comparison.

## Pipeline Architecture
PDF ──► Page Images ──► Preprocessing ──► Geometric Segmentation ──► Semantic Classification ──► Reading Order ──► OCR ──► Markdown

| Stage | Method | Key Technique |
|-------|--------|---------------|
| **1. PDF to Image** | Rendering at 150 DPI (default) | `pdf2image` / `PyMuPDF` |
| **2. Preprocessing** | Binarization, deskew, noise removal | CLAHE contrast enhancement, Sauvola thresholding, `minAreaRect`-based deskew |
| **3a. Geometric Segmentation** | Hierarchical spatial clustering | Connected components → Delaunay triangulation → VBGMM on 2D displacement vectors → word / line / block labelling |
| **3b. Semantic Classification** | Rule-based classifier | Features: font size ratios, position, aspect ratio, density, line count — pure if/else decision tree |
| **4. Reading Order** | Column-aware sorting | X-projection gap analysis → column assignment → top-to-bottom sort per column, left-to-right across columns |
| **5. OCR** | PaddleOCR PP-OCRv5 + PPStructureV3 | Neural OCR for text blocks; PPStructureV3 for structured Markdown table extraction |
| **6. Markdown Serialization** | Deterministic label mapping | OmniDocBench label → Markdown syntax (title → `#`, table pass-through, equation → `$$`, etc.) |

### Geometric Segmentation Detail

The core novelty lies in step 3a. The algorithm works as follows:

1. Extract connected components (ink blobs) from the binarized page
2. Merge sub-character fragments (CJK strokes, diacritics) via spatial proximity
3. XY-cut pre-segmentation splits the page at large whitespace gaps so each region has a homogeneous gap distribution
4. Build a neighbourhood graph via **Delaunay triangulation** (k-NN used as fallback only)
5. Compute normalised 2D displacement vectors between neighbours
6. Fit a **VBGMM** (Variational Bayesian GMM) — automatically selects the number of gap types
7. Classify clusters by polar angle: angle < 30° → word gap; angle ≥ 30° → line or block gap
8. Label every edge via MAP decision rule and merge components hierarchically (3-pass union-find)

```
Connected Components
        │
        ▼
   Sub-character merging (cKDTree proximity)
        │
        ▼
   XY-cut pre-segmentation
        │
        ▼
   Delaunay triangulation edges
        │
        ▼
   2D displacement vectors (normalised by median char height)
        │
        ▼
   VBGMM fitting (auto-selects K*)
        │
        ├── angle < 30° ──► word gap
        ├── angle ≥ 30°, small radius ──► line gap
        └── angle ≥ 30°, large radius ──► block gap
```

This approach is inspired by the **Docstrum algorithm** (O'Gorman, 1993) but replaces fixed k-NN with Delaunay triangulation and uses VBGMM instead of manual threshold tuning.

## Project Structure

```
RAG_HeadMind/
├── statparse/          # Core pipeline package (6-step pipeline)
│   ├── pipeline.py         # End-to-end Pipeline class
│   ├── pdf_to_image.py     # Step 1: PDF rendering
│   ├── preprocessing.py    # Step 2: Binarization, deskew, denoising
│   ├── segmentation.py     # Step 3a: Geometric segmentation (VBGMM)
│   ├── classification.py   # Step 3b: Semantic block classification
│   ├── reading_order.py    # Step 4: Block ordering
│   ├── ocr.py              # Step 5: PaddleOCR integration
│   └── serialization.py    # Step 6: Markdown output
├── dataset/            # OmniDocBench dataset loaders
├── metrics/            # Evaluation metrics (TEDS, CDM, Edit Distance, BLEU)
├── task/               # Evaluation task runners
├── registry/           # Plugin registry (datasets, metrics, tasks)
├── utils/              # Shared utilities (matching, table conversion, etc.)
├── tools/              # Docling baseline runner, result figures and notebooks
├── result/             # Committed benchmark scores (see Results)
├── configs/            # YAML evaluation configurations
├── viz/                # Visualization outputs (gitignored)
│   ├── compare/            # GT vs pipeline classification & reading order
│   └── pipeline/           # Per-stage pipeline visualizations
├── scripts/            # Runnable scripts
│   ├── run_statparse.py            # Run pipeline on a folder of PDFs
│   ├── pdf_validation.py           # Run evaluation against ground truth
│   ├── visualize_pipeline.py       # Visualize each pipeline stage
│   ├── visualize_results.py        # Visualize evaluation results
│   ├── compare_classification.py   # Compare GT vs pipeline classification
│   └── download_models.py          # Download PaddleOCR models (one-time setup)
├── requirements.txt
├── setup_environment.ipynb  # Step-by-step environment setup (Onyxia)
└── README.md
```

## Installation

### On Onyxia (recommended)

Open `setup_environment.ipynb` and run the cells in order. It installs all dependencies, pins the correct PaddleOCR versions, and downloads the models (~3–5 min).

### Manual setup

```bash
# 1. System dependency (PDF rendering)
sudo apt-get install -y poppler-utils     # Ubuntu/Debian
brew install poppler                       # macOS

# 2. Python dependencies
git clone https://github.com/omarboumhaousse/RAG_HeadMind.git
cd RAG_HeadMind
pip install -r requirements.txt
pip install opencv-python-headless        # headless override for servers

# 3. PaddleOCR — strict version order required
pip install paddlepaddle==3.0.0
pip install paddleocr==3.0.3
pip install paddlex==3.0.3               # downgrade from 3.5.0 pulled by paddleocr

# 4. Download OCR models (~3–5 min, saved to ~/.paddlex/official_models/)
python scripts/download_models.py
```

No GPU required. StatParse runs entirely on CPU.

## Usage

### Basic

```python
from statparse import Pipeline

pipeline = Pipeline()
pages = pipeline.parse("document.pdf")

# Save as markdown
with open("output.md", "w") as f:
    f.write("\n\n---\n\n".join(pages))
```

### Step-by-Step

```python
from statparse.pdf_to_image import render_pdf
from statparse.preprocessing import preprocess
from statparse.segmentation import segment
from statparse.classification import classify
from statparse.reading_order import order_blocks
from statparse.ocr import recognize_text
from statparse.serialization import to_markdown

images = render_pdf("document.pdf", dpi=300)

for page_image in images:
    clean = preprocess(page_image)
    blocks = segment(clean)
    labeled_blocks = classify(blocks, image_shape=clean.shape)
    ordered_blocks = order_blocks(labeled_blocks, image_shape=clean.shape)
    text_blocks = recognize_text(ordered_blocks, page_image)
    markdown = to_markdown(text_blocks)
```

## Evaluation

### Evaluation framework

The folders `configs/`, `registry/`, `dataset/`, `metrics/`, `task/`, and `utils/` are taken directly from the [OmniDocBench repository](https://github.com/opendatalab/OmniDocBench) and form a self-contained evaluation harness. StatParse does not depend on them at runtime — they only come into play when scoring parser output.

The two systems are fully decoupled: StatParse writes Markdown files to `result/statparse/`, and the OmniDocBench harness reads those files, matches them against the ground truth, and computes scores. Nothing in `statparse/` imports from these folders.

### Benchmark

We evaluate on **[OmniDocBench](https://github.com/opendatalab/OmniDocBench)**, a comprehensive benchmark for end-to-end document parsing covering diverse document types (scientific papers, books, financial reports, slides, newspapers).

Ground truth is loaded from `OmniDocBench.json`. Predictions are matched to ground truth blocks via `quick_match` (fast alignment algorithm).

### Metrics

Metrics are configured per element type in `configs/end2end_statparse.yaml`:

| Element type | Metric | What it measures |
|---|---|---|
| Text blocks | Edit Distance | Character-level accuracy of extracted text |
| Tables | TEDS + Edit Distance | Tree Edit Distance Similarity for table structure + text accuracy |
| Formulas | CDM + Edit Distance | Content Difference Metric for formula visual similarity + text accuracy |
| Reading order | Edit Distance | Sequence-level correctness of block ordering |

### Running Evaluation

There are two config files, one per parser being evaluated. They are identical except for the prediction path:

| Config | Evaluates | Prediction path |
|--------|-----------|-----------------|
| `configs/end2end_statparse.yaml` | StatParse output | `./result/statparse` |
| `configs/end2end.yaml` | Docling baseline output | `./result/docling` |

```bash
# Evaluate StatParse
python scripts/pdf_validation.py --config ./configs/end2end_statparse.yaml

# Evaluate Docling baseline
python scripts/pdf_validation.py --config ./configs/end2end.yaml
```

Run both to compare StatParse against Docling on the same OmniDocBench ground truth.

## Results

Both systems were scored on the same OmniDocBench ground truth, with the same `quick_match` alignment and the same metric code: 1,290 pages for text blocks and reading order, 351 pages containing tables, and 160 pages containing display formulas. Raw outputs are committed in `result/` — `statparse_quick_match_metric_result.json` and `docling_quick_match_metric_result.json`.

All figures below are page-level averages, so they match the numbers reported in the JSON under `page → ALL`.

### Overall comparison

| Element | Metric | Direction | StatParse | Docling |
|---|---|---|---|---|
| Text blocks | Edit distance | ↓ lower is better | **0.384** | 0.176 |
| Reading order | Edit distance | ↓ lower is better | **0.353** | 0.183 |
| Tables | Edit distance | ↓ lower is better | **0.834** | 0.643 |
| Tables | TEDS | ↑ higher is better | **0.156** | 0.667 |
| Tables | TEDS (structure only) | ↑ higher is better | **0.174** | 0.737 |
| Display formulas | Edit distance | ↓ lower is better | **0.960** | 0.345 |

### Resource requirements

| | StatParse | Docling |
|---|---|---|
| **GPU required** | No — CPU only | Recommended (layout and table transformers) |
| **Training data** | None — no pretrained or fitted weights | Pretrained layout and table models |
| **Learned components** | VBGMM fitted per page at inference time | Deep models trained offline |

StatParse uses PaddleOCR for text recognition only (step 5). Layout analysis — preprocessing, segmentation, classification, and reading order (steps 2–4) — involves no learned model and no training data. This is the part of the pipeline the results below are really measuring.

### Text blocks by document type

Edit distance, page average (↓ lower is better).

| Document type | StatParse | Docling |
|---|---|---|
| Research report | **0.187** | 0.037 |
| Book | 0.278 | 0.116 |
| Academic literature | 0.310 | 0.063 |
| Magazine | 0.311 | 0.093 |
| Colorful textbook | 0.345 | 0.227 |
| PPT2PDF | 0.385 | 0.096 |
| Exam paper | 0.414 | 0.218 |
| Newspaper | 0.498 | 0.312 |
| Handwritten note | 0.651 | 0.421 |

### Text blocks by language and layout

Language rows are block-level averages (`group → text_language`); layout rows are page-level averages (`page`). Edit distance, ↓ lower is better.

| Split | StatParse | Docling |
|---|---|---|
| English text | 0.400 | 0.153 |
| Simplified Chinese text | 0.485 | 0.218 |
| Mixed EN/CH text | 0.641 | 0.462 |
| Single column | 0.355 | 0.168 |
| Double column | 0.411 | 0.131 |
| Three column | **0.234** | 0.144 |
| Other layout | 0.467 | 0.266 |

### Interpretation

Docling outperforms StatParse on every element type. That is the expected outcome and we do not claim otherwise. The question this benchmark answers is *how much* performance a fully statistical pipeline gives up, and on which parts of the problem.

**Text and reading order — the hypothesis holds.** Expressed as character-level accuracy (1 − edit distance), StatParse reaches 0.616 against Docling's 0.824 on text blocks, retaining roughly 75% of the baseline's accuracy. On reading order the figures are 0.647 against 0.817, roughly 79%. These are precisely the stages where StatParse replaces a neural model with statistics — VBGMM segmentation, a rule-based classifier, X-projection column detection — and it recovers about three quarters of a GPU baseline with no training data and no GPU.

The gap narrows on structured, regular documents: research reports (0.187) and three-column layouts (0.234) are where StatParse comes closest to Docling. This is consistent with the method's core assumption, that inter-component spacing follows a clean multimodal distribution. Performance degrades exactly where that assumption breaks down — handwritten notes (0.651) and newspapers (0.498), which have irregular or heterogeneous spacing.

**Tables and formulas — the hypothesis does not hold.** A TEDS of 0.156 against 0.667, and a formula edit distance of 0.960, mean StatParse essentially fails on these two element types. Both require recovering internal structure — cell grids, LaTeX syntax — which is a recognition problem rather than a spatial-clustering problem, and the statistical approach has no mechanism for it. Closing this gap would require either a dedicated model or a substantially different algorithm; the current PPStructureV3 pass-through does not.

**Takeaway.** A GPU-free, training-free pipeline is a credible option for text extraction and reading order in resource-constrained environments, and is not currently a viable one for table and formula parsing.

## Roadmap

- [x] Literature review and pipeline design
- [x] Implement Docling baseline and compute baseline metrics
- [x] Implement preprocessing (binarization, deskew, denoising)
- [x] Implement geometric segmentation (connected components + hierarchical grouping)
- [x] Implement semantic classification
- [x] Implement reading order
- [x] Integrate PaddleOCR
- [x] Implement Markdown serialization
- [x] Benchmark StatParse vs Docling on layout analysis
- [ ] Integrate both methods into a RAG system
- [ ] End-to-end RAG evaluation (parsing → retrieval → generation)
- [ ] Write and submit paper

## Key References

| Paper | Relevance |
|-------|-----------|
| O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE TPAMI, 1993. [DOI](https://doi.org/10.1109/34.244674) | Core method: k-NN distance/angle distributions for layout analysis |
| Ha, Haralick & Phillips, "Recursive X-Y Cut," ICDAR, 1995. [DOI](https://doi.org/10.1109/ICDAR.1995.598983) | Hierarchical top-down segmentation |
| Wong, Casey & Wahl, "Document Analysis System," IBM JRD, 1982. [DOI](https://doi.org/10.1147/rd.261.0198) | RLSA and projection profiles |
| Breuel, "Two Geometric Algorithms for Layout Analysis," DAS, 2002. [DOI](https://doi.org/10.1007/3-540-45869-7_29) | Whitespace-based segmentation |
| Kise, Sato & Iwata, "Segmentation Using Area Voronoi Diagram," CVIU, 1998. [DOI](https://doi.org/10.1006/cviu.1997.0626) | Voronoi-based approach for complex layouts |
| Binmakhashen & Mahmoud, "Document Layout Analysis: A Comprehensive Survey," ACM CSUR, 2019. [DOI](https://doi.org/10.1145/3355610) | Modern survey covering classical and DL methods |
| Mao, Rosenfeld & Kanungo, "Document Structure Analysis Algorithms: A Literature Survey," SPIE, 2003. [DOI](https://doi.org/10.1117/12.502system) | Foundational survey |
| Auer et al., "Docling Technical Report," arXiv, 2024. [arXiv](https://arxiv.org/abs/2408.09869) | Our baseline system |

## Contributing

This is an academic research project. If you want to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/step-3a-segmentation`)
3. Write tests for your module
4. Submit a pull request with a description of what you implemented and why

Please follow the module structure. Each pipeline step is an independent module with clear input/output contracts.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is conducted as part of a student research initiative exploring lightweight statistical alternatives to deep learning for document understanding in RAG systems.