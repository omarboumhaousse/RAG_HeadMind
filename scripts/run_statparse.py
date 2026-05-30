"""
run_statparse.py
---------------
Run the full StatParse pipeline on a folder of PDFs and produce
one Markdown file per page, ready for OmniDocBench end-to-end evaluation.

OmniDocBench expects one .md file per PDF page, named after the PDF stem:
    e.g.  scihub_md.0000000000002934.pdf_0.md

Usage:
    # All PDFs in pdfs_selected/
    python run_statparse.py

    # Custom input/output
    python run_statparse.py --pdf-dir ./pdfs --out-dir ./results/markdown

    # Single PDF
    python run_statparse.py --pdf pdfs_selected/scihub_md.0000000000002934.pdf_0.pdf

    # Skip already-processed files
    python run_statparse.py --resume

    # Limit to N files (useful for quick testing)
    python run_statparse.py --n 5
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess
from statparse.segmentation import segment
from statparse.classification import classify
from statparse.reading_order import order_blocks
from statparse.ocr import recognize_text
from statparse.serialization import to_markdown

# ── Defaults ───────────────────────────────────────────────────────
PDF_DIR = Path("./pdfs_selected")
OUT_DIR = Path("./pipeline_output")
DPI     = 150


# ═══════════════════════════════════════════════════════════════════
# PDF → RGB
# ═══════════════════════════════════════════════════════════════════

def pdf_to_rgb(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    """Render all pages of a PDF to RGB numpy arrays."""
    try:
        import fitz
        doc    = fitz.open(str(pdf_path))
        pages  = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            import cv2
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pages.append(img)
        doc.close()
        return pages
    except ImportError:
        from pdf2image import convert_from_path
        return [np.array(p) for p in convert_from_path(str(pdf_path), dpi=dpi)]


# ═══════════════════════════════════════════════════════════════════
# Single page pipeline
# ═══════════════════════════════════════════════════════════════════

def process_page(page_image: np.ndarray) -> str:
    """Full pipeline: image → Markdown."""
    binary      = preprocess(page_image)
    blocks      = segment(binary)
    labeled     = classify(blocks, image_shape=binary.shape)
    ordered     = order_blocks(labeled, image_shape=binary.shape)
    text_blocks = recognize_text(ordered, page_image)
    return to_markdown(text_blocks)


# ═══════════════════════════════════════════════════════════════════
# Single PDF processing
# ═══════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: Path, out_dir: Path, dpi: int,
                resume: bool = False) -> dict:
    """Process one PDF → one .md file per page."""
    stem   = pdf_path.stem
    pages  = pdf_to_rgb(pdf_path, dpi)
    n_pages = len(pages)

    results = {"stem": stem, "n_pages": n_pages,
               "success": 0, "failed": 0, "skipped": 0}

    for page_idx, page_image in enumerate(pages):
        # OmniDocBench naming: stem already includes page index
        # e.g. scihub_md.0000000000002934.pdf_0 → single page PDF
        # For multi-page PDFs: stem_page{idx}
        if n_pages == 1:
            md_name = f"{stem}.md"
        else:
            md_name = f"{stem}_page{page_idx}.md"

        md_path = out_dir / md_name

        if resume and md_path.exists():
            results["skipped"] += 1
            continue

        try:
            t0 = time.time()
            md = process_page(page_image)
            elapsed = time.time() - t0

            md_path.write_text(md, encoding="utf-8")
            results["success"] += 1
            print(f"    p{page_idx} → {md_name}  ({elapsed:.1f}s, "
                  f"{len(md)} chars)")

        except Exception as e:
            results["failed"] += 1
            print(f"    p{page_idx} ERROR: {e}")
            traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Run StatParse pipeline and produce OmniDocBench-ready Markdowns")
    p.add_argument("--pdf-dir", type=Path, default=PDF_DIR,
                   help="Folder containing input PDFs")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output folder for .md files")
    p.add_argument("--dpi",    type=int,  default=DPI,
                   help="Rendering DPI (default 150)")
    p.add_argument("--pdf",    type=Path, default=None,
                   help="Process a single specific PDF")
    p.add_argument("--n",      type=int,  default=None,
                   help="Max number of PDFs to process")
    p.add_argument("--resume", action="store_true",
                   help="Skip PDFs that already have an output .md file")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Select PDFs
    if args.pdf:
        if not args.pdf.exists():
            print(f"[ERROR] File not found: {args.pdf}")
            sys.exit(1)
        pdfs = [args.pdf]
    else:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[ERROR] No PDFs found in {args.pdf_dir}")
            sys.exit(1)
        if args.n:
            pdfs = pdfs[:args.n]

    print(f"StatParse Pipeline")
    print(f"  Input  : {args.pdf_dir if not args.pdf else args.pdf}")
    print(f"  Output : {args.out_dir}")
    print(f"  DPI    : {args.dpi}")
    print(f"  PDFs   : {len(pdfs)}")
    print(f"  Resume : {args.resume}")
    print()

    total = {"success": 0, "failed": 0, "skipped": 0}
    t_start = time.time()

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        try:
            r = process_pdf(pdf_path, args.out_dir, args.dpi, args.resume)
            for k in total:
                total[k] += r[k]
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()
            total["failed"] += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Success : {total['success']}")
    print(f"  Failed  : {total['failed']}")
    print(f"  Skipped : {total['skipped']}")
    print(f"  Output  : {args.out_dir}/")


if __name__ == "__main__":
    main()