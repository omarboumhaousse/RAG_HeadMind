"""
Quick segmentation test — runs the pipeline on up to 5 random PDFs from pdfs/
and saves block/line/word visualisations to segmentation_test_output/.
"""

import sys
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess
from statparse.segmentation import segment

# ── Configuration ─────────────────────────────────────────────────
PDF_DIR = Path("./pdfs")
OUTPUT_DIR = Path("./segmentation_test_output")
DPI = 150
N_SAMPLES = 5

# Colours (BGR for OpenCV)
GREEN = (0, 200, 0)     # blocks
RED = (200, 0, 0)       # lines
BLUE = (0, 100, 255)    # words


def pdf_to_rgb(pdf_path: Path, dpi: int) -> np.ndarray:
    """Render first page of a PDF to an RGB numpy array."""
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


def draw_hierarchy(image_rgb: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Draw blocks (green), lines (red), words (blue) on the image."""
    vis = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for i, block in enumerate(blocks):
        # Block
        bx, by, bw, bh = block["bbox"]
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), GREEN, 2)
        cv2.putText(vis, f"B{i+1}", (bx + 4, max(by + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)

        for line in block.get("lines", []):
            # Line
            lx, ly, lw, lh = line["bbox"]
            cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), RED, 1)

            for word in line.get("words", []):
                # Word
                wx, wy, ww, wh = word["bbox"]
                cv2.rectangle(vis, (wx, wy), (wx + ww, wy + wh), BLUE, 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    # Sample up to N_SAMPLES
    sample = random.sample(all_pdfs, min(N_SAMPLES, len(all_pdfs)))
    print(f"Testing {len(sample)} PDFs\n")

    for pdf_path in sample:
        print(f"  {pdf_path.name}")
        try:
            page_image = pdf_to_rgb(pdf_path, DPI)
            binary = preprocess(page_image)
            blocks = segment(binary)

            n_lines = sum(len(b.get("lines", [])) for b in blocks)
            n_words = sum(len(w)
                          for b in blocks
                          for l in b.get("lines", [])
                          for w in [l.get("words", [])])
            print(f"    {len(blocks)} blocks, {n_lines} lines, {n_words} words")

            vis = draw_hierarchy(page_image, blocks)
            out_path = OUTPUT_DIR / f"{pdf_path.stem}_hierarchy.png"
            Image.fromarray(vis).save(out_path)
            print(f"    → {out_path}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
