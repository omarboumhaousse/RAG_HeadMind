"""
Visualisation des bounding boxes produites par segmentation.py ou segmentation_bis.py.
Usage :
    python visualize_segmentation.py --method bis     # segmentation_bis.py (VBGMM)
    python visualize_segmentation.py --method original # segmentation.py (KDE + kNN)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
import random

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess

# ── Configuration ─────────────────────────────────────────────────
PDF_DIR    = Path("./pdfs")
OUTPUT_DIR = Path("./segmentation_viz")
DPI        = 150
N_PDFS     = 5

COLOR_BLOCK = (0, 200, 0)
COLOR_COMP  = (200, 100, 0)
THICKNESS   = 2


def draw_blocks(image_rgb: np.ndarray, blocks: list[dict],
                draw_components: bool = False) -> np.ndarray:
    vis_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for i, block in enumerate(blocks):
        x, y, w, h = block["bbox"]
        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h),
                      color=COLOR_BLOCK, thickness=THICKNESS)
        cv2.putText(vis_bgr, f"B{i+1}", (x + 4, max(y + 20, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLOCK, 2, cv2.LINE_AA)

        if draw_components:
            for comp in block.get("components", []):
                cx, cy, cw, ch = comp["bbox"]
                cv2.rectangle(vis_bgr, (cx, cy), (cx + cw, cy + ch),
                              color=COLOR_COMP, thickness=1)

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def process_pdf(pdf_path: Path, segment_fn, method_name: str):
    print(f"\nTraitement : {pdf_path.name}")

    pages = convert_from_path(str(pdf_path), dpi=DPI)
    page_image = np.array(pages[0])
    print(f"  Page : {page_image.shape}")

    binary = preprocess(page_image)
    blocks = segment_fn(binary)
    print(f"  Blocs détectés ({method_name}) : {len(blocks)}")

    for i, b in enumerate(blocks):
        x, y, w, h = b["bbox"]
        n_comp = len(b.get("components", []))
        print(f"    Bloc {i+1:2d} : bbox=({x:4d},{y:4d},{w:4d},{h:4d})  composantes={n_comp}")

    stem = pdf_path.stem
    out_dir = OUTPUT_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(draw_blocks(page_image, blocks, draw_components=False))\
         .save(out_dir / f"{stem}_blocks.png")

    Image.fromarray(draw_blocks(page_image, blocks, draw_components=True))\
         .save(out_dir / f"{stem}_blocks_and_components.png")

    print(f"  Sauvegardé dans : {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bis", "original", "both"],
                        default="both",
                        help="Quelle méthode de segmentation utiliser")
    args = parser.parse_args()

    # Charger les fonctions segment selon le choix
    segment_fns = {}

    if args.method in ("bis", "both"):
        from statparse.segmentation_bis import segment as segment_bis
        segment_fns["bis"] = segment_bis

    if args.method in ("original", "both"):
        from statparse.segmentation import segment as segment_original
        segment_fns["original"] = segment_original

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_pdfs = list(PDF_DIR.glob("*.pdf"))

    if not all_pdfs:
        print(f"Aucun PDF trouvé dans {PDF_DIR}")
        sys.exit(1)

    # Sélection aléatoire de N_PDFS fichiers (sans répétition)
    pdfs = random.sample(all_pdfs, min(N_PDFS, len(all_pdfs)))        

    print(f"PDFs sélectionnés aléatoirement : {len(pdfs)}  |  Méthode(s) : {list(segment_fns.keys())}")

    for pdf in pdfs:
        for method_name, segment_fn in segment_fns.items():
            try:
                process_pdf(pdf, segment_fn, method_name)
            except Exception as e:
                print(f"  ERREUR ({method_name}) sur {pdf.name} : {e}")
                import traceback
                traceback.print_exc()

    print(f"\nTerminé. Images dans : {OUTPUT_DIR}/")