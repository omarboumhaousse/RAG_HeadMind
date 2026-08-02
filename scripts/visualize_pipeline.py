"""
visualize_pipeline.py
---------------------
Visualise les étapes du pipeline StatParse sur un PDF donné.

Génère une image par étape + une image finale combinée.

Usage:
    python visualize_pipeline.py --pdf pdfs_selected/scihub_md.0000000000002934.pdf_0.pdf
    python visualize_pipeline.py --pdf pdfs_selected/mon_pdf.pdf --out-dir ./viz/pipeline
    python visualize_pipeline.py --pdf pdfs_selected/mon_pdf.pdf --dpi 150 --no-ocr

Sorties dans out-dir/:
    step1_original.png
    step2_binarisation.png
    step3_segmentation.png
    step4_classification.png
    step5_reading_order.png
    step6_ocr.png          (si --no-ocr non spécifié)
    pipeline_overview.png  (toutes les étapes côte à côte)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess
from statparse.segmentation import segment
from statparse.classification import classify
from statparse.reading_order import order_blocks

# ── Palette OmniDocBench ───────────────────────────────────────────
PALETTE = {
    "title":             (220,  50,  50),
    "text_block":        ( 30, 100, 200),
    "figure":            ( 34, 160,  34),
    "figure_caption":    ( 80, 200,  80),
    "figure_footnote":   (120, 210, 120),
    "table":             (255, 165,   0),
    "table_caption":     (255, 200,  80),
    "table_footnote":    (200, 140,   0),
    "equation_isolated": ( 89,  13, 130),
    "header":            (  0, 180, 180),
    "footer":            (180, 140,   0),
    "page_number":       (130, 130, 130),
    "abandon":           (200, 200, 200),
    "code_txt":          ( 50,  50,  50),
    "unknown":           (150, 150, 150),
}
SHORT = {
    "title": "TITLE", "text_block": "TEXT", "figure": "FIG",
    "figure_caption": "FIG-CAP", "table": "TABLE",
    "table_caption": "TBL-CAP", "table_footnote": "TBL-FN",
    "equation_isolated": "EQ", "header": "HDR", "footer": "FTR",
    "page_number": "PG#", "abandon": "ABAND", "code_txt": "CODE",
}
RO_COLORS = [
    (220,50,50),(50,150,220),(34,160,34),(220,130,0),
    (150,0,200),(0,180,180),(180,140,0),(220,20,140),
    (255,100,0),(0,100,180),(100,200,50),(200,0,100),
]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()


def pdf_to_rgb(pdf_path: Path, dpi: int) -> np.ndarray:
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    except ImportError:
        from pdf2image import convert_from_path
        return np.array(convert_from_path(str(pdf_path), dpi=dpi)[0])


def add_title_bar(img_rgb: np.ndarray, title: str,
                  step_num: int, total_steps: int,
                  bg_color=(30, 30, 50)) -> np.ndarray:
    """Ajoute une barre de titre en haut de l'image."""
    h, w = img_rgb.shape[:2]
    bar_h = max(52, h // 18)
    bar = np.full((bar_h, w, 3), bg_color, dtype=np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)

    # Numéro d'étape
    step_font = _font(bar_h // 2)
    step_text = f"ÉTAPE {step_num}/{total_steps}"
    draw.text((12, bar_h//2 - bar_h//4), step_text,
              fill=(100, 150, 255), font=step_font)

    # Titre
    title_font = _font(bar_h // 2 + 2)
    tb = draw.textbbox((0, 0), title, font=title_font)
    tx = w // 2 - (tb[2] - tb[0]) // 2
    draw.text((tx, bar_h//2 - bar_h//4), title,
              fill=(255, 255, 255), font=title_font)

    bar_arr = np.array(pil)
    return np.vstack([bar_arr, img_rgb])


def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w: return img
    scale = target_w / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Image originale
# ═══════════════════════════════════════════════════════════════════

def viz_step1_original(page_rgb: np.ndarray, total: int) -> np.ndarray:
    print("  Étape 1: Image originale...")
    img = add_title_bar(page_rgb.copy(), "Original page (PDF render)", 1, total,
                        bg_color=(20, 40, 80))
    return img


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Binarisation (Sauvola)
# ═══════════════════════════════════════════════════════════════════

def viz_step2_binarisation(page_rgb: np.ndarray,
                            binary: np.ndarray, total: int) -> np.ndarray:
    print("  Étape 2: Binarisation Sauvola...")
    h, w = page_rgb.shape[:2]

    # Convertir binary (uint8, 0=noir, 255=blanc) en RGB
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # Côte à côte : original (petit) + binarisé
    target_h = h
    orig_small = cv2.resize(page_rgb, (w // 3, target_h))
    bin_large  = cv2.resize(binary_rgb, (w - w//3 - 4, target_h))

    # Séparateur
    sep = np.full((target_h, 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([orig_small, sep, bin_large])

    # Annotation
    pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil)
    font = _font(max(14, h // 60))
    draw.text((10, 10), "Original", fill=(200, 100, 50), font=font)
    draw.text((w//3 + 12, 10), "Binarisé (Sauvola thresholding)", fill=(30, 30, 30), font=font)
    combined = np.array(pil)

    return add_title_bar(combined, "Preprocessing: Sauvola binarisation", 2, total,
                          bg_color=(40, 20, 80))


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Segmentation
# ═══════════════════════════════════════════════════════════════════

def viz_step3_segmentation(page_rgb: np.ndarray,
                            blocks: list[dict], total: int) -> np.ndarray:
    print(f"  Étape 3: Segmentation ({len(blocks)} blocs)...")
    img = Image.fromarray(page_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = page_rgb.shape[:2]
    font = _font(max(10, h // 90))

    # Palette de couleurs pour les blocs (cycle)
    colors = [
        (220, 50, 50), (50, 120, 220), (34, 160, 34), (220, 130, 0),
        (150, 0, 200), (0, 180, 180), (180, 50, 100), (100, 180, 0),
        (220, 180, 0), (0, 100, 220), (180, 80, 0), (80, 180, 150),
    ]

    for i, block in enumerate(blocks):
        color = colors[i % len(colors)]
        bx, by, bw, bh = block["bbox"]
        draw.rectangle([bx, by, bx+bw, by+bh],
                        outline=color, width=2, fill=color+(15,))

        # Numéro du bloc
        num = str(i + 1)
        tb = draw.textbbox((0, 0), num, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        r = max(12, h // 80)
        cx, cy = bx + bw//2, by + bh//2
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
        draw.text((cx - tw//2, cy - th//2), num, fill=(255,255,255), font=font)

    # Légende
    result = np.array(img)
    h2, w2 = result.shape[:2]
    leg_h = 40
    leg = np.full((leg_h, w2, 3), 245, dtype=np.uint8)
    pleg = Image.fromarray(leg)
    dleg = ImageDraw.Draw(pleg)
    fleg = _font(14)
    n_blocks = len(blocks)
    n_lines  = sum(len(b.get("lines", [])) for b in blocks)
    n_words  = sum(len(l.get("words", [])) for b in blocks for l in b.get("lines", []))
    text = f"{n_blocks} blocs   |   {n_lines} lignes   |   {n_words} mots détectés   (VBGMM + Delaunay + XY-cut)"
    dleg.text((10, 12), text, fill=(50, 50, 50), font=fleg)
    result = np.vstack([result, np.array(pleg)])

    return add_title_bar(result, "Segmentation: VBGMM + Delaunay triangulation", 3, total,
                          bg_color=(20, 60, 40))


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Classification
# ═══════════════════════════════════════════════════════════════════

def viz_step4_classification(page_rgb: np.ndarray,
                              labeled: list[dict], total: int) -> np.ndarray:
    from collections import Counter
    print(f"  Étape 4: Classification ({len(labeled)} blocs)...")
    img = Image.fromarray(page_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = page_rgb.shape[:2]
    font = _font(max(10, h // 90))
    pad = 2

    for block in labeled:
        label = block.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        text  = SHORT.get(label, label[:6].upper())
        bx, by, bw, bh = block["bbox"]
        draw.rectangle([bx, by, bx+bw, by+bh],
                        outline=color, width=2, fill=color+(20,))
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        tx, ty = bx + pad, by + pad if bh >= th + 2*pad else max(0, by - th - 2*pad)
        draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=color)
        draw.text((tx, ty), text, fill=(255,255,255), font=font)

    # Légende des catégories présentes
    result = np.array(img)
    counts = Counter(b.get("label", "unknown") for b in labeled)
    present = sorted(counts.keys())
    leg_h = 48
    leg = np.full((leg_h, result.shape[1], 3), 245, dtype=np.uint8)
    pleg = Image.fromarray(leg)
    dleg = ImageDraw.Draw(pleg)
    fleg = _font(13)
    x = 8
    for lbl in present:
        if lbl.startswith("_"): continue
        color = PALETTE.get(lbl, PALETTE["unknown"])
        text  = f"{SHORT.get(lbl, lbl[:6])}:{counts[lbl]}"
        tb = dleg.textbbox((0, 0), text, font=fleg)
        tw = tb[2]-tb[0]
        dleg.rectangle([x, 14, x+14, 34], fill=color)
        dleg.text((x+18, 16), text, fill=(30,30,30), font=fleg)
        x += tw + 32
        if x > result.shape[1] - 60: break
    result = np.vstack([result, np.array(pleg)])

    return add_title_bar(result, "Classification: geometric heuristics", 4, total,
                          bg_color=(60, 20, 60))


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Ordre de lecture
# ═══════════════════════════════════════════════════════════════════

def viz_step5_reading_order(page_rgb: np.ndarray,
                             ordered: list[dict], total: int) -> np.ndarray:
    print(f"  Étape 5: Ordre de lecture ({len(ordered)} blocs)...")
    img = Image.fromarray(page_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = page_rgb.shape[:2]
    font_num = _font(max(14, h // 70))
    font_lbl = _font(max(10, h // 100))

    centres = [(b["bbox"][0]+b["bbox"][2]//2,
                b["bbox"][1]+b["bbox"][3]//2) for b in ordered]

    # Blocs
    for block in ordered:
        label = block.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        bx, by, bw, bh = block["bbox"]
        draw.rectangle([bx, by, bx+bw, by+bh],
                        outline=color, width=2, fill=color+(12,))

    # Flèches
    for i in range(len(centres) - 1):
        x1, y1 = centres[i]
        x2, y2 = centres[i+1]
        color = RO_COLORS[i % len(RO_COLORS)]
        draw.line([(x1,y1),(x2,y2)], fill=color+(180,), width=3)
        import math
        ang = math.atan2(y2-y1, x2-x1)
        al = max(14, h//90)
        sp = 0.4
        pts = [(x2,y2),
               (int(x2-al*math.cos(ang-sp)), int(y2-al*math.sin(ang-sp))),
               (int(x2-al*math.cos(ang+sp)), int(y2-al*math.sin(ang+sp)))]
        draw.polygon(pts, fill=color+(200,))

    # Cercles numérotés
    r = max(15, h//85)
    for i, (block, (cx, cy)) in enumerate(zip(ordered, centres)):
        label = block.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        num = str(i+1)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                      fill=color, outline=(255,255,255), width=2)
        tb = draw.textbbox((0,0), num, font=font_num)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        draw.text((cx-tw//2, cy-th//2), num, fill=(255,255,255), font=font_num)

        # Label court en dessous du bloc
        short = SHORT.get(label, label[:5])
        bx, by, bw, bh = block["bbox"]
        ltb = draw.textbbox((0,0), short, font=font_lbl)
        lw = ltb[2]-ltb[0]
        lx = max(0, bx+bw//2-lw//2)
        ly = min(h-15, by+bh+2)
        draw.rectangle([lx-2, ly-1, lx+lw+2, ly+ltb[3]+1], fill=color)
        draw.text((lx, ly), short, fill=(255,255,255), font=font_lbl)

    result = np.array(img)
    # Info barre
    leg_h = 40
    leg = np.full((leg_h, result.shape[1], 3), 245, dtype=np.uint8)
    pleg = Image.fromarray(leg)
    dleg = ImageDraw.Draw(pleg)
    fleg = _font(14)
    dleg.text((10, 12),
              f"{len(ordered)} blocs ordonnés  |  Détection colonnes par gap X  |  Flèches = séquence de lecture",
              fill=(50,50,50), font=fleg)
    result = np.vstack([result, np.array(pleg)])

    return add_title_bar(result, "Reading order: column detection + sort", 5, total,
                          bg_color=(60, 40, 20))


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 6 — OCR
# ═══════════════════════════════════════════════════════════════════

def viz_step6_ocr(page_rgb: np.ndarray,
                   text_blocks: list[dict], total: int) -> np.ndarray:
    print(f"  Étape 6: OCR ({len(text_blocks)} blocs)...")
    img = Image.fromarray(page_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = page_rgb.shape[:2]
    font_box  = _font(max(9, h // 100))
    font_text = _font(max(8, h // 110))

    for block in text_blocks:
        label = block.get("label", "unknown")
        text  = block.get("text", "").strip()
        color = PALETTE.get(label, PALETTE["unknown"])
        bx, by, bw, bh = block["bbox"]

        draw.rectangle([bx, by, bx+bw, by+bh],
                        outline=color, width=2, fill=color+(12,))

        if not text:
            # Pas de texte → marquer vide
            draw.text((bx+4, by+4), "∅", fill=(180,180,180), font=font_box)
            continue

        # Afficher les 2 premières lignes du texte OCR
        preview = text.replace("\n", " ").strip()
        max_chars = max(10, bw // 7)
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "…"

        # Fond blanc semi-transparent pour lisibilité
        tb = draw.textbbox((0,0), preview, font=font_text)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        tx = bx + 3
        ty = by + 3
        if tw < bw - 4 and th < bh - 4:
            draw.rectangle([tx-1, ty-1, tx+tw+1, ty+th+1],
                            fill=(255,255,255,200))
            draw.text((tx, ty), preview, fill=(20,20,20), font=font_text)

    result = np.array(img)
    # Stats OCR
    n_with_text = sum(1 for b in text_blocks if b.get("text","").strip())
    n_chars = sum(len(b.get("text","")) for b in text_blocks)
    leg_h = 40
    leg = np.full((leg_h, result.shape[1], 3), 245, dtype=np.uint8)
    pleg = Image.fromarray(leg)
    dleg = ImageDraw.Draw(pleg)
    fleg = _font(14)
    dleg.text((10, 12),
              f"{n_with_text}/{len(text_blocks)} blocs avec texte  |  {n_chars} caractères extraits  "
              f"|  PaddleOCR PP-OCRv5 + PPStructureV3",
              fill=(50,50,50), font=fleg)
    result = np.vstack([result, np.array(pleg)])

    return add_title_bar(result, "OCR: PaddleOCR PP-OCRv5 + PPStructureV3", 6, total,
                          bg_color=(20, 60, 60))


# ═══════════════════════════════════════════════════════════════════
# IMAGE FINALE — Toutes les étapes côte à côte
# ═══════════════════════════════════════════════════════════════════

def make_overview(steps_imgs: list[tuple[str, np.ndarray]],
                  pdf_name: str) -> np.ndarray:
    """Assemble toutes les étapes en une grille 2 × N."""
    print("  Assemblage de l'image finale...")

    n = len(steps_imgs)
    cols = 3
    rows = (n + cols - 1) // cols

    # Largeur cible par cellule
    cell_w = 700
    cell_imgs = []
    for name, img in steps_imgs:
        resized = resize_to_width(img, cell_w)
        cell_imgs.append(resized)

    # Hauteur max par ligne
    max_h = max(img.shape[0] for img in cell_imgs)

    # Padding pour égaliser les hauteurs
    padded = []
    for img in cell_imgs:
        h, w = img.shape[:2]
        if h < max_h:
            pad = np.full((max_h - h, w, 3), 245, dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    # Remplir avec des images vides si nécessaire
    while len(padded) % cols != 0:
        empty = np.full((max_h, cell_w, 3), 245, dtype=np.uint8)
        padded.append(empty)

    # Assembler en grille
    sep_v = np.full((max_h, 6, 3), 180, dtype=np.uint8)
    rows_imgs = []
    for r in range(rows):
        row_parts = []
        for c in range(cols):
            idx = r * cols + c
            row_parts.append(padded[idx])
            if c < cols - 1:
                row_parts.append(sep_v)
        rows_imgs.append(np.hstack(row_parts))

    sep_h = np.full((6, rows_imgs[0].shape[1], 3), 180, dtype=np.uint8)
    grid_parts = []
    for i, row in enumerate(rows_imgs):
        grid_parts.append(row)
        if i < len(rows_imgs) - 1:
            grid_parts.append(sep_h)
    grid = np.vstack(grid_parts)

    # Titre global
    title_h = 70
    title_bar = np.full((title_h, grid.shape[1], 3), (15, 25, 50), dtype=np.uint8)
    pil = Image.fromarray(title_bar)
    draw = ImageDraw.Draw(pil)
    font_big  = _font(28)
    font_small = _font(16)
    main_title = "StatParse — Pipeline de Traitement Documentaire"
    sub_title  = f"PDF : {pdf_name}   |   Étapes : {n}"
    tb = draw.textbbox((0,0), main_title, font=font_big)
    tx = grid.shape[1]//2 - (tb[2]-tb[0])//2
    draw.text((tx, 8), main_title, fill=(255,255,255), font=font_big)
    tb2 = draw.textbbox((0,0), sub_title, font=font_small)
    tx2 = grid.shape[1]//2 - (tb2[2]-tb2[0])//2
    draw.text((tx2, 44), sub_title, fill=(150,180,255), font=font_small)
    title_bar = np.array(pil)

    return np.vstack([title_bar, grid])


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",     type=Path, required=True, help="PDF à visualiser")
    p.add_argument("--out-dir", type=Path, default=Path("./viz/pipeline"))
    p.add_argument("--dpi",     type=int,  default=150)
    p.add_argument("--no-ocr",  action="store_true", help="Skip étape OCR (rapide)")
    p.add_argument("--n", type=int, default=1,
               help="Nombre de PDFs aléatoires à traiter")
    args = p.parse_args()

    import random

    if not args.pdf.exists():
        print(f"[ERREUR] Chemin introuvable : {args.pdf}")
        sys.exit(1)

# Si c'est un dossier → on prend n PDFs aléatoires
    if args.pdf.is_dir():
        all_pdfs = list(args.pdf.glob("*.pdf"))

        if len(all_pdfs) == 0:
            print("[ERREUR] Aucun PDF trouvé dans le dossier")
            sys.exit(1)

        n = min(args.n, len(all_pdfs))
        pdf_list = random.sample(all_pdfs, n)

# Si c'est un fichier → liste de taille 1
    else:
        pdf_list = [args.pdf]


    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 5 if args.no_ocr else 6

    print(f"\nStatParse Pipeline Visualizer")
    print(f"  PDF    : {args.pdf.name}")
    print(f"  DPI    : {args.dpi}")
    print(f"  Étapes : {total}")
    print(f"  Sortie : {args.out_dir}/\n")

    # ── Charger le PDF ────────────────────────────────────────────
    for pdf_path in pdf_list:

        print(f"\n==============================")
        print(f"Traitement : {pdf_path.name}")
        print(f"==============================")

        out_dir = args.out_dir / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        total = 5 if args.no_ocr else 6

        # Charger
        print("Chargement du PDF...")
        page_rgb = pdf_to_rgb(pdf_path, args.dpi)

        # Pipeline
        print("Pipeline en cours...")
        binary  = preprocess(page_rgb)
        blocks  = segment(binary)
        labeled = classify(blocks, image_shape=binary.shape)
        ordered = order_blocks(labeled, image_shape=binary.shape)

        text_blocks = None
        if not args.no_ocr:
            from statparse.ocr import recognize_text
            text_blocks = recognize_text(ordered, page_rgb)

        # Visuels
        print("Génération des visuels...")
        steps = []

        v1 = viz_step1_original(page_rgb, total)
        steps.append(("original", v1))
        Image.fromarray(v1).save(out_dir / "step1_original.png")

        v2 = viz_step2_binarisation(page_rgb, binary, total)
        steps.append(("binarisation", v2))
        Image.fromarray(v2).save(out_dir / "step2_binarisation.png")

        v3 = viz_step3_segmentation(page_rgb, blocks, total)
        steps.append(("segmentation", v3))
        Image.fromarray(v3).save(out_dir / "step3_segmentation.png")

        v4 = viz_step4_classification(page_rgb, labeled, total)
        steps.append(("classification", v4))
        Image.fromarray(v4).save(out_dir / "step4_classification.png")

        v5 = viz_step5_reading_order(page_rgb, ordered, total)
        steps.append(("reading_order", v5))
        Image.fromarray(v5).save(out_dir / "step5_reading_order.png")

        if not args.no_ocr and text_blocks is not None:
            v6 = viz_step6_ocr(page_rgb, text_blocks, total)
            steps.append(("ocr", v6))
            Image.fromarray(v6).save(out_dir / "step6_ocr.png")

        overview = make_overview(steps, pdf_path.stem[:60])
        Image.fromarray(overview).save(out_dir / "pipeline_overview.png")

        print(f"✅ Terminé pour {pdf_path.name}")
    Image.fromarray(overview).save(args.out_dir / "pipeline_overview.png")

    print(f"\n✅ Terminé. Images dans {args.out_dir}/")
    print(f"   → pipeline_overview.png  (vue complète)")
    for name, _ in steps:
        print(f"   → step*_{name}.png")


if __name__ == "__main__":
    main()