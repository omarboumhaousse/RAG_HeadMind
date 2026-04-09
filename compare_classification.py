"""
compare_classification.py
--------------------------
Side-by-side comparison: OmniDocBench ground truth vs our pipeline.

For each PDF in pdfs_selected/:
  Image 1 — Classification: GT (left) vs Pipeline (right)
  Image 2 — Reading order:  GT (left) vs Pipeline (right)

Also produces comparison_report.md.

Usage:
    python compare_classification.py
    python compare_classification.py --pdf ./pdfs_selected/foo.pdf
    python compare_classification.py --no-reading-order
    python compare_classification.py --n 5
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from statparse.preprocessing import preprocess
from statparse.segmentation import segment
from statparse.classification import classify
from statparse.reading_order import order_blocks

# ── Defaults ───────────────────────────────────────────────────────
GT_JSON    = Path("./OmniDocBench.json")
IMAGES_DIR = Path("./images")
PDF_DIR    = Path("./pdfs_selected")
OUTPUT_DIR = Path("./compare_viz")
DPI        = 150

# ── OmniDocBench colour palette (RGB) ─────────────────────────────
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
    "equation_caption":  (150,  50, 200),
    "header":            (  0, 180, 180),
    "footer":            (180, 140,   0),
    "page_number":       (130, 130, 130),
    "page_footnote":     (160, 160, 100),
    "abandon":           (200, 200, 200),
    "code_txt":          ( 50,  50,  50),
    "_caption_candidate":(255,   0, 255),
    "unknown":           (150, 150, 150),
}
SHORT = {
    "title": "TITLE", "text_block": "TEXT", "figure": "FIG",
    "figure_caption": "FIG-CAP", "figure_footnote": "FIG-FN",
    "table": "TABLE", "table_caption": "TBL-CAP", "table_footnote": "TBL-FN",
    "equation_isolated": "EQ", "equation_caption": "EQ-CAP",
    "header": "HDR", "footer": "FTR", "page_number": "PG#",
    "page_footnote": "PG-FN", "abandon": "ABAND", "code_txt": "CODE",
    "_caption_candidate": "CAP?", "unknown": "?",
}
BLOCK_CATS = {
    "title", "text_block", "figure", "figure_caption", "figure_footnote",
    "table", "table_caption", "table_footnote", "equation_isolated",
    "equation_caption", "header", "footer", "page_number", "page_footnote",
    "abandon", "code_txt",
}
RO_COLORS = [
    (220,50,50),(50,150,220),(34,160,34),(220,130,0),
    (150,0,200),(0,180,180),(180,140,0),(220,20,140),
]


# ═══════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════

def load_gt(gt_json: Path) -> dict:
    with open(gt_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {Path(s["page_info"]["image_path"]).stem: s for s in samples}


def poly_to_bbox(poly):
    xs = [poly[i] for i in range(0, 8, 2)]
    ys = [poly[i] for i in range(1, 8, 2)]
    x, y = int(min(xs)), int(min(ys))
    return (x, y, int(max(xs))-x, int(max(ys))-y)


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


def scale_boxes(boxes, gt_w, gt_h, img_w, img_h):
    sx, sy = img_w/gt_w, img_h/gt_h
    return [{**b, "bbox": (int(b["bbox"][0]*sx), int(b["bbox"][1]*sy),
                            int(b["bbox"][2]*sx), int(b["bbox"][3]*sy))}
            for b in boxes]


# ═══════════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════════

def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()


def draw_classification_boxes(base_rgb, boxes, title=""):
    img  = Image.fromarray(base_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h    = base_rgb.shape[0]
    fs   = max(11, h//85)
    font = _font(fs)
    pad  = 3

    for box in boxes:
        label = box.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        text  = SHORT.get(label, label[:7].upper())
        order = box.get("order")
        if order is not None:
            text = f"{order}:{text}"
        bx, by, bw, bh = box["bbox"]
        draw.rectangle([bx, by, bx+bw, by+bh], outline=color, width=2,
                        fill=color+(18,))
        tb = draw.textbbox((0,0), text, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        tx = bx + pad
        ty = by + pad if bh >= th + 2*pad else max(0, by - th - 2*pad)
        draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=color)
        draw.text((tx, ty), text, fill=(255,255,255), font=font)

    if title:
        bf = _font(max(14, h//65))
        tb = draw.textbbox((0,0), title, font=bf)
        draw.rectangle([0,0,tb[2]+12,tb[3]-tb[1]+10], fill=(30,30,30,210))
        draw.text((6,5), title, fill=(255,255,255), font=bf)
    return np.array(img)


def draw_reading_order(base_rgb, ordered_boxes, title=""):
    """Draw numbered circles + arrows showing reading order."""
    img  = Image.fromarray(base_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    H, W = base_rgb.shape[:2]

    font_num = _font(max(14, H//70))
    font_lbl = _font(max(10, H//100))

    # Compute block centres
    centres = [(b["bbox"][0]+b["bbox"][2]//2, b["bbox"][1]+b["bbox"][3]//2)
               for b in ordered_boxes]

    # Light fill for each block
    for box in ordered_boxes:
        label = box.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        bx, by, bw, bh = box["bbox"]
        draw.rectangle([bx,by,bx+bw,by+bh], outline=color, width=2,
                        fill=color+(15,))

    # Arrows N → N+1
    for i in range(len(centres)-1):
        x1, y1 = centres[i]
        x2, y2 = centres[i+1]
        color   = RO_COLORS[i % len(RO_COLORS)]
        draw.line([(x1,y1),(x2,y2)], fill=color+(180,), width=3)
        # Arrowhead
        ang = np.arctan2(y2-y1, x2-x1)
        al  = max(12, H//100)
        sp  = 0.4
        pts = [(x2,y2),
               (int(x2 - al*np.cos(ang-sp)), int(y2 - al*np.sin(ang-sp))),
               (int(x2 - al*np.cos(ang+sp)), int(y2 - al*np.sin(ang+sp)))]
        draw.polygon(pts, fill=color+(200,))

    # Numbered circles + labels
    r = max(14, H//90)
    for i, (box, (cx, cy)) in enumerate(zip(ordered_boxes, centres)):
        label = box.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        num   = str(i+1)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                      fill=color, outline=(255,255,255), width=2)
        tb = draw.textbbox((0,0), num, font=font_num)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        draw.text((cx-tw//2, cy-th//2), num, fill=(255,255,255), font=font_num)

        # Short label below block
        short  = SHORT.get(label, label[:6])
        bx, by, bw, bh = box["bbox"]
        ltb    = draw.textbbox((0,0), short, font=font_lbl)
        lw     = ltb[2]-ltb[0]
        lx     = max(0, bx + bw//2 - lw//2)
        ly     = min(H-15, by+bh+2)
        draw.rectangle([lx-2,ly-1,lx+lw+2,ly+ltb[3]+1], fill=color)
        draw.text((lx, ly), short, fill=(255,255,255), font=font_lbl)

    if title:
        bf = _font(max(14, H//65))
        tb = draw.textbbox((0,0), title, font=bf)
        draw.rectangle([0,0,tb[2]+12,tb[3]-tb[1]+10], fill=(30,30,30,210))
        draw.text((6,5), title, fill=(255,255,255), font=bf)
    return np.array(img)


def add_legend(image, labels):
    h, w  = image.shape[:2]
    leg   = np.full((52, w, 3), 245, dtype=np.uint8)
    pil   = Image.fromarray(leg)
    draw  = ImageDraw.Draw(pil)
    font  = _font(13)
    x     = 8
    for lbl in labels:
        if lbl.startswith("_"): continue
        color = PALETTE.get(lbl, PALETTE["unknown"])
        text  = SHORT.get(lbl, lbl[:7])
        tb    = draw.textbbox((0,0), text, font=font)
        tw    = tb[2]-tb[0]
        draw.rectangle([x,14,x+14,38], fill=color)
        draw.text((x+18,17), text, fill=(30,30,30), font=font)
        x += tw + 34
        if x > w - 50: break
    return np.vstack([image, np.array(pil)])


def side_by_side(left, right):
    hl, hr = left.shape[0], right.shape[0]
    if hl != hr:
        diff = abs(hl-hr)
        pad  = np.full((diff, left.shape[1] if hl<hr else right.shape[1], 3),
                        245, dtype=np.uint8)
        if hl < hr: left  = np.vstack([left,  pad])
        else:       right = np.vstack([right, pad])
    sep = np.full((left.shape[0], 6, 3), 180, dtype=np.uint8)
    return np.hstack([left, sep, right])


# ═══════════════════════════════════════════════════════════════════
# Per-PDF processing
# ═══════════════════════════════════════════════════════════════════

def process_pdf(pdf_path, gt_index, out_dir, dpi, do_ro=True):
    stem = pdf_path.stem
    print(f"\n  {pdf_path.name}")

    # Match GT
    gt_sample = gt_index.get(stem)
    if gt_sample is None:
        for key in gt_index:
            if stem in key or key in stem:
                gt_sample = gt_index[key]; break
    if gt_sample is None:
        print(f"    [WARN] No GT for {stem}"); return None

    pi   = gt_sample["page_info"]
    gt_w, gt_h = pi["width"], pi["height"]

    # GT image
    gt_img_path = IMAGES_DIR / Path(pi["image_path"]).name
    gt_rgb = (np.array(Image.open(gt_img_path).convert("RGB"))
              if gt_img_path.exists()
              else np.full((gt_h, gt_w, 3), 245, dtype=np.uint8))

    # GT boxes
    gt_boxes = []
    for det in gt_sample.get("layout_dets", []):
        cat = det.get("category_type", "unknown")
        if cat not in BLOCK_CATS or det.get("ignore", False): continue
        gt_boxes.append({"bbox": poly_to_bbox(det["poly"]),
                          "label": cat, "order": det.get("order")})

    # Pipeline
    page_rgb = pdf_to_rgb(pdf_path, dpi)
    binary   = preprocess(page_rgb)
    blocks   = segment(binary)
    labeled  = classify(blocks, image_shape=binary.shape)
    ordered  = order_blocks(labeled, image_shape=binary.shape)

    img_h, img_w    = page_rgb.shape[:2]
    gt_boxes_scaled = scale_boxes(gt_boxes, gt_w, gt_h, img_w, img_h)
    gt_rgb_rs       = np.array(Image.fromarray(gt_rgb).resize((img_w,img_h),
                                                               Image.LANCZOS))

    gt_counts   = Counter(b["label"] for b in gt_boxes)
    pred_counts = Counter(b.get("label","unknown") for b in ordered)
    all_labels  = sorted(set(gt_counts)|set(pred_counts))

    # ── Image 1: Classification ────────────────────────────────────
    left_cls  = draw_classification_boxes(gt_rgb_rs, gt_boxes_scaled,
                                           "GROUND TRUTH (OmniDocBench)")
    right_cls = draw_classification_boxes(page_rgb, ordered,
                                           "PIPELINE PREDICTION")
    cls_img   = add_legend(side_by_side(left_cls, right_cls), all_labels)
    cls_path  = out_dir / f"{stem}_compare.png"
    Image.fromarray(cls_img).save(cls_path)
    print(f"    Classification → {cls_path.name}")

    ro_path = None
    # ── Image 2: Reading order ─────────────────────────────────────
    if do_ro:
        # GT reading order
        gt_ordered = [b for b in gt_boxes_scaled if b.get("order") is not None]
        gt_ordered.sort(key=lambda b: b["order"])
        if not gt_ordered:
            gt_ordered = sorted(gt_boxes_scaled, key=lambda b: b["bbox"][1])
            for i, b in enumerate(gt_ordered): b["order"] = i+1

        left_ro  = draw_reading_order(gt_rgb_rs, gt_ordered,
                                       "READING ORDER — Ground Truth")
        right_ro = draw_reading_order(page_rgb, ordered,
                                       "READING ORDER — Pipeline")
        ro_img   = add_legend(side_by_side(left_ro, right_ro), all_labels)
        ro_path  = out_dir / f"{stem}_reading_order.png"
        Image.fromarray(ro_img).save(ro_path)
        print(f"    Reading order  → {ro_path.name}")

    return {
        "stem": stem,
        "gt_counts": dict(gt_counts),
        "pred_counts": dict(pred_counts),
        "gt_n_blocks": len(gt_boxes),
        "pred_n_blocks": len(ordered),
        "cls_path": str(cls_path),
        "ro_path":  str(ro_path) if ro_path else None,
    }


# ═══════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════

def write_report(reports, out_dir, do_ro):
    global_gt, global_pred = Counter(), Counter()
    for r in reports:
        global_gt.update(r["gt_counts"])
        global_pred.update(r["pred_counts"])
    all_cats = sorted(set(global_gt)|set(global_pred))

    lines = [
        "# Classification Comparison Report", "",
        "Comparison between **OmniDocBench ground truth** and **our pipeline**.", "",
        f"Pages evaluated: **{len(reports)}**", "", "---", "",
        "## Global category counts", "",
        "| Category | GT total | Pred total | Diff |",
        "|----------|----------|------------|------|",
    ]
    for cat in all_cats:
        gt_n, pred_n = global_gt.get(cat,0), global_pred.get(cat,0)
        diff = pred_n - gt_n
        lines.append(f"| `{cat}` | {gt_n} | {pred_n} | "
                      f"{'+' if diff>0 else ''}{diff} |")
    lines += ["", "---", "", "## Per-page breakdown", ""]

    for r in reports:
        page_cats = sorted(set(r["gt_counts"])|set(r["pred_counts"]))
        lines += [f"### {r['stem']}", "",
                  f"- GT: **{r['gt_n_blocks']}** blocks  |  "
                  f"Pipeline: **{r['pred_n_blocks']}** blocks", "",
                  "| Category | GT | Pred | Diff |",
                  "|----------|----|------|------|"]
        weaknesses = []
        for cat in page_cats:
            gt_n, pred_n = r["gt_counts"].get(cat,0), r["pred_counts"].get(cat,0)
            diff = pred_n - gt_n
            flag = " ⚠️" if abs(diff)>1 else ""
            lines.append(f"| `{cat}` | {gt_n} | {pred_n} | "
                          f"{'+' if diff>0 else ''}{diff}{flag} |")
            if abs(diff)>1:
                weaknesses.append(f"`{cat}`: GT={gt_n}, pred={pred_n}")
        if weaknesses:
            lines += ["", "**Weaknesses detected:**", ""]
            for w in weaknesses: lines.append(f"- {w}")
        lines += ["", f"![compare]({Path(r['cls_path']).name})"]
        if do_ro and r.get("ro_path"):
            lines += [f"![reading_order]({Path(r['ro_path']).name})"]
        lines += ["", "---", ""]

    # Summary
    lines += ["## Summary: categories with largest discrepancy", "",
               "| Category | |GT - Pred| |", "|----------|------------|"]
    for cat, d in sorted(
            [(c, abs(global_gt.get(c,0)-global_pred.get(c,0))) for c in all_cats],
            key=lambda x: -x[1])[:10]:
        if d == 0: continue
        lines.append(f"| `{cat}` | {d} "
                      f"(GT={global_gt.get(cat,0)}, pred={global_pred.get(cat,0)}) |")

    md = out_dir / "comparison_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report → {md}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    global IMAGES_DIR   
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--gt",     type=Path, default=GT_JSON)
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--dpi",    type=int,  default=DPI)
    p.add_argument("--n",      type=int,  default=None)
    p.add_argument("--pdf",    type=Path, default=None)
    p.add_argument("--no-reading-order", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR = args.images

    print("Loading OmniDocBench GT...")
    gt_index = load_gt(args.gt)
    print(f"  {len(gt_index)} pages indexed.")

    pdfs = [args.pdf] if args.pdf else sorted(args.pdf_dir.glob("*.pdf"))
    if args.n: pdfs = pdfs[:args.n]
    if not pdfs:
        print(f"No PDFs in {args.pdf_dir}"); sys.exit(1)

    do_ro = not args.no_reading_order
    print(f"\nProcessing {len(pdfs)} PDF(s){'  (+reading order)' if do_ro else ''}...")

    reports = []
    for pdf in pdfs:
        try:
            r = process_pdf(pdf, gt_index, args.out_dir, args.dpi, do_ro)
            if r: reports.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    if reports:
        write_report(reports, args.out_dir, do_ro)
    print(f"\nDone. {len(reports)} pages → {args.out_dir}/")


if __name__ == "__main__":
    main()