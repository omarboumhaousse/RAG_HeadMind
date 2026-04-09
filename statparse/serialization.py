"""Step 6: Markdown serialization — convert labeled text blocks to Markdown.

Maps OmniDocBench-compatible labels to Markdown syntax.
No external dependencies. Deterministic.

Labels handled:
    title              → # Heading
    text_block         → paragraph
    table              → pass-through (already Markdown from PPStructureV3)
    table_caption      → *italic caption*
    table_footnote     → *italic footnote*
    figure_caption     → *italic caption*
    figure_footnote    → *italic footnote*
    equation_isolated  → $$ display math $$
    equation_caption   → plain text (formula number)
    code_txt           → ```code block```
    header             → skipped (not useful for RAG)
    footer             → skipped
    page_number        → skipped
    page_footnote      → small footnote
    abandon            → skipped
    figure             → skipped (no text)
"""


def to_markdown(blocks: list[dict]) -> str:
    """Convert ordered, OCR'd blocks into a Markdown string.

    Each block must have 'label' and 'text' keys.

    Returns:
        Markdown-formatted string for the full page.
    """
    lines = []

    # Labels to skip entirely
    SKIP_LABELS = {
        "figure", "header", "footer", "page_number", "abandon",
        # legacy labels (in case old classification is used)
        "header", "footer",
    }

    for block in blocks:
        text  = block.get("text", "").strip()
        label = block.get("label", "text_block")

        if label in SKIP_LABELS:
            continue

        if not text:
            continue

        # ── Title ───────────────────────────────────────────────────
        if label == "title":
            level  = block.get("heading_level", 1)
            prefix = "#" * max(1, min(level, 6))
            lines.append(f"{prefix} {text}")

        # ── Body text ────────────────────────────────────────────────
        elif label in ("text_block", "paragraph"):
            lines.append(text)

        # ── Table ────────────────────────────────────────────────────
        # PPStructureV3 already returns Markdown; pass through as-is.
        elif label == "table":
            lines.append(text)

        # ── Captions and footnotes ───────────────────────────────────
        elif label in ("table_caption", "figure_caption",
                       "table_footnote", "figure_footnote"):
            lines.append(f"*{text}*")

        # ── Equations (display / block-level) ────────────────────────
        elif label in ("equation_isolated", "equation"):
            lines.append(f"$$\n{text}\n$$")

        # ── Equation number / caption ────────────────────────────────
        elif label == "equation_caption":
            lines.append(text)

        # ── Code block ───────────────────────────────────────────────
        elif label == "code_txt":
            lines.append(f"```\n{text}\n```")

        # ── Page footnote ────────────────────────────────────────────
        elif label == "page_footnote":
            lines.append(f"[^note]: {text}")

        # ── Caption (legacy label) ───────────────────────────────────
        elif label == "caption":
            lines.append(f"*{text}*")

        # ── Fallback: treat as body text ─────────────────────────────
        else:
            lines.append(text)

    return "\n\n".join(lines)