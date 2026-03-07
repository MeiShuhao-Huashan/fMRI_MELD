#!/usr/bin/env python3
"""
Assemble the final *revision Figure 3* into a single 180mm × 225mm PDF canvas.

User layout request (2026-02-11 / updated):
  - Panel A: paper/revision/figure2/panelC_deploy_gate_scope38.pdf
  - Panel B: paper/revision/figure2/panelF_subject_endpoints_scope38.pdf
  - Panel C: paper/revision/figure2/panelG_complementarity_scope38.pdf
  - Panel D: paper/revision/figure3/panelE_intermediate_vs_difficult_scope38.pdf
  - Panel E: paper/revision/figure3/panelB_complementarity_scope38.pdf
  - Panel F: paper/revision/figure3/panelC_seeg_scope38.pdf

Output:
  - paper/revision/figure3/figure3_revision_composite_180x225mm.pdf

Notes
  - This preserves vectors by merging PDF pages (no rasterization).
  - The layout is only meant to fix *relative positions/sizes*; fine edits can be
    done later in Illustrator/PowerPoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from pypdf import PdfReader, PdfWriter, Transformation

PT_PER_INCH = 72.0
MM_PER_INCH = 25.4


def _mm_to_pt(mm: float) -> float:
    return float(mm) * PT_PER_INCH / MM_PER_INCH


def _place_panel(
    dest_page,
    *,
    src_pdf: Path,
    box_mm: Tuple[float, float, float, float],
) -> None:
    """
    Place the first page of `src_pdf` into `dest_page` within `box_mm` = (x,y,w,h).
    Keeps aspect ratio and centers the panel inside the box.
    """

    reader = PdfReader(str(src_pdf))
    if not reader.pages:
        raise ValueError(f"Empty PDF: {src_pdf}")
    src = reader.pages[0]

    src_w = float(src.mediabox.width)
    src_h = float(src.mediabox.height)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid mediabox for {src_pdf}: {src_w}×{src_h}")

    x_mm, y_mm, w_mm, h_mm = (float(v) for v in box_mm)
    target_w = _mm_to_pt(w_mm)
    target_h = _mm_to_pt(h_mm)

    scale = min(target_w / src_w, target_h / src_h)
    scaled_w = src_w * scale
    scaled_h = src_h * scale

    tx = _mm_to_pt(x_mm) + (target_w - scaled_w) / 2.0
    ty = _mm_to_pt(y_mm) + (target_h - scaled_h) / 2.0

    dest_page.merge_transformed_page(src, Transformation((scale, 0.0, 0.0, scale, tx, ty)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_pdf",
        type=Path,
        default=Path("paper/revision/figure3/figure3_revision_composite_180x225mm.pdf"),
        help="Output PDF path.",
    )
    ap.add_argument("--width_mm", type=float, default=180.0, help="Canvas width (mm).")
    ap.add_argument("--height_mm", type=float, default=225.0, help="Canvas height (mm).")
    args = ap.parse_args()

    # Inputs (fixed per user request).
    panel_a = Path("paper/revision/figure2/panelC_deploy_gate_scope38.pdf")
    panel_b = Path("paper/revision/figure2/panelF_subject_endpoints_scope38.pdf")
    panel_c = Path("paper/revision/figure2/panelG_complementarity_scope38.pdf")
    panel_d = Path("paper/revision/figure3/panelE_intermediate_vs_difficult_scope38.pdf")
    panel_e = Path("paper/revision/figure3/panelB_complementarity_scope38.pdf")
    panel_f = Path("paper/revision/figure3/panelC_seeg_scope38.pdf")

    for p in [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Canvas: 180×225 mm.
    canvas_w_pt = _mm_to_pt(float(args.width_mm))
    canvas_h_pt = _mm_to_pt(float(args.height_mm))

    # Layout boxes in mm: (x, y, w, h), origin = bottom-left.
    # Symmetric 3×2 grid so panel numbering is sequential (A→F; left-to-right, top-to-bottom).
    margin_l = 8.0
    margin_r = 8.0
    margin_b = 8.0
    margin_t = 8.0
    hgap = 6.0
    vgap = 6.0

    col_w = (float(args.width_mm) - margin_l - margin_r - hgap) / 2.0
    row_h = (float(args.height_mm) - margin_b - margin_t - 2.0 * vgap) / 3.0

    x_left = margin_l
    x_right = margin_l + col_w + hgap

    y_row3 = margin_b
    y_row2 = margin_b + row_h + vgap
    y_row1 = y_row2 + row_h + vgap

    boxes = {
        "A": (x_left, y_row1, col_w, row_h),
        "B": (x_right, y_row1, col_w, row_h),
        "C": (x_left, y_row2, col_w, row_h),
        "D": (x_right, y_row2, col_w, row_h),
        "E": (x_left, y_row3, col_w, row_h),
        "F": (x_right, y_row3, col_w, row_h),
    }

    writer = PdfWriter()
    dest_page = writer.add_blank_page(width=canvas_w_pt, height=canvas_h_pt)

    _place_panel(dest_page, src_pdf=panel_a, box_mm=boxes["A"])
    _place_panel(dest_page, src_pdf=panel_b, box_mm=boxes["B"])
    _place_panel(dest_page, src_pdf=panel_c, box_mm=boxes["C"])
    _place_panel(dest_page, src_pdf=panel_d, box_mm=boxes["D"])
    _place_panel(dest_page, src_pdf=panel_e, box_mm=boxes["E"])
    _place_panel(dest_page, src_pdf=panel_f, box_mm=boxes["F"])

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pdf.open("wb") as f:
        writer.write(f)


if __name__ == "__main__":
    main()
