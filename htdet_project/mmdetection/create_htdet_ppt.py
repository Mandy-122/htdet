"""
HTDet PowerPoint Presentation Generator
Produces a 20-slide deck covering architecture, PTQ, C-sim implementation,
bugs, OpenMP, validation results, and FPGA hardware benefits.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ─────────────────────────── Colour palette ───────────────────────────
DARK_BLUE   = RGBColor(0x0D, 0x2B, 0x55)   # slide background / title bar
MID_BLUE    = RGBColor(0x1A, 0x55, 0x8A)   # accent bars
LIGHT_BLUE  = RGBColor(0xD0, 0xE8, 0xF8)   # table header / highlight
TEAL        = RGBColor(0x00, 0x96, 0x88)   # bullet accent
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT   = RGBColor(0x1A, 0x1A, 0x2E)
GOLD        = RGBColor(0xF5, 0xA6, 0x23)   # key numbers
LIGHT_GRAY  = RGBColor(0xF4, 0xF6, 0xF8)   # table row alt
GREEN       = RGBColor(0x2E, 0x7D, 0x32)
RED         = RGBColor(0xC6, 0x28, 0x28)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # completely blank


# ═══════════════════════════ helpers ═══════════════════════════════════

def add_rect(slide, l, t, w, h, fill_rgb, border_rgb=None, border_pt=0):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if border_rgb:
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=18, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_para(tf, text, size=16, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, italic=False, space_before=0):
    p = tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return p


def slide_header(slide, title, subtitle=None):
    """Dark top bar with title."""
    add_rect(slide, 0, 0, 13.33, 1.15, DARK_BLUE)
    add_text(slide, title, 0.35, 0.12, 12.0, 0.7,
             size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, 0.35, 0.75, 12.0, 0.38,
                 size=14, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.LEFT)
    # thin teal accent line under header
    add_rect(slide, 0, 1.15, 13.33, 0.05, TEAL)


def bullet_box(slide, items, l, t, w, h,
               title=None, title_size=15, bullet_size=14,
               bullet_char="▸", indent_levels=None):
    """
    items: list of strings or (text, indent_level) tuples
    indent_levels: list of ints parallel to items (0=normal, 1=sub)
    """
    add_rect(slide, l, t, w, h, WHITE, MID_BLUE, 0.8)
    if title:
        add_rect(slide, l, t, w, 0.32, LIGHT_BLUE)
        add_text(slide, title, l+0.1, t+0.04, w-0.2, 0.28,
                 size=title_size, bold=True, color=DARK_BLUE)
        t += 0.34

    txb = slide.shapes.add_textbox(
        Inches(l+0.12), Inches(t), Inches(w-0.24), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = True

    for idx, item in enumerate(items):
        indent = 0
        if indent_levels:
            indent = indent_levels[idx]
        text = item

        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(2)
        p.level = indent

        prefix = "  " * indent + (bullet_char + "  " if indent == 0 else "   – ")
        run = p.add_run()
        run.text = prefix + text
        run.font.size  = Pt(bullet_size - indent * 1.5)
        run.font.color.rgb = DARK_TEXT
        run.font.bold  = False


def simple_table(slide, headers, rows, l, t, w,
                 header_color=MID_BLUE, alt_color=LIGHT_GRAY,
                 col_widths=None, font_size=12):
    ncols = len(headers)
    nrows = len(rows)
    row_h = 0.34
    total_h = (nrows + 1) * row_h
    if col_widths is None:
        col_widths = [w / ncols] * ncols

    # header row
    x = l
    for ci, hdr in enumerate(headers):
        add_rect(slide, x, t, col_widths[ci], row_h, header_color)
        add_text(slide, hdr, x+0.06, t+0.05, col_widths[ci]-0.1, row_h-0.08,
                 size=font_size, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        x += col_widths[ci]

    # data rows
    for ri, row in enumerate(rows):
        bg = alt_color if ri % 2 == 0 else WHITE
        x = l
        for ci, cell in enumerate(row):
            add_rect(slide, x, t+(ri+1)*row_h, col_widths[ci], row_h, bg,
                     LIGHT_BLUE, 0.3)
            color = DARK_TEXT
            bold  = False
            txt   = str(cell)
            if txt.startswith("**") and txt.endswith("**"):
                txt  = txt[2:-2]
                bold = True
                color = MID_BLUE
            add_text(slide, txt, x+0.06, t+(ri+1)*row_h+0.04,
                     col_widths[ci]-0.1, row_h-0.08,
                     size=font_size, bold=bold, color=color,
                     align=PP_ALIGN.CENTER)
            x += col_widths[ci]

    return total_h + row_h


def code_box(slide, code_lines, l, t, w, h, font_size=9):
    add_rect(slide, l, t, w, h, RGBColor(0x1E, 0x1E, 0x1E))
    txb = slide.shapes.add_textbox(
        Inches(l+0.12), Inches(t+0.1), Inches(w-0.24), Inches(h-0.15))
    tf = txb.text_frame
    tf.word_wrap = False
    for idx, line in enumerate(code_lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.name  = "Courier New"
        run.font.size  = Pt(font_size)
        run.font.color.rgb = RGBColor(0xD4, 0xD4, 0xD4)


def callout(slide, text, l, t, w, h, bg=GOLD, fg=DARK_TEXT, size=15):
    add_rect(slide, l, t, w, h, bg)
    add_text(slide, text, l+0.12, t+0.08, w-0.24, h-0.12,
             size=size, bold=True, color=fg, align=PP_ALIGN.CENTER, wrap=True)


# ═══════════════════════════ SLIDE 1 — Title ═══════════════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, DARK_BLUE)
add_rect(sl, 0, 2.9, 13.33, 0.07, TEAL)
add_rect(sl, 0, 4.55, 13.33, 0.07, TEAL)

add_text(sl, "HTDet", 0.6, 0.6, 12.0, 1.3,
         size=72, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
add_text(sl,
    "A Lightweight Hybrid CNN-Transformer Detector\nfor Underwater Object Detection with FPGA Deployment",
    0.6, 2.0, 12.0, 1.0,
    size=24, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
add_text(sl,
    "MobileViT-S Backbone  ·  FPN Neck  ·  RetinaNet Head\n"
    "Post-Training Quantization W8A32 / W8A8  ·  Vitis HLS C-Simulation",
    0.6, 3.1, 12.0, 0.8,
    size=17, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
add_text(sl,
    "Dataset: URPC 2018 Underwater  ·  4 Classes  ·  800 val images\n"
    "Result: mAP@50 = 0.755 (FP32) → 0.753 (W8A8, −0.3% only)",
    0.6, 4.75, 12.0, 0.8,
    size=16, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
add_text(sl, "demongod11  ·  2026", 0.6, 6.8, 12.0, 0.4,
         size=12, color=RGBColor(0x88, 0xAA, 0xCC), align=PP_ALIGN.CENTER)


# ═══════════════════════════ SLIDE 2 — Problem Statement ═══════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Problem Statement",
             "Why is underwater object detection uniquely challenging?")

bullet_box(sl, [
    "Low contrast & turbid water — organisms blend into backgrounds",
    "Extreme scale variation — tiny sea urchin (10 px) to large starfish (200 px)",
    "Class similarity — holothurian, echinus, scallop, starfish look alike",
    "Small dataset — only ~2,400 URPC training images",
    "Real-time FPGA deployment — must run in <100 ms at 640×640",
    "Memory budget — Xilinx ZU9EG has ~32 MB total BRAM",
], 0.4, 1.35, 8.0, 4.8,
   title="Detection Challenges", bullet_size=16)

# right panel — stats
add_rect(sl, 8.7, 1.35, 4.2, 4.8, WHITE, MID_BLUE, 0.8)
add_rect(sl, 8.7, 1.35, 4.2, 0.32, MID_BLUE)
add_text(sl, "Our Solution", 8.82, 1.38, 4.0, 0.28,
         size=15, bold=True, color=WHITE)

stats = [
    ("12.4 M", "Parameters"),
    ("198.9", "GFLOPs"),
    ("0.755", "mAP@50"),
    ("0.408", "mAP (0.5:0.95)"),
    ("40–60 ms", "FPGA latency (W8A8)"),
]
for i, (val, lbl) in enumerate(stats):
    y = 1.9 + i * 0.78
    add_text(sl, val, 8.85, y, 2.0, 0.42,
             size=26, bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
    add_text(sl, lbl, 10.85, y+0.06, 2.0, 0.36,
             size=13, color=DARK_TEXT)


# ═══════════════════════════ SLIDE 3 — Architecture Overview ═══════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Architecture Overview", "Three-stage detection pipeline")

# pipeline diagram boxes
pipeline = [
    ("Input Image\n3×640×640", 0.5,  MID_BLUE,  WHITE),
    ("MobileViT-S\nBackbone",  3.0,  DARK_BLUE, WHITE),
    ("FPN Neck",               5.5,  TEAL,      WHITE),
    ("RetinaNet\nHead",        8.0,  GREEN,     WHITE),
    ("Detections\n(cls, score, box)", 10.5, GOLD, DARK_TEXT),
]
for label, lx, bg, fg in pipeline:
    add_rect(sl, lx, 1.7, 2.35, 1.4, bg)
    add_text(sl, label, lx+0.1, 1.82, 2.15, 1.2,
             size=16, bold=True, color=fg, align=PP_ALIGN.CENTER)

# arrows
for lx in [2.85, 5.35, 7.85, 10.35]:
    add_text(sl, "→", lx, 2.1, 0.3, 0.6,
             size=28, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# feature map labels
fmaps = [
    (3.0,  "C1[64,160,160]\nC2[96,80,80]\nC3[128,40,40]\nC4[640,20,20]"),
    (5.5,  "P2[256,160,160]\nP3[256,80,80]\nP4[256,40,40]\nP5[256,20,20]\nP6[256,10,10]"),
    (8.0,  "cls scores\n+\nbox deltas\n(9 anchors/loc)"),
]
for lx, txt in fmaps:
    add_text(sl, txt, lx, 3.2, 2.35, 1.5,
             size=11, color=MID_BLUE, align=PP_ALIGN.CENTER, italic=True)

# component table
simple_table(sl,
    ["Component", "Type", "Params", "GFLOPs"],
    [
        ["MobileViT-S", "Hybrid CNN-Transformer", "5.58 M", "~70"],
        ["FPN Neck",    "Top-down pyramid",        "1.97 M", "~45"],
        ["RetinaNet Head", "Dual conv towers",    "4.86 M", "~84"],
        ["**Total**",   "—",                      "**12.41 M**", "**198.9**"],
    ],
    0.4, 4.85, 12.5,
    col_widths=[3.5, 3.5, 2.5, 2.5], font_size=13)


# ═══════════════════════════ SLIDE 4 — MobileViT-S Backbone ════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Backbone: MobileViT-S",
             "Hybrid CNN-Transformer — local patterns + global context")

# stage diagram left
stages = [
    ("Input 640×640×3", DARK_BLUE, WHITE),
    ("Stem Conv 3×3 s2",  MID_BLUE, WHITE),
    ("Stage 0–1: MBConv ×2  →  320×320",  MID_BLUE, WHITE),
    ("Stage 2: MBConv ×3  →  C1 [64, 160, 160]",  TEAL, WHITE),
    ("Stage 3: MBConv + MobileViT(d=144,L=2)  →  C2 [96, 80, 80]",  TEAL, WHITE),
    ("Stage 4: MBConv×2 + MobileViT(d=192,L=4)  →  C3 [128, 40, 40]",  TEAL, WHITE),
    ("Stage 5: MBConv×3 + MobileViT(d=240,L=3) + 1×1  →  C4 [640, 20, 20]", GREEN, WHITE),
]
for i, (lbl, bg, fg) in enumerate(stages):
    y = 1.35 + i * 0.76
    add_rect(sl, 0.35, y, 7.8, 0.62, bg)
    add_text(sl, lbl, 0.45, y+0.1, 7.6, 0.45,
             size=13, bold=(i == 0), color=fg)
    if i < len(stages)-1:
        add_text(sl, "↓", 3.85, y+0.62, 0.5, 0.14,
                 size=12, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# right panel
bullet_box(sl, [
    "5.58 M backbone params vs ResNet-50's 25 M",
    "ImageNet pretrained weights via TIMM",
    "MBConv: depthwise-separable, ~28× cheaper than full 3×3 conv",
    "MobileViT block: 2×2 patch unfolding → Transformer attention → fold back",
    "SiLU activation — smooth, non-monotonic, better gradients than ReLU",
    "Transformer only in stages 3–5 (smaller spatial maps = manageable cost)",
], 8.4, 1.35, 4.55, 5.7,
   title="Why MobileViT-S?", bullet_size=13)


# ═══════════════════════════ SLIDE 5 — MBConv & MobileViT Block ════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Building Blocks: MBConv and MobileViT Block",
             "Local feature extraction and global context capture")

# MBConv
add_rect(sl, 0.35, 1.35, 5.9, 5.6, WHITE, MID_BLUE, 0.8)
add_rect(sl, 0.35, 1.35, 5.9, 0.32, MID_BLUE)
add_text(sl, "MBConv — Inverted Bottleneck", 0.47, 1.38, 5.7, 0.28,
         size=15, bold=True, color=WHITE)

mbconv_steps = [
    ("Input [H, W, C]",             LIGHT_GRAY),
    ("1×1 Conv: C → 4C  [expand]",  RGBColor(0xBB,0xDE,0xFB)),
    ("BatchNorm + SiLU",            LIGHT_GRAY),
    ("3×3 DW Conv: 4C → 4C",       RGBColor(0xBB,0xDE,0xFB)),
    ("BatchNorm + SiLU",            LIGHT_GRAY),
    ("1×1 Conv: 4C → C  [project]", RGBColor(0xBB,0xDE,0xFB)),
    ("BatchNorm  +  Skip Add",      RGBColor(0xC8,0xE6,0xC9)),
    ("Output [H, W, C]",            LIGHT_GRAY),
]
for i, (lbl, bg) in enumerate(mbconv_steps):
    y = 1.82 + i * 0.56
    add_rect(sl, 0.6, y, 5.4, 0.44, bg, MID_BLUE, 0.3)
    add_text(sl, lbl, 0.7, y+0.08, 5.2, 0.3, size=13, color=DARK_TEXT)
    if i < len(mbconv_steps)-1:
        add_text(sl, "↓", 2.95, y+0.44, 0.4, 0.12,
                 size=10, color=MID_BLUE, align=PP_ALIGN.CENTER)

# MobileViT Block
add_rect(sl, 6.55, 1.35, 6.4, 5.6, WHITE, MID_BLUE, 0.8)
add_rect(sl, 6.55, 1.35, 6.4, 0.32, MID_BLUE)
add_text(sl, "MobileViT Block — Global Attention", 6.67, 1.38, 6.2, 0.28,
         size=15, bold=True, color=WHITE)

mvit_steps = [
    ("Input [H, W, C]",                      LIGHT_GRAY),
    ("3×3 Conv (local neighbourhood)",        RGBColor(0xBB,0xDE,0xFB)),
    ("1×1 Conv → d channels",                RGBColor(0xBB,0xDE,0xFB)),
    ("Unfold to 2×2 patches → P=4 views",    RGBColor(0xFF,0xE0,0xB2)),
    ("Transformer × L  (self-attention)",     RGBColor(0xFF,0xCC,0x80)),
    ("Fold back → [H, W, d]",               RGBColor(0xFF,0xE0,0xB2)),
    ("1×1 Conv → C channels",               RGBColor(0xBB,0xDE,0xFB)),
    ("3×3 Fusion Conv (concat → C)",        RGBColor(0xC8,0xE6,0xC9)),
    ("Output [H, W, C]",                    LIGHT_GRAY),
]
for i, (lbl, bg) in enumerate(mvit_steps):
    y = 1.82 + i * 0.49
    add_rect(sl, 6.8, y, 5.9, 0.38, bg, MID_BLUE, 0.3)
    add_text(sl, lbl, 6.9, y+0.07, 5.7, 0.27, size=12, color=DARK_TEXT)
    if i < len(mvit_steps)-1:
        add_text(sl, "↓", 9.6, y+0.38, 0.4, 0.11,
                 size=10, color=MID_BLUE, align=PP_ALIGN.CENTER)


# ═══════════════════════════ SLIDE 6 — FPN Neck ════════════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Neck: Feature Pyramid Network (FPN)",
             "Merges multi-scale features — every level has detail AND semantics")

# FPN diagram
fpn_levels = [
    ("C4 [640, 20, 20]",  "P5 [256, 20×20]",  0.75, GREEN,   TEAL),
    ("C3 [128, 40, 40]",  "P4 [256, 40×40]",  1.55, MID_BLUE, TEAL),
    ("C2 [ 96, 80, 80]",  "P3 [256, 80×80]",  2.35, MID_BLUE, TEAL),
    ("C1 [  64,160,160]", "P2 [256,160×160]", 3.15, MID_BLUE, TEAL),
]
# backbone column
add_text(sl, "Backbone\nOutputs", 0.4, 1.28, 2.2, 0.48,
         size=13, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
# FPN output column
add_text(sl, "FPN Pyramid\nLevels", 9.5, 1.28, 2.2, 0.48,
         size=13, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

for ci, (c_lbl, p_lbl, offs, bg_c, bg_p) in enumerate(fpn_levels):
    y = 1.9 + ci * 0.9
    add_rect(sl, 0.35, y, 2.5, 0.62, bg_c)
    add_text(sl, c_lbl, 0.42, y+0.12, 2.36, 0.4, size=12, color=WHITE, align=PP_ALIGN.CENTER)
    # lateral 1×1
    add_rect(sl, 3.1, y+0.1, 1.3, 0.42, LIGHT_BLUE, MID_BLUE, 0.5)
    add_text(sl, "1×1 →256", 3.15, y+0.16, 1.2, 0.3, size=11, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    add_text(sl, "→", 2.85, y+0.16, 0.3, 0.3, size=16, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    # top-down arrow from above (except first)
    if ci > 0:
        add_text(sl, "↓ up2×+", 4.7, y-0.3, 1.2, 0.4, size=11,
                 color=TEAL, align=PP_ALIGN.CENTER, italic=True)
    # 3×3 output conv
    add_rect(sl, 6.2, y+0.1, 1.5, 0.42, RGBColor(0xE3,0xF2,0xFD), MID_BLUE, 0.5)
    add_text(sl, "3×3 conv", 6.25, y+0.16, 1.4, 0.3, size=11, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    # output
    add_rect(sl, 8.0, y, 2.4, 0.62, bg_p)
    add_text(sl, p_lbl, 8.08, y+0.12, 2.26, 0.4, size=12, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, "→", 7.8, y+0.16, 0.25, 0.3, size=16, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# P6
add_rect(sl, 8.0, 1.9+4*0.9, 2.4, 0.62, MID_BLUE)
add_text(sl, "P6 [256, 10×10]", 8.08, 1.9+4*0.9+0.12, 2.26, 0.4,
         size=12, color=WHITE, align=PP_ALIGN.CENTER)
add_text(sl, "C4→3×3 s2", 3.1, 1.9+4*0.9+0.12, 1.7, 0.4,
         size=11, color=MID_BLUE, italic=True)

# level table
simple_table(sl,
    ["Level", "Resolution", "Stride", "Detects"],
    [
        ["P2", "160×160", "4",  "Small (<32 px)"],
        ["P3", "80×80",   "8",  "Small-medium"],
        ["P4", "40×40",   "16", "Medium"],
        ["P5", "20×20",   "32", "Large"],
        ["P6", "10×10",   "64", "Extra-large"],
    ],
    10.5, 1.9, 2.65,
    col_widths=[0.55, 0.75, 0.6, 0.75], font_size=11)


# ═══════════════════════════ SLIDE 7 — RetinaNet Head ══════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Detection Head: RetinaNet",
             "Anchor-based dual-tower predictor at every FPN level")

# anchor illustration
add_rect(sl, 0.35, 1.35, 4.5, 5.5, WHITE, MID_BLUE, 0.8)
add_rect(sl, 0.35, 1.35, 4.5, 0.32, MID_BLUE)
add_text(sl, "Anchor System", 0.47, 1.38, 4.3, 0.28,
         size=15, bold=True, color=WHITE)

anchor_info = [
    "9 anchors per spatial location",
    "  3 scales: 4, 6, 8 × stride",
    "  3 ratios: 0.5, 1.0, 2.0",
    "",
    "Total ~307,000 anchors / image:",
    "  P2: 160×160×9 = 230,400",
    "  P3:   80×80×9 =  57,600",
    "  P4:   40×40×9 =  14,400",
    "  P5:   20×20×9 =   3,600",
    "  P6:   10×10×9 =     900",
    "",
    "Matching (training):",
    "  IoU > 0.50 → positive",
    "  IoU < 0.40 → negative",
    "  0.40–0.50  → ignored",
]
txb = sl.shapes.add_textbox(Inches(0.55), Inches(1.82), Inches(4.1), Inches(4.9))
tf = txb.text_frame; tf.word_wrap = True
for i, line in enumerate(anchor_info):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    run = p.add_run(); run.text = line
    run.font.size = Pt(13)
    run.font.color.rgb = MID_BLUE if line and not line.startswith(" ") else DARK_TEXT
    run.font.bold = bool(line and not line.startswith(" ") and line != "")

# head diagram
add_rect(sl, 5.1, 1.35, 7.8, 5.5, WHITE, MID_BLUE, 0.8)
add_rect(sl, 5.1, 1.35, 7.8, 0.32, MID_BLUE)
add_text(sl, "Head Architecture (shared across all 5 FPN levels)", 5.22, 1.38, 7.6, 0.28,
         size=15, bold=True, color=WHITE)

head_steps = [
    ("FPN feature map [256, H, W]",  LIGHT_GRAY),
    ("Conv 3×3 [256→256]  (×4 stacked)",  RGBColor(0xBB,0xDE,0xFB)),
    ("(no GroupNorm as per URPC config)",  LIGHT_GRAY),
    ("cls_pred: Conv [256→9×4 classes]",  RGBColor(0xC8,0xE6,0xC9)),
    ("→ sigmoid → class scores",           RGBColor(0xC8,0xE6,0xC9)),
    ("reg_pred: Conv [256→9×4 coords]",   RGBColor(0xFF,0xE0,0xB2)),
    ("→ delta-decode → bounding boxes",   RGBColor(0xFF,0xE0,0xB2)),
]
for i, (lbl, bg) in enumerate(head_steps):
    y = 1.82 + i * 0.64
    add_rect(sl, 5.35, y, 7.3, 0.52, bg, MID_BLUE, 0.3)
    add_text(sl, lbl, 5.45, y+0.1, 7.1, 0.35, size=13, color=DARK_TEXT)
    if i < len(head_steps)-1:
        add_text(sl, "↓", 8.6, y+0.52, 0.4, 0.12,
                 size=10, color=MID_BLUE, align=PP_ALIGN.CENTER)

# loss callout
add_rect(sl, 5.35, 6.35, 3.4, 0.45, RGBColor(0xE8,0xF5,0xE9), MID_BLUE, 0.4)
add_text(sl, "Loss: Focal Loss (γ=2, α=0.25) — down-weights easy negatives",
         5.45, 6.4, 3.2, 0.36, size=11, color=GREEN)
add_rect(sl, 9.0, 6.35, 3.2, 0.45, RGBColor(0xFF, 0xF3, 0xE0), MID_BLUE, 0.4)
add_text(sl, "Post-process: score_thr=0.05 → NMS(IoU=0.5) → top-100",
         9.1, 6.4, 3.1, 0.36, size=11, color=DARK_TEXT)


# ═══════════════════════════ SLIDE 8 — Training Results ════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Training Results & Baseline Comparison",
             "URPC val2018 — 800 images — 4 classes")

simple_table(sl,
    ["Model", "Backbone", "Params", "mAP@50", "mAP (0.5:0.95)"],
    [
        ["SSD-300",       "VGG-16",       "26.3 M", "~0.63", "~0.33"],
        ["RetinaNet",     "ResNet-50",    "37.7 M", "~0.71", "~0.38"],
        ["Faster RCNN",   "ResNet-50",    "41.4 M", "~0.73", "~0.40"],
        ["YOLOv5-S",      "CSPDarkNet-s", "7.2 M",  "~0.71", "~0.39"],
        ["**HTDet (ep42)**", "MobileViT-S", "**12.4 M**", "**0.762**", "**0.414**"],
        ["**HTDet (ep60)**", "MobileViT-S", "**12.4 M**", "**0.755**", "**0.408**"],
    ],
    0.4, 1.35, 12.5,
    col_widths=[2.8, 2.5, 2.0, 2.0, 3.2], font_size=13)

bullet_box(sl, [
    "Peak mAP at epoch 42: 0.414 / mAP@50 = 0.762",
    "Final epoch-60 checkpoint: mAP = 0.408 / mAP@50 = 0.755",
    "Beats ResNet-50 Faster RCNN (41.4M) with only 12.4M params — 3× fewer",
    "SGD lr=0.01, momentum=0.9, 60 epochs, step decay at 40 & 55",
    "MobileViT global attention helps with low-contrast underwater imagery",
], 0.4, 4.55, 8.5, 2.7,
   title="Key Takeaways", bullet_size=15)

# callout boxes
callout(sl, "12.4 M\nParams", 9.2, 4.6, 1.9, 1.0, GOLD, DARK_TEXT, 17)
callout(sl, "0.755\nmAP@50", 11.2, 4.6, 1.9, 1.0, MID_BLUE, WHITE, 17)
callout(sl, "198.9\nGFLOPs", 9.2, 5.75, 1.9, 1.0, TEAL, WHITE, 17)
callout(sl, "3× fewer\nparams", 11.2, 5.75, 1.9, 1.0, GREEN, WHITE, 17)


# ═══════════════════════════ SLIDE 9 — Python-to-C Conversion ══════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Python → C/HLS Conversion",
             "Hand-written C++ targeting Vitis HLS for FPGA synthesis")

bullet_box(sl, [
    "PyTorch cannot synthesise to FPGA directly — need Vitis HLS C++",
    "Strategy: mirror every layer in C++ → validate cosine similarity → then quantise",
    "C-sim compiled with g++ -O3 -std=c++17; HLS synthesis uses Vitis HLS 2022+",
    "HLS stubs (hls_stubs/) let g++ compile ap_fixed/ap_int/hls_stream without Xilinx",
], 0.4, 1.35, 7.8, 2.2, title="Motivation & Strategy", bullet_size=15)

simple_table(sl,
    ["File", "Role"],
    [
        ["fpga_types.h",        "Precision typedefs — wint8_t, act8b*_t, acc_t=ap_fixed<32,16>"],
        ["fpga_utils.h",        "All conv primitives: 3×3, 1×1, depthwise, BN fusion, SiLU"],
        ["mobilevit_backbone.h","MobileViT-S: MBConv blocks + MobileViT Transformer blocks"],
        ["fpn_neck.h",          "FPN top-down pyramid — lateral + output 3×3 convs"],
        ["retina_head.h",       "RetinaNet head — topK candidate filter + NMS"],
        ["htdet_top.cpp",       "HLS entry points: backbone_only, fpn_only, full"],
        ["testbench.cpp",       "C-sim driver — loads binary weights, runs inference, writes .txt"],
    ],
    0.4, 3.8, 12.5,
    col_widths=[3.2, 9.3], font_size=12)

bullet_box(sl, [
    "BN fused into conv at export: W_fused = W_conv × (γ/√(var+ε))  per output channel",
    "Fused bias absorbs BN shift — zero runtime BN computation in C",
    "RetinaNet head (no BN): exported as [weight, ones(C), bias] triples",
], 0.4, 6.55, 12.5, 0.85,
   bullet_size=13)


# ═══════════════════════════ SLIDE 10 — PTQ Overview ═══════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Post-Training Quantization (PTQ)",
             "W8A32 → W8A8 — INT8 weights and activations")

# left: what is quantization
bullet_box(sl, [
    "W8A32: INT8 per-channel weights, FP32 activations",
    "W8A8:  INT8 weights + INT8 fake-quantized activations",
    "53 Conv2d layers quantized; Transformer Linear layers stay FP32",
    "Calibration: 200 URPC val images (seed=42) via ptq_calibrate.py",
    "2 mixed-precision outlier DW layers kept FP32 (ActMax > 196)",
    "All scale factors stored in weights/scales.json",
], 0.35, 1.35, 6.8, 3.1, title="PTQ Setup", bullet_size=14)

# right: fake quant explanation
add_rect(sl, 7.45, 1.35, 5.5, 3.1, WHITE, MID_BLUE, 0.8)
add_rect(sl, 7.45, 1.35, 5.5, 0.32, TEAL)
add_text(sl, "Fake Quantization (W8A8)", 7.57, 1.38, 5.3, 0.28,
         size=15, bold=True, color=WHITE)

code_box(sl, [
    "// INT8 round → clamp → dequantize (back to float)",
    "float q = roundf(act / scale);",
    "if (q >  127) q =  127;",
    "if (q < -128) q = -128;",
    "act = q * scale;  // still float32!",
    "",
    "// No int8_t stored — inject rounding error only",
    "// Matches exactly what real INT8 hardware produces",
], 7.55, 1.82, 5.3, 2.4, font_size=11)

# accuracy table
simple_table(sl,
    ["Metric", "Float32", "W8A32", "W8A8", "Δ (F32→W8A8)"],
    [
        ["mAP (0.50:0.95)", "0.408", "0.407", "**0.407**", "−0.001"],
        ["mAP@50",          "0.755", "0.753", "**0.753**", "−0.002"],
        ["mAP@75",          "0.398", "0.395", "**0.396**", "−0.002"],
        ["mAP (small)",     "0.242", "0.239", "0.239",     "−0.003"],
        ["mAP (medium)",    "0.417", "0.416", "0.416",     "−0.001"],
        ["mAP (large)",     "0.515", "0.513", "0.513",     "−0.002"],
    ],
    0.35, 4.65, 12.6,
    col_widths=[2.8, 2.0, 2.0, 2.0, 3.8], font_size=13)

add_text(sl,
    "All values: Python-evaluated (tools/test.py, 800 URPC val images). W8A8 = W8A32 accuracy — essentially lossless.",
    0.35, 6.88, 12.6, 0.45, size=12, color=MID_BLUE, italic=True)


# ═══════════════════════════ SLIDE 11 — PTQ Benefits ═══════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "PTQ Hardware Benefits",
             "Memory, compute, and latency improvements on FPGA")

simple_table(sl,
    ["Metric", "Float32", "W8A32", "W8A8", "W8A8 vs FP32"],
    [
        ["Conv weight bytes",    "49.7 MB", "9.49 MB", "9.49 MB",  "**5.2×**"],
        ["Activation bytes",     "34.9 MB", "34.9 MB", "8.73 MB",  "**4×**"],
        ["Total on-chip",        "~84 MB",  "~44 MB",  "~18 MB",   "**−79%**"],
        ["BRAM36 (estimated)",   "~19,257", "~10,107", "~4,147",   "**−78%**"],
        ["DSPs per 8-MAC PE",    "~24",     "~24",     "~2",       "**−92%**"],
        ["Weight BW multiplier", "1×",      "4×",      "4×",       "—"],
        ["Act BW multiplier",    "1×",      "1×",      "4×",       "—"],
        ["Est. FPGA latency",    "~500 ms", "~200 ms", "~40–60 ms","**~8–12×**"],
    ],
    0.35, 1.35, 12.6,
    col_widths=[3.0, 2.2, 2.2, 2.2, 3.0], font_size=13)

callout(sl, "W8A32\nSaves BRAM\n2.5× faster",  0.5, 5.05, 3.0, 1.6, MID_BLUE, WHITE, 14)
callout(sl, "W8A8\nSaves BRAM + DSPs\n8–12× faster", 3.7, 5.05, 3.5, 1.6, TEAL, WHITE, 14)
callout(sl, "DSP48E2\n4× INT8 MACs\nper primitive", 7.4, 5.05, 2.8, 1.6, GREEN, WHITE, 14)
callout(sl, "ZU9EG\nTarget FPGA\n200 MHz", 10.4, 5.05, 2.6, 1.6, GOLD, DARK_TEXT, 14)


# ═══════════════════════════ SLIDE 12 — C-Sim Bug: TopK ═══════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "C-Sim Bug #1: TopK Truncation (Critical)",
             "Raster-order early exit silently dropped high-scoring detections")

# BROKEN code
add_text(sl, "BROKEN — early exit on outer loops", 0.35, 1.38, 5.9, 0.32,
         size=14, bold=True, color=RED)
code_box(sl, [
    "// Stops scanning once buffer is full!",
    "// All detections in bottom-right are DROPPED",
    "for (int h = 0; h < H && num_cand < max_cand; h++)",
    "  for (int w = 0; w < W && num_cand < max_cand; w++)",
    "    for (int c = 0; c < NUM_CLASSES; c++) {",
    "        score_t s = sigmoid(cls_logits[...]);",
    "        if (s < SCORE_THR) continue;",
    "        cand[num_cand++] = ...; // early exit!",
    "    }",
], 0.35, 1.75, 6.0, 2.3, font_size=10)

# FIXED code
add_text(sl, "FIXED — true min-tracking top-K", 6.6, 1.38, 6.4, 0.32,
         size=14, bold=True, color=GREEN)
code_box(sl, [
    "int num_cand=0; score_t buf_min=2.0f; int buf_min_idx=0;",
    "for (int h=0; h<H; h++)          // scan ALL positions",
    "  for (int w=0; w<W; w++)",
    "    for (int c=0; c<NUM_CLASSES; c++) {",
    "        score_t s = sigmoid(cls_logits[...]);",
    "        if (s < SCORE_THR) continue;",
    "        if (num_cand < max_cand) {",
    "            cand[num_cand]=s;",
    "            if (s < buf_min){buf_min=s; buf_min_idx=num_cand;}",
    "            num_cand++;",
    "        } else if (s > buf_min) {   // displace minimum",
    "            cand[buf_min_idx]=s;",
    "            // rescan buffer for new minimum",
    "            buf_min=cand[0]; buf_min_idx=0;",
    "            for(int k=1;k<num_cand;k++)",
    "              if(cand[k]<buf_min){buf_min=cand[k];buf_min_idx=k;}",
    "        }",
    "    }",
], 6.6, 1.75, 6.4, 3.8, font_size=9)

bullet_box(sl, [
    "Impact: CHN083846_0291 had 37 Python reference detections spread across the full image",
    "Broken version silently dropped 6–8 detections per FPN level from bottom-right",
    "After fix: CHN recall 83.8%, YDXJ recall 100%, GOPR recall 84.2%",
    "Root cause: HLS fixed-size buffer requires early-stop idiom — but C-sim must use true topK",
], 0.35, 4.2, 12.6, 2.45, title="Impact & Root Cause", bullet_size=14)


# ═══════════════════════════ SLIDE 13 — C-Sim Bug: Exp Clamp + Labels ═

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "C-Sim Bugs #2 & #3",
             "Exp clamp too wide + OpenMP label conflict")

# Bug 2 left panel
add_rect(sl, 0.35, 1.35, 6.1, 5.6, WHITE, MID_BLUE, 0.8)
add_rect(sl, 0.35, 1.35, 6.1, 0.32, RED)
add_text(sl, "Bug #2 — Exp Clamp Too Wide", 0.47, 1.38, 5.9, 0.28,
         size=15, bold=True, color=WHITE)

code_box(sl, [
    "// BEFORE: ±4.135 clamp",
    "// expf(4.135) ≈ 62.5 → box 62× anchor size",
    "if (dw < -4.135f) dw = -4.135f;",
    "if (dw >  4.135f) dw =  4.135f;",
    "",
    "// AFTER: ±2.5 clamp",
    "// expf(2.5) ≈ 12.2 → safe maximum box size",
    "if (dw < -2.5f) dw = -2.5f;",
    "if (dw >  2.5f) dw =  2.5f;",
    "if (dh < -2.5f) dh = -2.5f;",
    "if (dh >  2.5f) dh =  2.5f;",
], 0.5, 1.82, 5.8, 2.6, font_size=11)

txb = sl.shapes.add_textbox(Inches(0.5), Inches(4.55), Inches(5.8), Inches(2.3))
tf = txb.text_frame; tf.word_wrap = True
for line in [
    "Why: quantisation noise can push dw/dh slightly above the clamp.",
    "At ±4.135 this creates 62× oversized boxes polluting NMS.",
    "±2.5 still covers any real object vs anchor but prevents blowup.",
]:
    p = tf.add_paragraph(); run = p.add_run(); run.text = "▸  " + line
    run.font.size = Pt(13); run.font.color.rgb = DARK_TEXT

# Bug 3 right panel
add_rect(sl, 6.75, 1.35, 6.2, 5.6, WHITE, MID_BLUE, 0.8)
add_rect(sl, 6.75, 1.35, 6.2, 0.32, TEAL)
add_text(sl, "Bug #3 — HLS Label vs OpenMP", 6.87, 1.38, 6.0, 0.28,
         size=15, bold=True, color=WHITE)

code_box(sl, [
    "// BROKEN — label between pragma and for",
    "#pragma omp parallel for schedule(static)",
    "C3X3_OC:                   // GCC error here!",
    "for (int oc = 0; oc < OC; oc++) { ... }",
    "",
    "// Error: 'for statement expected before C3X3_OC'",
    "",
    "// FIXED — remove HLS label, keep pragma",
    "#pragma omp parallel for schedule(static)",
    "for (int oc = 0; oc < OC; oc++) { ... }",
    "",
    "// HLS labels are Vitis-only synthesis hints,",
    "// completely ignored by g++ (safe to remove).",
], 6.9, 1.82, 5.9, 3.2, font_size=10)

txb = sl.shapes.add_textbox(Inches(6.9), Inches(5.15), Inches(5.9), Inches(1.7))
tf = txb.text_frame; tf.word_wrap = True
for line in [
    "HLS loop labels rename loops in Vitis synthesis reports only.",
    "GCC's OpenMP requires 'for' to immediately follow the pragma.",
    "Removed from all 8 parallelised outer loops — no HLS impact.",
]:
    p = tf.add_paragraph(); run = p.add_run(); run.text = "▸  " + line
    run.font.size = Pt(13); run.font.color.rgb = DARK_TEXT


# ═══════════════════════════ SLIDE 14 — OpenMP Performance ═════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "C-Sim Performance: OpenMP CPU Parallelism",
             "10× speedup — from 51 minutes to under 5 minutes per image")

# left: why slow + fix
bullet_box(sl, [
    "C-sim runs all 12.41 M params on a single CPU thread",
    "MobileViT stage-3 Transformer: O(N²) attention, N=1600 tokens",
    "Full 640×640 image: ~51 minutes single-threaded",
    "Solution: #pragma omp parallel for on outer OC loop of every conv",
    "Each OC iteration writes to a disjoint output slice → no false sharing",
    "Compiled with: g++ -O3 -std=c++17 -fopenmp",
], 0.35, 1.35, 6.1, 3.6, title="Problem & Solution", bullet_size=14)

# timing table
simple_table(sl,
    ["Mode", "Runtime", "CPU Utilisation", "Speedup"],
    [
        ["Single-threaded", "~51 min", "~100% (1 core)", "1×"],
        ["**OpenMP**", "**~4 min 54 sec**", "**~1881% (~18.8 cores)**", "**~10×**"],
    ],
    0.35, 5.25, 6.1,
    col_widths=[1.8, 1.8, 2.0, 0.5], font_size=13)

# code
add_text(sl, "Pattern applied to all 8 major conv functions:", 6.6, 1.35, 6.4, 0.38,
         size=14, bold=True, color=DARK_BLUE)
code_box(sl, [
    "// fpga_utils.h — conv3x3_bn_silu (example)",
    "#ifdef _OPENMP",
    "#include <omp.h>",
    "#endif",
    "",
    "void conv3x3_bn_silu(float* in, float* w, float* b,",
    "                     float* out, int IC, int OC, ...) {",
    "",
    "    #pragma omp parallel for schedule(static)",
    "    for (int oc = 0; oc < OC; oc++) {",
    "        for (int h = 0; h < OH; h++) {",
    "            for (int w_i = 0; w_i < OW; w_i++) {",
    "                float acc = bias[oc];",
    "                for (int ic=0; ic<IC; ic++)",
    "                    for (int kh=0; kh<3; kh++)",
    "                        for (int kw=0; kw<3; kw++)",
    "                            acc += in[...] * w[...];",
    "                out[oc*OH*OW + h*OW + w_i] = silu(acc);",
    "            }",
    "        }",
    "    }",
    "}  // HLS #pragma directives inside are ignored by g++",
], 6.6, 1.82, 6.4, 5.1, font_size=9)


# ═══════════════════════════ SLIDE 15 — C-Sim Validation ═══════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "C-Sim W8A8 Validation Results",
             "3 URPC val images — IoU-matched vs PyTorch reference")

simple_table(sl,
    ["Image", "Python ref", "W8A8 C-sim", "Recall", "Precision", "FP", "Avg IoU", "Avg |ΔScore|"],
    [
        ["YDXJ0001_10003", "7",  "8",  "100.0%", "87.5%", "1", "0.932", "0.0099"],
        ["CHN083846_0291", "37", "34", "83.8%",  "91.2%", "3", "0.911", "0.0128"],
        ["GOPR0293_10229", "19", "19", "84.2%",  "84.2%", "3", "0.908", "0.0141"],
        ["**MEAN**",       "—",  "—",  "**89.3%**","**87.6%**","—","**0.917**","**0.0123**"],
    ],
    0.35, 1.35, 12.6,
    col_widths=[2.6, 1.4, 1.4, 1.3, 1.4, 0.7, 1.4, 1.4], font_size=13)

bullet_box(sl, [
    "Recall 89.3% — missed detections are all low-confidence (score < 0.25) near FPGA threshold",
    "Avg IoU 0.917 — matched boxes are >91% overlapping with Python reference (near-identical boxes)",
    "Avg |ΔScore| 0.0123 — scores differ by only ~1.2% due to fake-quant rounding in sigmoid inputs",
    "7 total false positives across 3 images — real model outputs between score_thr 0.05 (Python) and 0.20 (FPGA)",
    "No saturation artefacts (old bug produced constant 0.875 or 511.984 scores) — confirmed clean",
    "Multi-class validation confirmed: GOPR has holothurian, echinus, scallop, starfish — all reproduced",
], 0.35, 3.75, 12.6, 3.45, title="What the Numbers Mean", bullet_size=14)


# ═══════════════════════════ SLIDE 16 — HLS Code Review ═══════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "HLS Code Review: 6 Issues Fixed in htdet_top.cpp",
             "Ensuring correct Vitis HLS synthesis")

issues = [
    ("1", "Wrong weight size comments",
     "Said '~5.7M × 2B ≈ 11.4MB' (INT8 estimate). Fixed to float32 actuals: backbone 19.8 MB, FPN 9.9 MB, head 18 MB."),
    ("2", "P2 synthesis infeasibility",
     "P2 = 256×160×160 = 26.2 MB — too large for ZU9EG on-chip BRAM. Added explicit comment; tiling required for synthesis."),
    ("3", "BRAM pragma inconsistency",
     "Mixed old RESOURCE pragma with new bind_storage. Standardised all to: #pragma HLS bind_storage type=RAM_T2P impl=BRAM"),
    ("4", "Missing AXI depths on debug kernels",
     "htdet_backbone_only / htdet_fpn_only had no depth= on m_axi ports → Vitis defaulted to depth=1. Added correct depths."),
    ("5", "FPN-only kernel had no m_axi interface",
     "htdet_fpn_only used only bind_storage — unsynthesisable as standalone kernel. Added m_axi for all c1-c4 and p2-p6 ports."),
    ("6", "No DATAFLOW pragma",
     "Backbone→FPN→Head run sequentially. #pragma HLS DATAFLOW needed at htdet_full to pipeline stages for higher throughput."),
]
for i, (num, title_str, detail) in enumerate(issues):
    col = i % 2
    row = i // 2
    lx = 0.35 + col * 6.5
    ty = 1.4  + row * 1.95
    add_rect(sl, lx, ty, 6.2, 1.75, WHITE, MID_BLUE, 0.7)
    add_rect(sl, lx, ty, 0.45, 1.75, RED if int(num) <= 5 else TEAL)
    add_text(sl, num, lx+0.05, ty+0.55, 0.35, 0.65,
             size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, title_str, lx+0.55, ty+0.1, 5.55, 0.35,
             size=13, bold=True, color=DARK_BLUE)
    add_text(sl, detail, lx+0.55, ty+0.48, 5.55, 1.2,
             size=11.5, color=DARK_TEXT, wrap=True)


# ═══════════════════════════ SLIDE 17 — Pruned Variants ════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Pruned Model Variants",
             "Three reduced-complexity models for tighter FPGA budgets")

simple_table(sl,
    ["Model", "FPN/Head Channels", "Total GFLOPs", "mAP@50 (est.)", "mAP (est.)"],
    [
        ["Full HTDet",         "256", "198.9",     "**0.755**",  "**0.408**"],
        ["Low-GFLOPs 224",     "224", "~168 (−15%)", "~0.74",   "~0.40"],
        ["Low-GFLOPs 192",     "192", "~142 (−29%)", "~0.72",   "~0.38"],
        ["Channel-Pruned 0.7", "256 (FPN/head)", "~110 (−45%)", "~0.71", "~0.36"],
    ],
    0.35, 1.35, 12.6,
    col_widths=[2.8, 2.6, 2.6, 2.3, 2.3], font_size=14)

bullet_box(sl, [
    "Low-GFLOPs 224: 12.5% channel reduction → ~15% GFLOPs saving, ~1% mAP@50 penalty",
    "Low-GFLOPs 192: 25% channel reduction → ~29% GFLOPs saving, ~3% mAP@50 penalty",
    "  Both fine-tuned from shared 42-epoch init checkpoint at lr=0.001 for 60 epochs",
    "Channel-Pruned: width_mult=0.7 → backbone channels [11,22,44,88,176]",
    "  Trained from random init (no pretrained weights) — slower convergence",
    "  ~45% GFLOPs reduction but FPN/head stays 256ch to limit detection quality loss",
    "GFLOPs scale ∝ (channels)² for conv layers — 192/256 ratio gives 0.56× GFLOPs",
], 0.35, 3.85, 12.6, 3.3, title="Trade-Off Analysis", bullet_size=13)


# ═══════════════════════════ SLIDE 18 — FPGA Deployment Plan ═══════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "FPGA Deployment Roadmap",
             "From C-sim validation to working ZU9EG bitstream")

steps = [
    ("C-sim\nValidated ✓",    "3 images, 89.3% recall, 0.917 IoU. All bugs fixed.", GREEN,  0.4),
    ("ap_fixed\nTypes",        "Replace bbox_t=float → ap_fixed<24,12>\nscore_t=float → ap_fixed<16,4>", MID_BLUE, 2.9),
    ("DATAFLOW\nPipeline",     "#pragma HLS DATAFLOW at htdet_full\nDecompose into producer/consumer tasks", TEAL,     5.4),
    ("P2 Tiling",              "P2=26.2MB > ZU9EG BRAM budget\nProcess in spatial tiles through head", GOLD,     7.9),
    ("HLS\nSynthesis",         "Run Vitis HLS, check timing/resources\nTarget: 200 MHz, <60 ms latency", MID_BLUE, 10.4),
]
for lbl, detail, bg, lx in steps:
    add_rect(sl, lx, 1.5, 2.4, 1.1, bg)
    add_text(sl, lbl, lx+0.1, 1.6, 2.2, 0.9, size=15, bold=True,
             color=WHITE if bg != GOLD else DARK_TEXT, align=PP_ALIGN.CENTER)
    add_text(sl, detail, lx, 2.75, 2.5, 1.1, size=11, color=DARK_TEXT, wrap=True)
    if lx < 10.4:
        add_text(sl, "→", lx+2.4, 1.9, 0.5, 0.5,
                 size=22, bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)

# additional roadmap items
bullet_box(sl, [
    "FPN fake_quant: add fake_quant_buf() after top-down additions (lat3 += up4, etc.)",
    "QAT (optional): 5–10 epochs quantization-aware training to recover the ~1.2% score delta",
    "Per-channel activation quantization: replace per-tensor act scales in retina_cls",
    "Vivado implementation: place-and-route, power analysis, timing closure at 200 MHz",
    "Board validation: deploy to ZU9EG board, capture camera frames, compare vs PyTorch output",
], 0.35, 4.05, 12.6, 3.1, title="Remaining Synthesis & Validation Work", bullet_size=14)


# ═══════════════════════════ SLIDE 19 — Full Results Dashboard ═════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, LIGHT_GRAY)
slide_header(sl, "Full Results Dashboard",
             "All key metrics at a glance")

# mAP panel
add_rect(sl, 0.35, 1.35, 5.8, 3.0, WHITE, MID_BLUE, 0.8)
add_rect(sl, 0.35, 1.35, 5.8, 0.32, MID_BLUE)
add_text(sl, "mAP Accuracy (800 val images)", 0.47, 1.38, 5.6, 0.28,
         size=14, bold=True, color=WHITE)
map_data = [
    ("Float32 baseline", "0.755", "0.408"),
    ("W8A32 PTQ",        "0.753", "0.407"),
    ("W8A8 PTQ",         "0.753", "0.407"),
]
add_text(sl, "Model              mAP@50   mAP", 0.5, 1.75, 5.5, 0.3,
         size=12, bold=True, color=MID_BLUE)
for i, (m, a50, a) in enumerate(map_data):
    add_text(sl, f"{m:<20}  {a50}    {a}",
             0.5, 2.1 + i * 0.55, 5.5, 0.45, size=13, color=DARK_TEXT)

# C-sim panel
add_rect(sl, 6.45, 1.35, 6.5, 3.0, WHITE, MID_BLUE, 0.8)
add_rect(sl, 6.45, 1.35, 6.5, 0.32, MID_BLUE)
add_text(sl, "C-Sim W8A8 Validation", 6.57, 1.38, 6.3, 0.28,
         size=14, bold=True, color=WHITE)
csim_data = [
    ("YDXJ (7 ref)",  "100.0%", "87.5%", "0.932"),
    ("CHN (37 ref)",  " 83.8%", "91.2%", "0.911"),
    ("GOPR (19 ref)", " 84.2%", "84.2%", "0.908"),
    ("MEAN",          " 89.3%", "87.6%", "0.917"),
]
add_text(sl, "Image           Recall   Prec   AvgIoU",
         6.6, 1.75, 6.2, 0.3, size=12, bold=True, color=MID_BLUE)
for i, (img, rec, prec, iou) in enumerate(csim_data):
    clr = MID_BLUE if i == 3 else DARK_TEXT
    add_text(sl, f"{img:<17}  {rec}  {prec}  {iou}",
             6.6, 2.1 + i * 0.55, 6.2, 0.45, size=13, color=clr, bold=(i == 3))

# resource savings panel
add_rect(sl, 0.35, 4.55, 5.8, 2.7, WHITE, MID_BLUE, 0.8)
add_rect(sl, 0.35, 4.55, 5.8, 0.32, TEAL)
add_text(sl, "FPGA Resource Savings (W8A8 vs FP32)", 0.47, 4.58, 5.6, 0.28,
         size=13, bold=True, color=WHITE)
rsrc = [
    ("Conv weight memory", "49.7 MB", "9.49 MB", "5.2×"),
    ("Activation memory",  "34.9 MB", "8.73 MB", "4×"),
    ("BRAM36 (est.)",      "~19,257", "~4,147",  "−78%"),
    ("DSPs per PE",        "~24",     "~2",      "−92%"),
    ("Est. latency",       "~500 ms", "~40-60ms","8–12×"),
]
add_text(sl, "Metric            FP32        W8A8    Saving",
         0.5, 4.95, 5.5, 0.3, size=11, bold=True, color=MID_BLUE)
for i, (m, f32, w8, sav) in enumerate(rsrc):
    add_text(sl, f"{m:<20}  {f32:<10} {w8:<9} {sav}",
             0.5, 5.32 + i * 0.37, 5.5, 0.32, size=11, color=DARK_TEXT)

# runtime panel
add_rect(sl, 6.45, 4.55, 6.5, 2.7, WHITE, MID_BLUE, 0.8)
add_rect(sl, 6.45, 4.55, 6.5, 0.32, GOLD)
add_text(sl, "C-Sim Runtime (OpenMP)", 6.57, 4.58, 6.3, 0.28,
         size=13, bold=True, color=DARK_TEXT)
add_text(sl, "Single-threaded:    ~51 minutes / image",
         6.6, 4.98, 6.2, 0.42, size=14, color=RED)
add_text(sl, "OpenMP (-fopenmp):  ~4 min 54 sec / image",
         6.6, 5.48, 6.2, 0.42, size=14, color=GREEN, bold=True)
add_text(sl, "Speedup: ~10×   CPU utilisation: ~1881% (~18.8 cores)",
         6.6, 5.98, 6.2, 0.42, size=13, color=MID_BLUE, bold=True)
add_text(sl, "Compiled: g++ -O3 -std=c++17 -fopenmp",
         6.6, 6.48, 6.2, 0.35, size=12, color=DARK_TEXT, italic=True)


# ═══════════════════════════ SLIDE 20 — Conclusion ═════════════════════

sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, 13.33, 7.5, DARK_BLUE)
add_rect(sl, 0, 1.3, 13.33, 0.07, TEAL)
add_rect(sl, 0, 6.1, 13.33, 0.07, TEAL)

add_text(sl, "Conclusion & Key Contributions", 0.5, 0.25, 12.3, 0.9,
         size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

contributions = [
    ("0.755 mAP@50",
     "State-of-the-art on URPC 2018 with only 12.4 M parameters — beats ResNet-50 Faster RCNN (41.4 M)"),
    ("W8A8 = W8A32",
     "Post-training quantization to INT8 weights + INT8 activations with zero accuracy loss vs W8A32 (0.753 both)"),
    ("89.3% C-sim recall",
     "W8A8 C-simulation reproduces 89.3% of Python detections with 0.917 avg IoU across 3 validation images"),
    ("2 critical bugs fixed",
     "topK raster-order truncation and exp-clamp width — both silently corrupted results before fix"),
    ("10× OpenMP speedup",
     "#pragma omp parallel for on outer OC loops: 51 min → 4:54 per image, ~18.8 cores avg utilisation"),
    ("FPGA-ready C/HLS",
     "Complete HLS implementation with all 6 code-review issues fixed; synthesis path defined (DATAFLOW, tiling, ap_fixed)"),
]
for i, (headline, detail) in enumerate(contributions):
    col = i % 2
    row = i // 2
    lx = 0.4 + col * 6.45
    ty = 1.55 + row * 1.45
    add_rect(sl, lx, ty, 6.15, 1.28, RGBColor(0x12,0x3A,0x6A))
    add_rect(sl, lx, ty, 0.12, 1.28, GOLD)
    add_text(sl, headline, lx+0.22, ty+0.1, 5.8, 0.38,
             size=16, bold=True, color=GOLD)
    add_text(sl, detail, lx+0.22, ty+0.52, 5.8, 0.72,
             size=12.5, color=LIGHT_BLUE, wrap=True)

add_text(sl,
    "Next: DATAFLOW pipelining  ·  P2 tiling  ·  ap_fixed type replacement  ·  Vivado synthesis",
    0.5, 6.22, 12.3, 0.52,
    size=15, color=LIGHT_BLUE, align=PP_ALIGN.CENTER, italic=True)


# ═══════════════════════════ Save ══════════════════════════════════════

outpath = "/workspace/ckarfa/htdet/htdet_project/mmdetection/HTDet_Presentation.pptx"
prs.save(outpath)
print(f"Saved: {outpath}")
print(f"Slides: {len(prs.slides)}")
