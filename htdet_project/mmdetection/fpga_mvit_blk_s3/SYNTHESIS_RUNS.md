# mvit_blk_s3 Synthesis Run History

**Module**: `mvit_blk_s3_top` — Full MobileViT Block Stage-3  
**Config**: in_ch=128 (C3_CH), transformer_dim=192 (MVIT_S3_DIM), depth=4, patch=2  
**Device**: xczu28dr-ffvg1517-2-e, Clock: 5 ns (200 MHz)  
**Spatial**: H=W=20 (C3_H/C3_W), N_tokens=100 per view, 4 views

---

## Expected Latency Breakdown

| Sub-module | Est. cycles | Est. time |
|---|---|---|
| local conv3×3 (128→128, 20×20) | ~1–3M | ~5–15 ms |
| proj 1×1 (128→192, 20×20) | ~0.5–1M | ~2–5 ms |
| transformer × 4 depth × 4 views | **65M** (from fpga_t_blk_s3) | **325 ms** |
| back-proj + fusion conv3×3 | ~3–8M | ~15–40 ms |
| **Total estimate** | **~70–80M** | **~350–400 ms** |

---

## Synthesis Runs

| Date | Settings | Total cycles | @ 200 MHz | BRAM | Notes |
|---|---|---|---|---|---|
| pending | baseline | TBD | TBD | TBD | First run |

---

## File Map

| File | Description |
|---|---|
| `mvit_blk_s3_top.cpp` | Full block: transformer_block_s3 + mobilevit_block_s3 + top |
| `mvit_blk_s3_top.h` | Constants: MVIT_S3_WCONV/WMETA_ELEMS |
| `fpga_utils.h` | CHW conv primitives (ptq_192 version) |
