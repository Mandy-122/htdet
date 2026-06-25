# mvit_blk_s4 Synthesis Run History

**Module**: `mvit_blk_s4_top` — Full MobileViT Block Stage-4
**Config**: in_ch=160 (STAGE4_PRE_CH), transformer_dim=240 (MVIT_S4_DIM), depth=3, patch=2
**Device**: xczu28dr-ffvg1517-2-e, Clock: 5 ns (200 MHz)
**Spatial**: H=W=10 (C4_H/C4_W), N_tokens=25 per view, 4 views

## Expected Latency

| Sub-module | Est. cycles | Est. time |
|---|---|---|
| local + proj + fold ops | <1M | <5 ms |
| transformer × 3 depth × 4 views | **18M** (from fpga_t_blk_s4) | **90 ms** |
| back-proj + fusion | <1M | <5 ms |
| **Total** | **~20M** | **~100 ms** |

Transformer dominates. Smallest MobileViT block due to 10×10 spatial.

## Synthesis Runs

| Date | Total cycles | @ 200 MHz | Notes |
|---|---|---|---|
| pending | TBD | TBD | First run |
