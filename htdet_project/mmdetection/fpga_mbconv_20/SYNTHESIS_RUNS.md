# mbconv_20 Synthesis Run History

**Module**: `mbconv_20_top` — MBConv residual block at C3 stage  
**Config**: in_ch=128, out_ch=128, expand=4, stride=1, hid=512  
**Device**: xczu28dr-ffvg1517-2-e, Clock: 5 ns (200 MHz)  
**Layers**: expand 1×1 (128→512) + DW 3×3 (512) + proj 1×1 (512→128) + residual add  
**Spatial**: H=20, W=20 (C3_H, C3_W)  
**Layout**: HWC [H*W * ch]

## Key Parameters

| Layer | in_ch | out_ch | IC tiles | HW pixels |
|---|---|---|---|---|
| expand 1×1 | 128 | 512 | 8 | 400 |
| DW 3×3 | 512 | 512 | — | 400 |
| proj 1×1 | 512 | 128 | 32 | 400 |

## URAM Budget (estimate)
- ex_buf: 20×20×512×4 = 819,200 bytes ≈ 0.78 MB
- dw_buf: 20×20×512×4 = 819,200 bytes ≈ 0.78 MB
- res_buf: 20×20×128×4 = 204,800 bytes ≈ 0.20 MB
- Total: **1.78 MB** < device URAM capacity (2.88 MB) ✓

## Synthesis Runs

| Date | OC_TILE | MAX_IC_TILES | Total cycles | @ 200 MHz | DSP | URAM | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-31 | 4 | 32 | 20,736,200 | **104 ms** | 244 (5%) | 125 (156%) | expand II=2, proj II=2 (not flat, 32 tiles), DW II=5 |
