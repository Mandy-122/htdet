# c4_expand Synthesis Run History

**Module**: `c4_expand_top` — Final backbone expansion conv  
**Config**: conv1×1(160→640, 10×10) + BN + SiLU  
**Device**: xczu28dr-ffvg1517-2-e, Clock: 5 ns (200 MHz)  
**Layout**: CHW (matches mobilevit_block_s4 output)  
**Input**: [160, 10, 10] → **Output**: [640, 10, 10]

---

## Key Parameters

| Parameter | Value |
|---|---|
| in_ch | 160 (STAGE4_PRE_CH) |
| out_ch | 640 (C4_CH) |
| Spatial | 10×10 = 100 pixels |
| MACs | 640 × 100 × 160 = 10.24M |
| w_conv | 102,400 int8 |
| w_meta | 1,280 float |
| URAM | none needed (no intermediate buffer) |

## Expected Latency

```
Outer trip = OC × HW = 640 × 100 = 64,000
At II=1: 64,000 cycles ≈ 0.32 ms @ 200 MHz
```
Smallest module in the pipeline. Latency negligible vs FPN/backbone blocks.

## Synthesis Runs

| Date | Settings | Total cycles | @ 200 MHz | Notes |
|---|---|---|---|---|
| pending | baseline | TBD | TBD | First run |
