# mvit_blk_s2 Synthesis Run History

**Module**: `mvit_blk_s2_top` — Full MobileViT Block Stage-2  
**Config**: in_ch=96 (C2_CH), transformer_dim=144 (MVIT_S2_DIM), depth=2, patch=2  
**Device**: xczu28dr-ffvg1517-2-e, Clock: 5 ns (200 MHz)  
**Spatial**: H=W=40 (C2_H/C2_W), N_tokens=400 per view, 4 views

---

## Module Decomposition

| Sub-module | Operation | Spatial / Tokens | Status |
|---|---|---|---|
| local conv3×3 | 96→96, stride=1, CHW | 40×40 | uncharacterised |
| proj 1×1 | 96→144, CHW | 40×40 | uncharacterised |
| unfold (×4 views) | patch extract, p=2 | 40×40 → 400 tokens | trivial |
| transformer_block_s2 × 2 × 4 views | MHSA+FFN | N=400, d=144 | ✅ `fpga_t_blk_s2`: 94M cycles total |
| layer_norm | seq=400, dim=144 | 400 tokens | trivial |
| fold (×4 views) | patch fold | 400 tokens → 40×40 | trivial |
| back-proj 1×1 | 144→96, CHW | 40×40 | uncharacterised |
| concat + fusion conv3×3 | 192→96, stride=1, CHW | 40×40 | uncharacterised |

---

## Expected Latency Breakdown

| Sub-module | Est. cycles | Est. time |
|---|---|---|
| local conv3×3 (96→96, 40×40) | ~5–10M | ~25–50 ms |
| proj 1×1 (96→144, 40×40) | ~1–3M | ~5–15 ms |
| unfold × 4 views | ~640K (trivial) | ~3 ms |
| transformer × 2 depth × 4 views | **94M** (from fpga_t_blk_s2) | **470 ms** |
| fold × 4 views | ~640K (trivial) | ~3 ms |
| back-proj 1×1 (144→96, 40×40) | ~1–3M | ~5–15 ms |
| concat + fusion conv3×3 (192→96, 40×40) | ~10–20M | ~50–100 ms |
| **Total estimate** | **~110–135M** | **~550–675 ms** |

*Transformer dominates at ~70–85% of total block latency.*

---

## BRAM Estimate

| Buffer | Size | BRAM_18K |
|---|---|---|
| local_feat [96×40×40] | 614 KB | ~43 |
| proj_feat [144×40×40] | 922 KB | ~64 |
| tokens [400×144] | 230 KB | ~16 |
| fold_feat [144×40×40] | 922 KB | ~64 |
| proj_back [96×40×40] | 614 KB | ~43 |
| concat_buf [192×40×40] | 1.2 MB | ~86 |
| transformer buffers (×2 depth) | ~2×700 KB | ~100 |
| **Total** | **~6 MB** | **~416** |

Plus ~400 BRAM from local conv3×3 weight storage. Total ~500–600 BRAM_18K (27–33%).

---

## Synthesis Runs

| Date | Settings | Total cycles | @ 200 MHz | BRAM | Notes |
|---|---|---|---|---|---|
| pending | baseline | TBD | TBD | TBD | First run |

---

## File Map

| File | Description |
|---|---|
| `mvit_blk_s2_top.cpp` | Full block: transformer_block_s2 + mobilevit_block_s2 + top wrapper |
| `mvit_blk_s2_top.h` | Constants: MVIT_S2_WCONV/WMETA_ELEMS, IN/OUT_ELEMS |
| `fpga_utils.h` | CHW conv3×3, conv1×1, layer_norm, linear, softmax (ptq_192 version) |
| `fpga_types.h` | Shared data types |
| `testbench_mvit_blk_s2.cpp` | C-sim (warning: slow due to O(N²) attention, N=400) |
