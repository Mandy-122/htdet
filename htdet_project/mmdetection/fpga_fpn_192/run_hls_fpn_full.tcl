# run_hls_fpn_full.tcl
# Vitis HLS — FPN Phase 4: full P2+P3+P4+P5+P6
#
# ⚠ BRAM WARNING: lat1[192,80,80] = 4.8 MB requires ~2,401 BRAM18K.
#   ZCU9EG has 1,824 total. Synthesis completes with >100% BRAM utilization;
#   use the report for latency validation only. P&R requires xczu28dr.
#
# Expected synthesis time: ~5-15 min
# Expected latency: ~4.63 s (OC_TILE=16, cyclic-17 partition, 2× faster than OC_TILE=8)
# Key numbers to watch:
#   conv1x1_plain (lat1): C1P_OC=12, trip=6400*64=409600, II=5 → ~123 ms
#   ADD_UP (P3→P2):       trip=1,228,800, II=1 → ~6 ms
#   fpn_conv3x3 (P2):     FPN3_OC_OH_OW=76800 outer × 8659 inner → ~3.33 s
#
# OC_TILE=16 requires cyclic-17 partition on w_conv:
#   16 simultaneous weight reads at stride in_ch or FPN_OUT_CH×9 — coprime with 17
#   → all 16 reads in distinct banks → II_BRAM=1 < II_DSP=5 ✓
#   Without partition: dual-port BRAM limits II≥8 → only ~10% gain over OC_TILE=8.
#
# Run from fpga_fpn_192/:
#   vitis_hls -f run_hls_fpn_full.tcl

set SRC_DIR  "."
set PRJ_NAME "fpn_full_prj"
set SOL_NAME "fpn_full_200MHz"
set TOP_FUNC "fpn_full_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"

open_project $PRJ_NAME

add_files "${SRC_DIR}/fpn_full_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_fpn_full.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

set_top $TOP_FUNC
open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default

# Cyclic-17 partition on w_conv: enables 16 simultaneous weight reads per pipeline
# cycle for FPN_OC_TILE=16.  Factor 17 is chosen because all FPN in_ch values
# (640, 128, 96, 64) and FPN_OUT_CH×9=1728 are coprime with 17, so consecutive
# OC reads always land in distinct banks → II_BRAM=1 < II_DSP=5.
# This creates a 17-bank BRAM interface for w_conv in the generated RTL; the
# system integrator must provide 17 physical BRAM banks connected to those ports.
set_directive_array_partition -type cyclic -factor 17 "fpn_full_top" w_conv

csynth_design

puts "==================================================="
puts "fpn_full_top synthesis complete (OC_TILE=16, cyclic-17 partition)."
puts "Note: BRAM >100% expected — lat1 (4.8 MB) exceeds xczu9eg capacity."
puts "Latency estimate valid; use xczu28dr for P&R."
puts "Expected total latency: ~4.63 s (vs 9.258 s at OC_TILE=8, 2× speedup)."
puts "==================================================="
