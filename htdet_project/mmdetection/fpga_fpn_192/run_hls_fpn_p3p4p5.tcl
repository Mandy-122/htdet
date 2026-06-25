# run_hls_fpn_p3p4p5.tcl
# Vitis HLS — FPN Phase 3: P3+P4+P5+P6
#
# New vs Phase 2:
#   + C2 lateral (96→192, 40×40)
#   + add_upsampled_2x_inplace for both top-down merges (no up-buffers)
#   + P3 output conv3x3 (OC_TILE=16, 12×1600 outer × 8659 inner ≈ 833 ms)
#
# Expected synthesis time: ~2–5 min
# Key numbers to watch (OC_TILE=16, cyclic-17 partition):
#   conv1x1_plain (lat2): C1P_OC=12, trip=1600*96=153600, II=5 → ~46 ms
#   fpn_conv3x3   (P3):   FPN3_OC_OH_OW outer=19200 (12*40*40), II=5 → ~833 ms
#   ADD_UP_OH_OW  (both): trip=19200 / 76800, II=1
#
# Run from fpga_fpn_192/:
#   vitis_hls -f run_hls_fpn_p3p4p5.tcl

set SRC_DIR  "."
set PRJ_NAME "fpn_p3p4p5_prj"
set SOL_NAME "fpn_p3p4p5_200MHz"
set TOP_FUNC "fpn_p3p4p5_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"

open_project $PRJ_NAME

add_files "${SRC_DIR}/fpn_p3p4p5_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_fpn_p3p4p5.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

set_top $TOP_FUNC
open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default

set_directive_array_partition -type cyclic -factor 17 "fpn_p3p4p5_top" w_conv

csynth_design

puts "==================================================="
puts "fpn_p3p4p5_top synthesis complete (OC_TILE=16, cyclic-17 partition)."
puts "==================================================="
