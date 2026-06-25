# run_hls_fpn_p4p5.tcl
# Vitis HLS script — FPN Phase 2: P4+P5+P6
#
# Adds C3 lateral path (128→192, 20×20) and top-down upsample+add on top of Phase 1.
# Target: xczu9eg-ffvb1156-2-e @ 200 MHz
#
# Run from the fpga_fpn_192/ directory:
#   vitis_hls -f run_hls_fpn_p4p5.tcl
#
# Expected synthesis time: 10-20 min (P4 conv3x3 is 4× P5 trip count).
# Key numbers to watch (OC_TILE=16, cyclic-17 partition):
#   FPN3_IC_FPN3_KH_FPN3_KW (P5): trip=1728, II=5; outer=12*100=1200 → ~52 ms
#   FPN3_IC_FPN3_KH_FPN3_KW (P4): trip=1728, II=5; outer=12*400=4800 → ~208 ms
#   C1P_HW_C1P_IC (lat4):    trip=64000, II=5; 12 OC trips → ~19 ms
#   C1P_HW_C1P_IC (lat3):    trip=51200, II=5; 12 OC trips → ~15 ms
#   ADD_UP:                   trip=76800, II=1 (add_upsampled_2x_inplace)

set SRC_DIR  "."
set PRJ_NAME "fpn_p4p5_prj"
set SOL_NAME "fpn_p4p5_200MHz"
set TOP_FUNC "fpn_p4p5_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"

open_project $PRJ_NAME

add_files "${SRC_DIR}/fpn_p4p5_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_fpn_p4p5.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

set_top $TOP_FUNC

open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default

# csim_design -O

set_directive_array_partition -type cyclic -factor 17 "fpn_p4p5_top" w_conv

csynth_design

puts "==================================================="
puts "fpn_p4p5_top synthesis complete (OC_TILE=16, cyclic-17 partition)."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/fpn_p4p5_top_csynth.rpt"
puts "==================================================="
