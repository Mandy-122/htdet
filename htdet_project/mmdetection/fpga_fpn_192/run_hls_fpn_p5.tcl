# run_hls_fpn_p5.tcl
# Vitis HLS script — FPN Phase 1: P5+P6 only
#
# Module: fpn_p5_top
# Input:  C4 feature map [640, 10, 10]  (320×320 input)
# Output: P5 [192, 10, 10], P6 [192, 5, 5]
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Edit PART to match your board.
#
# Run from the fpga_fpn_192/ directory:
#   vitis_hls -f run_hls_fpn_p5.tcl
#
# Outputs:
#   fpn_p5_prj/fpn_p5_200MHz/syn/report/fpn_p5_top_csynth.rpt

# ============================================================
# Configuration
# ============================================================
set SRC_DIR  "."
set PRJ_NAME "fpn_p5_prj"
set SOL_NAME "fpn_p5_200MHz"
set TOP_FUNC "fpn_p5_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"   ;# 5 ns = 200 MHz

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source + testbench files
# ============================================================
add_files "${SRC_DIR}/fpn_p5_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_fpn_p5.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

# ============================================================
# 3. Set top function
# ============================================================
set_top $TOP_FUNC

# ============================================================
# 4. Create solution, set part and clock
# ============================================================
open_solution $SOL_NAME -flow_target vivado

set_part $PART

create_clock -period $CLK_NS -name default

# ============================================================
# 5. C-simulation — uncomment to run before synthesis
# ============================================================
# csim_design -O

# ============================================================
# 6. C-synthesis
# OC_TILE=16: cyclic-17 partition on w_conv enables 16 simultaneous
# weight reads per pipeline cycle → II_BRAM=1 < II_DSP=5 ✓.
# ============================================================
set_directive_array_partition -type cyclic -factor 17 "fpn_p5_top" w_conv

csynth_design

# ============================================================
# 7. Skip export — synthesis report is the deliverable
# ============================================================

puts "==================================================="
puts "fpn_p5_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/fpn_p5_top_csynth.rpt"
puts "==================================================="
