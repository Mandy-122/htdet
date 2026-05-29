# run_hls_mbconv80.tcl
# Vitis HLS synthesis script — isolated mbconv_80 module (PTQ INT8, W8A32)
#
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Edit PART below if you use a different board.
#
# Run from the fpga_ptq_192_test/ directory:
#   vitis_hls -f run_hls_mbconv80.tcl
#
# This script synthesises ONLY mbconv_80 (160×160 spatial, 64-ch in/out).
# Use it to get per-module resource/timing reports before tackling the full
# htdet_inference top.  Replace with run_hls_mbconv40.tcl etc. for other blocks.
#
# Outputs (in the project directory):
#   mbconv80_prj/mbconv80_200MHz/syn/report/mbconv_80_top_csynth.rpt

# ============================================================
# Configuration
# ============================================================
set SRC_DIR  "."
set PRJ_NAME "mbconv80_prj"
set SOL_NAME "mbconv80_200MHz"
set TOP_FUNC "mbconv_80_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"   ;# 5 ns = 200 MHz

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source files
# ============================================================
add_files "${SRC_DIR}/mbconv_80_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_mbconv80.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

# ============================================================
# 3. Set top function
# ============================================================
set_top $TOP_FUNC

# ============================================================
# 4. Create solution, set device and clock
# ============================================================
open_solution $SOL_NAME -flow_target vivado

set_part $PART

create_clock -period $CLK_NS -name default

# ============================================================
# 5. C-simulation (optional — uncomment to verify before synth)
# ============================================================
# csim_design -O

# ============================================================
# 6. C-synthesis  (resource + timing report for mbconv_80)
# ============================================================
csynth_design

# ============================================================
# 7. Skip export — we only need the synthesis report here
# ============================================================

puts "==================================================="
puts "mbconv_80_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/mbconv_80_top_csynth.rpt"
puts "==================================================="
