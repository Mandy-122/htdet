# run_hls_ptq.tcl — Vitis HLS synthesis script for HTDet PTQ (W8A32, 192-ch)
#
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Change PART below if you use a different board.
#
# Run from the fpga_ptq_192/ directory:
#   vitis_hls -f run_hls_ptq.tcl
#
# Estimated runtimes (xczu9eg, 200 MHz target):
#   C-simulation : ~20-40 min  (640x640 input, heavy MobileViT attention)
#   C-synthesis  : ~2-4 hours  (large design, ~600k LUTs estimated)
#   Co-simulation: ~4-8 hours  (skip unless needed for RTL verify)

# ============================================================
# Configuration — edit these if your setup differs
# ============================================================
set SRC_DIR  "."
set PRJ_NAME "htdet_ptq_hls_prj"
set SOL_NAME "htdet_ptq_200MHz"
set TOP_FUNC "htdet_inference"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"   ;# 5 ns = 200 MHz

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source files
#    htdet_top.cpp pulls in all other headers via #include.
#    The HLS stubs (ap_int.h etc.) are NOT needed here —
#    Vitis HLS has its own versions in $HLS_ROOT/include.
# ============================================================
add_files "${SRC_DIR}/htdet_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14 -DTESTBENCH_DATA_DIR=\\\"../ptq_results_192\\\""

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
# 5. C-simulation (comment out to skip, go straight to csynth)
#    Requires ptq_results_192/*.bin files one level up.
#    Provide one test image:
#      set_param csim.testbench_args "../csim_validation/multi_image/GOPR0293_10229/input_image.bin"
# ============================================================
# csim_design -O

# ============================================================
# 6. C-synthesis
# ============================================================
csynth_design

# ============================================================
# 7. C/RTL co-simulation (optional, slow)
# ============================================================
# cosim_design -O -rtl verilog

# ============================================================
# 8. Export synthesised IP
# ============================================================
export_design \
    -flow impl \
    -rtl verilog \
    -format ip_catalog \
    -description "HTDet PTQ W8A32 FPGA inference kernel (192-ch)" \
    -vendor   "htdet" \
    -version  "1.0" \
    -output   "${PRJ_NAME}/${SOL_NAME}/htdet_ptq_ip"

puts "==================================================="
puts "HLS synthesis complete."
puts "IP package: ${PRJ_NAME}/${SOL_NAME}/htdet_ptq_ip"
puts "==================================================="
