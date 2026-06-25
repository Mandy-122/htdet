# run_hls_t_blk_s2.tcl
# Vitis HLS script — transformer_block_s2 incremental synthesis
#
# Module: transformer_blk_s2_top
# Dims  : DIM=16, SEQ=4, HEADS=2, HEAD_DIM=8, MLP_HID=32, DEPTH=1
#         (edit t_blk_s2_types.h to scale up once CSIM passes)
#
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Edit PART below for a different board.
#
# Run from the fpga_t_blk_s2/ directory:
#   vitis_hls -f run_hls_t_blk_s2.tcl
#
# Steps executed:
#   csim_design  — C-simulation (3 tests; must all PASS before synth)
#   csynth_design — Resource + timing report
#
# Outputs:
#   t_blk_s2_prj/t_blk_s2_200MHz/syn/report/transformer_blk_s2_top_csynth.rpt

# ============================================================
# Configuration
# ============================================================
set SRC_DIR  "."
set PRJ_NAME "t_blk_s2_prj"
set SOL_NAME "t_blk_s2_200MHz"
set TOP_FUNC "transformer_blk_s2_top"
set PART     "xczu9eg-ffvb1156-2-e"
set CLK_NS   "5"   ;# 5 ns = 200 MHz

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source files
# ============================================================
add_files "${SRC_DIR}/t_blk_s2_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_t_blk_s2.cpp" \
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
# 5. C-simulation  (all 3 tests must PASS before proceeding)
# ============================================================
csim_design -O

# ============================================================
# 6. C-synthesis  (resource + timing report)
# ============================================================
csynth_design

# ============================================================
# 7. Skip export — synthesis report is sufficient for now
# ============================================================

puts "==========================================================="
puts "transformer_blk_s2_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/transformer_blk_s2_top_csynth.rpt"
puts ""
puts "Scaling checklist (edit t_blk_s2_types.h):"
puts "  Step 1 (done):    DIM=16,  SEQ=4,   HEADS=2, MLP=32,  DEPTH=1 — 8114 cycles"
puts "  Step 2 (done):    DIM=32,  SEQ=4,   HEADS=2, MLP=64,  DEPTH=1 — 23532 cycles"
puts "  Step 3 (done):    DIM=64,  SEQ=16,  HEADS=4, MLP=128, DEPTH=1 — 310804 cycles"
puts "  Step 4 (done):    DIM=144, SEQ=100, HEADS=4, MLP=288, DEPTH=1 — 9.47M cycles"
puts "  Step 5 (current): DIM=144, SEQ=400, HEADS=4, MLP=288, DEPTH=2 — Full S2"
puts "==========================================================="
