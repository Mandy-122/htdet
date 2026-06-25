# run_hls_retina_head.tcl
# Vitis HLS synthesis script — isolated retina_head_top module (W8A32 PTQ)
#
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Edit PART below if you use a different board.
#
# Run from the fpga_retina_head/ directory:
#   vitis_hls -f run_hls_retina_head.tcl
#
# ── Scaling up spatial dimensions ─────────────────────────────────────────
# Default: P6 = 10×10   (fastest CSIM, smallest arrays)
# To test P5 = 20×20, P4 = 40×40, etc., change TEST_H / TEST_W in CFLAGS below.
#
# Phased test plan:
#   Phase 1  TEST_H=10  TEST_W=10   P6  — verify basic computation, CSIM < 1 min
#   Phase 2  TEST_H=20  TEST_W=20   P5  — 4× more ops
#   Phase 3  TEST_H=40  TEST_W=40   P4  — 16× more ops
#   Phase 4  TEST_H=80  TEST_W=80   P3  — 64× more ops
#   Phase 5  TEST_H=160 TEST_W=160  P2  — full level (use fpga_ptq_192 CSIM instead)
#
# ── Key synthesis outputs ──────────────────────────────────────────────────
#   retina_head_prj/retina_head_200MHz/syn/report/retina_head_top_csynth.rpt

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
set SRC_DIR    "."
set PRJ_NAME   "retina_head_prj"
set SOL_NAME   "retina_head_200MHz"
set TOP_FUNC   "retina_head_top"
set PART       "xczu9eg-ffvb1156-2-e"
set CLK_NS     "5"   ;# 5 ns = 200 MHz

# Spatial size: edit -DTEST_H / -DTEST_W to step through phases
set TEST_H     "10"
set TEST_W     "10"

set CFLAGS "-I${SRC_DIR} -std=c++14 -DTEST_H=${TEST_H} -DTEST_W=${TEST_W}"

# ══════════════════════════════════════════════════════════════════════════
# 1. Create / open project
# ══════════════════════════════════════════════════════════════════════════
open_project $PRJ_NAME

# ══════════════════════════════════════════════════════════════════════════
# 2. Add source and testbench files
# ══════════════════════════════════════════════════════════════════════════
add_files "${SRC_DIR}/retina_head_top.cpp" \
    -cflags $CFLAGS

add_files -tb "${SRC_DIR}/testbench_retina_head.cpp" \
    -cflags $CFLAGS

# ══════════════════════════════════════════════════════════════════════════
# 3. Set top function
# ══════════════════════════════════════════════════════════════════════════
set_top $TOP_FUNC

# ══════════════════════════════════════════════════════════════════════════
# 4. Create solution, set device and clock
# ══════════════════════════════════════════════════════════════════════════
open_solution $SOL_NAME -flow_target vivado

set_part $PART

create_clock -period $CLK_NS -name default

# ══════════════════════════════════════════════════════════════════════════
# 5. C-simulation — uncomment to run before synthesis
#    Test 1 (synthetic) runs always.
#    Test 2 (real weights) needs binary files in ../ptq_results_192_ep47/.
# ══════════════════════════════════════════════════════════════════════════
csim_design -O

# ══════════════════════════════════════════════════════════════════════════
# 6. C-synthesis
# ══════════════════════════════════════════════════════════════════════════
csynth_design

# ══════════════════════════════════════════════════════════════════════════
# 7. Skip export for now — review the synthesis report first
# ══════════════════════════════════════════════════════════════════════════

puts "==================================================="
puts "retina_head_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/retina_head_top_csynth.rpt"
puts "==================================================="
