# run_hls_nms.tcl
# Vitis HLS synthesis script — isolated NMS module (score-filter + sort + greedy NMS)
#
# Target: Zynq UltraScale+ ZCU102  (xczu9eg-ffvb1156-2-e)
#         Edit PART below for a different board.
#
# Run from the fpga_nms/ directory:
#   vitis_hls -f run_hls_nms.tcl
#
# ── What this synthesises ──────────────────────────────────────────────────
# Isolates the three sequential stages that follow the conv computation:
#   [1] score_filter  — score threshold + anchor decode (PIPELINE II=1 inner loop)
#   [2] sort_scores   — insertion sort, O(N²) bounded for-loops
#   [3] nms_greedy    — greedy multiclass NMS, O(N²) sequential
#
# This module takes cls_logits + reg_deltas as direct inputs, NOT FPN features.
# Use fpga_retina_head to validate the conv → logits/deltas path separately.
#
# ── Phase scaling ──────────────────────────────────────────────────────────
# Default: P6 (10×10) — fastest CSIM, smallest candidate buffer.
# Edit TEST_H / TEST_W / TEST_STRIDE below to advance phases:
#
#   Phase 1  H=10  W=10  stride=64  P6  CAND_BUF=3 600
#   Phase 2  H=20  W=20  stride=32  P5  CAND_BUF=14 400
#   Phase 3  H=40  W=40  stride=16  P4  CAND_BUF=57 600
#   Phase 4  H=80  W=80  stride= 8  P3  CAND_BUF=230 400   (slow sort!)
#
# ── Expected synthesis results ─────────────────────────────────────────────
# • score_filter inner loop: II=1  (target; may be II=2 due to expf)
# • sort_scores: no II target (sequential bounded for-loops, O(N²) latency)
# • nms_greedy outer loop: sequential (suppressed[] inter-iter dependency)
# • nms_greedy inner loop: target II=1 (iou_f is purely combinational)
#
# ── Outputs ────────────────────────────────────────────────────────────────
#   nms_prj/nms_200MHz/syn/report/nms_top_csynth.rpt

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
set SRC_DIR    "."
set PRJ_NAME   "nms_prj"
set SOL_NAME   "nms_200MHz"
set TOP_FUNC   "nms_top"
set PART       "xczu9eg-ffvb1156-2-e"
set CLK_NS     "6.5" ;# 6.5 ns = 154 MHz (relaxed from 5 ns to close fexp timing)

# Spatial size + stride — edit to step through phases
set TEST_H      "10"
set TEST_W      "10"
set TEST_STRIDE "64"

set CFLAGS "-I${SRC_DIR} -std=c++14 \
    -DTEST_H=${TEST_H} -DTEST_W=${TEST_W} -DTEST_STRIDE=${TEST_STRIDE}"

# ══════════════════════════════════════════════════════════════════════════
# 1. Create / open project
# ══════════════════════════════════════════════════════════════════════════
open_project $PRJ_NAME

# ══════════════════════════════════════════════════════════════════════════
# 2. Add source and testbench files
# ══════════════════════════════════════════════════════════════════════════
add_files "${SRC_DIR}/nms_top.cpp" \
    -cflags $CFLAGS

add_files -tb "${SRC_DIR}/testbench_nms.cpp" \
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
# 5. C-simulation
# All 4 tests should pass.  Test 3 validates NMS IoU suppression.
# ══════════════════════════════════════════════════════════════════════════
csim_design -O

# ══════════════════════════════════════════════════════════════════════════
# 6. C-synthesis
# Key things to check in the report:
#   • score_filter / SF_C loop: II=1 or II=2 (expf latency)
#   • sort_scores: II won't be 1; look at total latency estimate
#   • nms_greedy NMS_I: no pipeline (inter-iter dependency); note II
#   • nms_greedy NMS_J: target II=1 (check if iou_f inlined correctly)
#   • BRAM usage: cand_boxes/scores/cls/order + suppressed[]
# ══════════════════════════════════════════════════════════════════════════
csynth_design

puts "==================================================="
puts "nms_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/nms_top_csynth.rpt"
puts ""
puts "Key metrics to report back:"
puts "  score_filter SF_C loop II"
puts "  sort_scores total latency (cycles)"
puts "  nms_greedy NMS_I/NMS_J latency"
puts "  Total BRAM / DSP / LUT usage"
puts "==================================================="
