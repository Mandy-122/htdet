# run_hls_mvit_blk_s2.tcl — Full MobileViT Block Stage-2
# Run from fpga_mvit_blk_s2/: vitis_hls -f run_hls_mvit_blk_s2.tcl
#
# ⚠ Large module: expect 15-30 min synthesis time.
# Transformer MHSA (N=400 tokens) dominates latency.
# Transformer contributes 94M cycles (470ms); local convs ~20-50M more.

set PRJ  "mvit_blk_s2_prj"
set SOL  "mvit_blk_s2_200MHz"
set TOP  "mvit_blk_s2_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "5"

open_project $PRJ
add_files "mvit_blk_s2_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mvit_blk_s2.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== mvit_blk_s2_top synthesis complete ==="
