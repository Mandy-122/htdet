# run_hls_c4_expand.tcl — Final backbone conv1×1 expansion (160→640, 10×10)
# Run from fpga_c4_expand/: vitis_hls -f run_hls_c4_expand.tcl

set PRJ  "c4_expand_prj"
set SOL  "c4_expand_200MHz"
set TOP  "c4_expand_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "5"

open_project $PRJ
add_files "c4_expand_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_c4_expand.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== c4_expand_top synthesis complete ==="
