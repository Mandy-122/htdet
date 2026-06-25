# run_hls_mbconv20.tcl
# C-synthesis for mbconv_20: in=128, out=128, expand=4, stride=1, 20x20
# Run from fpga_mbconv_20/:  vitis_hls -f run_hls_mbconv20.tcl

set PRJ  "mbconv_20_prj"
set SOL  "mbconv_20_200MHz"
set TOP  "mbconv_20_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "5"

open_project $PRJ
add_files "mbconv_20_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mbconv20.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== mbconv_20_top synthesis complete ==="
puts "Report: ${PRJ}/${SOL}/syn/report/mbconv_20_top_csynth.rpt"
