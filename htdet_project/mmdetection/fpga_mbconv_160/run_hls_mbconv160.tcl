# run_hls_mbconv160.tcl
# C-synthesis for mbconv_160: in=32, out=64, expand=4, stride=2, 160x160
# Run from fpga_mbconv_160/:  vitis_hls -f run_hls_mbconv160.tcl

set PRJ  "mbconv_160_prj"
set SOL  "mbconv_160_222MHz_OC8"
set TOP  "mbconv_160_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "4.5"

open_project $PRJ
add_files "mbconv_160_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mbconv160.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== mbconv_160_top synthesis complete ==="
puts "Report: ${PRJ}/${SOL}/syn/report/mbconv_160_top_csynth.rpt"
