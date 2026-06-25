# run_hls_mbconv_s0.tcl — Stage-0 MBConv (16→32, s=1, 160×160)
# Run from fpga_mbconv_s0/: vitis_hls -f run_hls_mbconv_s0.tcl

set PRJ  "mbconv_s0_prj"
set SOL  "mbconv_s0_200MHz"
set TOP  "mbconv_s0_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "5"

open_project $PRJ
add_files "mbconv_s0_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mbconv_s0.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== mbconv_s0_top synthesis complete ==="
puts "Report: ${PRJ}/${SOL}/syn/report/mbconv_s0_top_csynth.rpt"
