# run_hls_mbconv_s2a.tcl — Stage-2a MBConv (64→96, s=2, 80×80→40×40)
# Run from fpga_mbconv_s2a/: vitis_hls -f run_hls_mbconv_s2a.tcl

set PRJ  "mbconv_s2a_prj"
set SOL  "mbconv_s2a_200MHz"
set TOP  "mbconv_s2a_top"
set PART "xczu28dr-ffvg1517-2-e"
set CLK  "5"

open_project $PRJ
add_files "mbconv_s2a_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mbconv_s2a.cpp" -cflags "-I. -std=c++14"
set_top $TOP
open_solution $SOL -flow_target vivado
set_part $PART
create_clock -period $CLK -name default
csim_design -O
csynth_design
puts "=== mbconv_s2a_top synthesis complete ==="
puts "Report: ${PRJ}/${SOL}/syn/report/mbconv_s2a_top_csynth.rpt"
