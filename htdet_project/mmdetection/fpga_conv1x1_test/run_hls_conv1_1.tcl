# run_hls_conv1_1.tcl
# Standalone synthesis for conv1x1_bn_silu with tiled IC (TILE=16).
# Tiny dims (in_ch=32, out_ch=64, H=8, W=8) -> csim + csynth finish in minutes.
#
# Run from fpga_conv1x1_test/:
#   vitis_hls -f run_hls_conv1_1.tcl

set SRC_DIR  "."
set PRJ_NAME "conv1_1_prj"
set SOL_NAME "conv1_1_200MHz"
set TOP_FUNC "conv1x1_top"
set PART     "xczu28dr-ffvg1517-2-e"
set CLK_NS   "5"

open_project $PRJ_NAME

add_files "${SRC_DIR}/conv1_1_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_conv1_1.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

set_top $TOP_FUNC

open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default

csim_design -O

csynth_design

puts "==================================================="
puts "conv1x1_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/conv1x1_top_csynth.rpt"
puts "==================================================="
