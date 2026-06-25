# run_hls_stem.tcl — Vitis HLS: MobileViT stem conv3×3
# Run from fpga_stem/: vitis_hls -f run_hls_stem.tcl

set SRC_DIR  "."
set PRJ_NAME "stem_prj"
set SOL_NAME "stem_200MHz"
set TOP_FUNC "stem_top"
set PART     "xczu28dr-ffvg1517-2-e"
set CLK_NS   "5"

open_project $PRJ_NAME
add_files "${SRC_DIR}/stem_top.cpp" -cflags "-I${SRC_DIR} -std=c++14"
add_files -tb "${SRC_DIR}/testbench_stem.cpp" -cflags "-I${SRC_DIR} -std=c++14"
set_top $TOP_FUNC
open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default
csim_design -O
csynth_design
puts "=== stem_top synthesis complete ==="
