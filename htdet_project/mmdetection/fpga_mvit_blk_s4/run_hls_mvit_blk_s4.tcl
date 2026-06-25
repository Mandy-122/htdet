# run_hls_mvit_blk_s4.tcl — Full MobileViT Block Stage-4
set PRJ "mvit_blk_s4_prj"; set SOL "mvit_blk_s4_200MHz"; set TOP "mvit_blk_s4_top"
set PART "xczu28dr-ffvg1517-2-e"; set CLK "5"
open_project $PRJ
add_files "mvit_blk_s4_top.cpp" -cflags "-I. -std=c++14"
add_files -tb "testbench_mvit_blk_s4.cpp" -cflags "-I. -std=c++14"
set_top $TOP; open_solution $SOL -flow_target vivado; set_part $PART
create_clock -period $CLK -name default; csim_design -O; csynth_design
puts "=== mvit_blk_s4_top synthesis complete ==="
