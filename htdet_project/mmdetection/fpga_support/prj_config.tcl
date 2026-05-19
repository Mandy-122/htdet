# prj_config.tcl
# Vitis HLS project configuration and synthesis script for HTDet.
#
# Run with:  vitis_hls -f prj_config.tcl
#
# What this script does:
#   1. Creates a new HLS project (or reopens if it exists).
#   2. Adds all RTL source files from fpga_rtl_src/.
#   3. Sets the top function to htdet_inference.
#   4. Selects the target device (xczu9eg).
#   5. Creates a 200 MHz solution.
#   6. Runs C simulation (csim), C synthesis (csynth), and RTL export.
#
# After this script completes, the synthesised IP is in:
#   htdet_hls_prj/htdet_200MHz/impl/ip/
# which can be imported into Vivado as a packaged IP.

# ============================================================
# Paths — edit if your directory layout differs
# ============================================================
set RTL_SRC_DIR  "../fpga_rtl_src"
set PRJ_NAME     "htdet_hls_prj"
set SOL_NAME     "htdet_200MHz"
set TOP_FUNC     "htdet_inference"
set PART         "xczu9eg-ffvb1156-2-e"
set CLOCK_PERIOD "5"   ;# 5 ns = 200 MHz

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source files
#    htdet_top.cpp includes the other headers via #include,
#    so we only need to add it as the single compilation unit.
#    testbench.cpp is the C simulation testbench.
# ============================================================
add_files "${RTL_SRC_DIR}/htdet_top.cpp" \
    -cflags "-I${RTL_SRC_DIR} -std=c++14"

add_files -tb "${RTL_SRC_DIR}/testbench.cpp" \
    -cflags "-I${RTL_SRC_DIR} -std=c++14"

# ============================================================
# 3. Set top function
# ============================================================
set_top $TOP_FUNC

# ============================================================
# 4. Create solution and set target device / clock
# ============================================================
open_solution $SOL_NAME -flow_target vivado

set_part $PART

# Clock: 5 ns period, 12.5% uncertainty (tolerated by HLS)
create_clock -period $CLOCK_PERIOD -name default

# ============================================================
# 5. (Optional) C simulation
#    Runs testbench.cpp with zero weights / checkerboard image.
#    Comment out to skip and go straight to csynth.
# ============================================================
csim_design -O

# ============================================================
# 6. C synthesis
# ============================================================
csynth_design

# ============================================================
# 7. (Optional) C/RTL co-simulation
#    This takes long; enable when verifying RTL correctness.
# ============================================================
# cosim_design -O -rtl verilog

# ============================================================
# 8. Export synthesised IP as a Vivado IP package
#    The generated .zip can be imported as a Repository in Vivado.
# ============================================================
export_design -flow impl -rtl verilog -format ip_catalog \
    -description "HTDet FPGA inference kernel" \
    -vendor "htdet" \
    -version "1.0" \
    -output "${PRJ_NAME}/${SOL_NAME}/htdet_ip"

puts "==================================================="
puts "HLS synthesis complete."
puts "IP package: ${PRJ_NAME}/${SOL_NAME}/htdet_ip"
puts "==================================================="
