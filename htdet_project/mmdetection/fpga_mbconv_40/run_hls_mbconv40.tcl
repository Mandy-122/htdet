# run_hls_mbconv40.tcl
# Vitis HLS synthesis script — isolated mbconv_40 module (PTQ INT8, W8A32)
#
# Config  : in_ch=96, out_ch=128, expand=4, stride=2
# Spatial : C2 (40×40 in, 20×20 out) for 320×320 input
# Target  : xczu28dr-ffvg1517-2-e @ 5 ns (200 MHz)
#
# Run from the fpga_ptq_192_test/ directory:
#   vitis_hls -f run_hls_mbconv40.tcl
#
# Lessons from mbconv_80 applied:
#   - ex_buf / dw_buf → URAM (no cyclic partition; avoids BRAM overflow)
#   - w_conv cyclic factor=192 → 384 ports → proj IC=384 II=1
#   - in     cyclic factor=48  → 96  ports → expand IC=96 II=1
#   - 5 ns clock (sitofp=3.24ns fits within 4.46ns effective budget)
#
# Expected resources (xczu28dr: 576 URAM, 984 BRAM_18K):
#   URAM  : ~35-40% (ex_buf + dw_buf, no res_buf — stride=2 has no residual)
#   BRAM  : ~25-30% (w_conv factor=192 + in factor=48 + interface)
#   LUT   : ~20-25%
#   Timing: PASS (no sitofp violation at 5 ns)

# ============================================================
# Configuration
# ============================================================
set SRC_DIR  "."
set PRJ_NAME "mbconv40_prj"
set SOL_NAME "mbconv40_200MHz"
set TOP_FUNC "mbconv_40_top"
set PART     "xczu28dr-ffvg1517-2-e"
set CLK_NS   "5"

# ============================================================
# 1. Create / open project
# ============================================================
open_project $PRJ_NAME

# ============================================================
# 2. Add source files
# ============================================================
add_files "${SRC_DIR}/mbconv_40_top.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

add_files -tb "${SRC_DIR}/testbench_mbconv40.cpp" \
    -cflags "-I${SRC_DIR} -std=c++14"

# ============================================================
# 3. Set top function
# ============================================================
set_top $TOP_FUNC

# ============================================================
# 4. Create solution
# ============================================================
open_solution $SOL_NAME -flow_target vivado
set_part $PART
create_clock -period $CLK_NS -name default

# ============================================================
# 5. C-simulation (uncomment to verify before synth)
# ============================================================
# csim_design -O

# ============================================================
# 6. C-synthesis
# ============================================================
csynth_design

puts "==================================================="
puts "mbconv_40_top synthesis complete."
puts "Report: ${PRJ_NAME}/${SOL_NAME}/syn/report/mbconv_40_top_csynth.rpt"
puts "==================================================="
