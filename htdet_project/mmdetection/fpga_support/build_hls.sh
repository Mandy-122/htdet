#!/usr/bin/env bash
# build_hls.sh
# One-shot build script for the HTDet Vitis HLS flow.
#
# Steps:
#   1. Optionally export weights from a PyTorch checkpoint.
#   2. Run Vitis HLS (C sim → csynth → IP export).
#   3. Optionally run Vitis v++ to link the IP into a .xclbin
#      (requires a Vitis platform — skip on pure HLS hosts).
#
# Prerequisite: source Vitis settings before calling this script.
#   source /tools/Xilinx/Vitis/2022.2/settings64.sh
#
# Usage:
#   ./build_hls.sh [--checkpoint <path.pth>] [--skip-weights] [--skip-vpp]
#
# Outputs:
#   weights/           — quantised weight binaries
#   htdet_hls_prj/     — HLS project + IP
#   htdet.xclbin       — (optional) FPGA bitstream for Zynq

set -euo pipefail

# ============================================================
# Defaults
# ============================================================
CHECKPOINT=""
SKIP_WEIGHTS=0
SKIP_VPP=1           # v++ link disabled by default (needs a Vitis platform)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTL_SRC_DIR="${SCRIPT_DIR}/../fpga_rtl_src"
WEIGHTS_DIR="${SCRIPT_DIR}/../weights"
PLATFORM=""          # e.g. xilinx_zcu102_base_202220_1

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --skip-vpp)     SKIP_VPP=1; shift ;;
        --vpp)          SKIP_VPP=0; shift ;;
        --platform)     PLATFORM="$2"; shift 2 ;;
        --weights-dir)  WEIGHTS_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--checkpoint <pth>] [--skip-weights] [--vpp --platform <name>]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ============================================================
# Check Vitis HLS is on PATH
# ============================================================
if ! command -v vitis_hls &>/dev/null; then
    echo "ERROR: vitis_hls not found. Source Vitis settings64.sh first."
    echo "  source /tools/Xilinx/Vitis/<version>/settings64.sh"
    exit 1
fi
echo "Using: $(which vitis_hls)"
vitis_hls --version | head -2

# ============================================================
# Step 1: Export weights
# ============================================================
if [[ $SKIP_WEIGHTS -eq 0 ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
        echo ""
        echo "WARNING: --checkpoint not provided. Skipping weight export."
        echo "         HLS csim will run with zero weights (architecture test only)."
    else
        echo ""
        echo "=== Exporting weights from $CHECKPOINT ==="
        mkdir -p "$WEIGHTS_DIR"
        python3 "${SCRIPT_DIR}/export_weights.py" \
            --checkpoint "$CHECKPOINT" \
            --outdir     "$WEIGHTS_DIR"
        echo "Weights written to $WEIGHTS_DIR"
        echo ""
    fi
fi

# ============================================================
# Step 2: Run Vitis HLS
# ============================================================
echo "=== Running Vitis HLS ==="
cd "$SCRIPT_DIR"
vitis_hls -f prj_config.tcl 2>&1 | tee build_hls.log
echo ""
echo "HLS log: ${SCRIPT_DIR}/build_hls.log"

# Check for synthesis errors
if grep -q "ERROR:" build_hls.log; then
    echo ""
    echo "=== HLS ERRORS DETECTED ==="
    grep "ERROR:" build_hls.log
    exit 1
fi

echo ""
echo "=== HLS synthesis completed successfully ==="
echo "IP location: ${SCRIPT_DIR}/htdet_hls_prj/htdet_200MHz/htdet_ip"

# ============================================================
# Step 3: (Optional) v++ link to generate .xclbin
# ============================================================
if [[ $SKIP_VPP -eq 0 ]]; then
    if [[ -z "$PLATFORM" ]]; then
        echo "ERROR: --platform required for v++ link. Example:"
        echo "  --platform xilinx_zcu102_base_202220_1"
        exit 1
    fi

    echo ""
    echo "=== Running v++ link (platform: $PLATFORM) ==="

    # Connectivity file routes the kernel's AXI masters to DDR banks.
    # For Zynq, all bundles share DDR (single bank); edit for PCIe boards.
    cat > /tmp/htdet_connectivity.cfg << 'EOF'
[connectivity]
nk=htdet_inference:1:htdet_inference_1
sp=htdet_inference_1.m_axi_gmem0:HP0
sp=htdet_inference_1.m_axi_gmem1:HP1
sp=htdet_inference_1.m_axi_gmem2:HP2
sp=htdet_inference_1.m_axi_gmem3:HP3
sp=htdet_inference_1.m_axi_gmem4:HP0
sp=htdet_inference_1.m_axi_gmem5:HP1
sp=htdet_inference_1.m_axi_gmem6:HP2
sp=htdet_inference_1.m_axi_gmem7:HP3
EOF

    v++ -l \
        --platform "$PLATFORM" \
        --config /tmp/htdet_connectivity.cfg \
        --temp_dir _vpp_tmp \
        --log_dir  _vpp_log \
        --report_dir _vpp_report \
        -t hw \
        -o htdet.xclbin \
        "${SCRIPT_DIR}/htdet_hls_prj/htdet_200MHz/htdet_ip/htdet_inference.xo" \
        2>&1 | tee build_vpp.log

    echo ""
    echo "xclbin written to: ${SCRIPT_DIR}/htdet.xclbin"
fi

echo ""
echo "=== Build complete ==="
