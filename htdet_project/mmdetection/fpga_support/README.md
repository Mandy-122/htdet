# HTDet FPGA Support Files

This folder contains everything needed to run the HTDet model on an FPGA
**except** the HLS kernel sources (those live in `fpga_rtl_src/`).

## Folder contents

| File | Purpose |
|------|---------|
| `export_weights.py` | Export PyTorch checkpoint → Q8.8 binary files |
| `weights_loader.h`  | C utility to load binaries into host memory |
| `host_app.cpp`      | Zynq PS host application (XRT API, launches kernel) |
| `prj_config.tcl`    | Vitis HLS project setup + csynth + IP export script |
| `build_hls.sh`      | One-shot build script (weights → HLS → optional v++ link) |

## End-to-end flow

### 1. Export weights
```bash
python3 export_weights.py \
    --checkpoint ../work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \
    --outdir     ../weights
```
Outputs: `../weights/{backbone,fpn,cls_conv,reg_conv,cls_pred_w,cls_pred_b,reg_pred_w,reg_pred_b}.bin` + `sizes.txt`

### 2. Run Vitis HLS synthesis
```bash
source /tools/Xilinx/Vitis/2022.2/settings64.sh
./build_hls.sh --checkpoint ../work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth
```
- Runs C simulation (zero-weights architecture check).
- Runs `csynth_design` targeting `xczu9eg-ffvb1156-2-e` at 200 MHz.
- Exports packaged IP to `htdet_hls_prj/htdet_200MHz/htdet_ip/`.

### 3. (Optional) Generate xclbin with v++
```bash
./build_hls.sh --skip-weights --vpp --platform xilinx_zcu102_base_202220_1
```

### 4. Run on-board
```bash
# Cross-compile or build natively on the Zynq board
g++ -O2 -std=c++14 host_app.cpp -o htdet_host \
    -lxrt_coreutil -luuid -pthread \
    -I${XILINX_XRT}/include -L${XILINX_XRT}/lib

./htdet_host \
    --xclbin  htdet.xclbin \
    --weights ../weights   \
    --image   test_image.bin \
    --thresh  0.3 \
    --out     results.txt
```

## Image binary format
`float32`, CHW layout, `3 × 320 × 320`, normalised to `[-1, 1]` using
ImageNet mean/std (same as the MMDetection training pipeline).

To convert a JPEG/PNG:
```python
import torch, torchvision.transforms as T, numpy as np
from PIL import Image

img = Image.open("test.jpg").convert("RGB").resize((320, 320))
t = T.Compose([T.ToTensor(),
               T.Normalize(mean=[0.485,0.456,0.406],
                           std=[0.229,0.224,0.225])])
arr = t(img).numpy().astype(np.float32)   # shape (3,320,320)
arr.tofile("test_image.bin")
```

## Class IDs
| ID | Class |
|----|-------|
| 0  | holothurian |
| 1  | echinus |
| 2  | scallop |
| 3  | starfish |
