"""Run W8A8 PTQ on the 192-channel model."""
import sys, os
sys.path.insert(0, '/workspace/ckarfa/htdet/htdet_project/mmdetection')
os.chdir('/workspace/ckarfa/htdet/htdet_project/mmdetection')

import fpga_support.ptq_w8a8 as w
w.CONFIG = 'configs/htdet/htdet_gpu_low_gflops_192.py'

sys.argv = [
    'ptq_w8a8.py',
    '--checkpoint', 'work_dirs/htdet_low_gflops_192/latest.pth',
    '--num-cal', '200',
    '--num-det', '50',
    '--outdir',  'w8a8/ptq_results_192',
    '--save-checkpoint'
]
w.main()
