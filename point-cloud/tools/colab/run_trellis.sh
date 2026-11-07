#!/bin/bash
export MAMBA_ROOT_PREFIX=/content/mamba
eval "$(/content/bin/micromamba shell hook -s bash)"; micromamba activate trellis2
export CUDA_HOME=$CONDA_PREFIX
cd /content/TRELLIS.2
python /content/run_trellis.py /content/input_120.png /content/out_120.glb
echo RUN_DONE
