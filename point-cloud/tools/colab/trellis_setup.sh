#!/bin/bash
# Colab VM 에서 TRELLIS.2 환경 만들기 (공식 setup.sh 를 micromamba Python 3.10 · CUDA 12.4 환경 안에서). 기록: /content/setup.log
set -x
cd /content
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=/content/mamba
./bin/micromamba create -y -n trellis2 -c conda-forge -c nvidia/label/cuda-12.4.1 python=3.10 cuda-nvcc cuda-toolkit git || exit 1
eval "$(./bin/micromamba shell hook -s bash)"; micromamba activate trellis2
export CUDA_HOME=$CONDA_PREFIX
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || exit 2
ABI=$(python -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl" || echo "flash-attn 휠 실패"
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive || exit 3
cd TRELLIS.2
. ./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
python -c "import torch, flash_attn, nvdiffrast, o_voxel; print('OK', torch.__version__, torch.cuda.get_device_name(0))"
echo SETUP_DONE
