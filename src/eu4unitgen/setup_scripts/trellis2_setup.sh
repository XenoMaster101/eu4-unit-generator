#!/usr/bin/env bash
# TRELLIS.2 environment inside WSL2 (Ubuntu), SPEC M21. Run as the normal Linux user, after the root step:
#   sudo apt-get install -y build-essential gcc-13 g++-13 git libjpeg-dev
#   NVIDIA's wsl-ubuntu repo (cuda-keyring) + cuda-toolkit-12-9  (toolkit only; the driver is Windows')
#   sudo bash cuda_glibc_patch.sh   (glibc 2.43's noexcept math declarations vs CUDA 12.9's headers)
# Windows side, %USERPROFILE%\.wslconfig (32 GB host): [wsl2] memory=20GB, swap=16GB; [experimental]
# autoMemoryReclaim=dropCache (24 GB with no reclaim starved Windows), then wsl --shutdown.
# The env, build tree and uv cache live on the Linux disk (~/eu4unitgen_ai); the TRELLIS.2 checkout and every
# weight folder are reused from the Windows AI root through /mnt/d (nothing re-downloaded).
# CUDA 12.9, not 12.8: 12.8's math headers clash with glibc 2.41+ (cospi/sinpi/rsqrt noexcept); torch cu128
# loads extensions built by any 12.x nvcc. nvcc accepts gcc <= 14, so they build with gcc-13 (Ubuntu 26.04: 15).
set -euo pipefail

AI_WIN=${AI_WIN:-/mnt/d/eu4unitgen_ai}
AI=${AI:-$HOME/eu4unitgen_ai}
ENV=$AI/trellis2_env
EXT=$AI/ext
mkdir -p "$AI" "$EXT"
export UV_CACHE_DIR=$AI/uv_cache
export PATH=$HOME/.local/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LIBRARY_PATH=/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}   # libcuda.so (-lcuda) is the Windows driver's
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 NVCC_CCBIN=g++-13 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}   # RTX 3060
export MAX_JOBS=${MAX_JOBS:-3}   # 6 jobs were OOM-killed in WSL's default 15 GB

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
echo "== python 3.11"
[ -x "$ENV/bin/python" ] || uv venv --python 3.11 "$ENV"
PIP="uv pip install --python $ENV/bin/python"
echo "== torch"
$PIP torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PIP xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
echo "== basic"
# transformers 4.57: TRELLIS.2's DINOv3 feature extractor walks DINOv3ViTModel.layer, which 5.x moved
$PIP imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "transformers==4.57.1" zstandard kornia timm \
    setuptools wheel "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

clone() {  # url dir [branch]; records the commit built
    [ -d "$EXT/$2" ] || git clone --recursive ${3:+-b $3} "$1" "$EXT/$2"
    echo "$2 $(git -C "$EXT/$2" rev-parse HEAD)" >> "$AI/ext_commits.txt"
}
: > "$AI/ext_commits.txt"
echo "== extensions"
# No nvdiffrast or nvdiffrec (NVIDIA Source Code License, research or evaluation only; SPEC M21): the bake's
# rasteriser is ours (ai/jobs/_uv_raster.py), and nvdiffrec only serves TRELLIS.2's preview renderer.
clone https://github.com/JeffreyXiang/CuMesh.git CuMesh
clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM
rm -rf "$EXT/o-voxel" && cp -r "$AI_WIN/TRELLIS.2/o-voxel" "$EXT/o-voxel"
# the checkout's Eigen submodule is empty: fetch the commit TRELLIS.2 pins into the build copy only
EIGEN=$(git -C "$AI_WIN/TRELLIS.2" ls-tree HEAD o-voxel/third_party/eigen | awk '{print $3}')
[ -d "$EXT/eigen/.git" ] || git clone -q https://gitlab.com/libeigen/eigen.git "$EXT/eigen"
git -C "$EXT/eigen" checkout -q "$EIGEN" && echo "eigen $EIGEN" >> "$AI/ext_commits.txt"
rm -rf "$EXT/o-voxel/third_party/eigen" && cp -r "$EXT/eigen" "$EXT/o-voxel/third_party/eigen"
uv pip uninstall --python "$ENV/bin/python" nvdiffrast nvdiffrec_render 2>/dev/null || true
for spec in CuMesh:cumesh FlexGEMM:flex_gemm o-voxel:o_voxel; do
    d=${spec%%:*}
    echo "-- $d"
    if (cd /tmp && "$ENV/bin/python" -c "import ${spec#*:}" 2>/dev/null); then echo "   already built"; continue; fi
    jobs=$MAX_JOBS; [ "$d" = o-voxel ] && jobs=${OVOXEL_JOBS:-2}   # its units peak near 4 GB each
    MAX_JOBS=$jobs $PIP --no-build-isolation "$EXT/$d"
done

echo "== import check"
cd "$AI_WIN/TRELLIS.2"
JOBS=$(dirname "$(readlink -f "$0")")/../../src/eu4unitgen/ai/jobs   # run from the repo (*.sh are LF)
ATTN_BACKEND=xformers "$ENV/bin/python" -c "import sys; sys.path.insert(0, '$JOBS'); import _uv_raster; \
_uv_raster.install(); import torch, o_voxel, cumesh, flex_gemm; \
from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('ok', torch.__version__, torch.cuda.get_device_name(0))"
