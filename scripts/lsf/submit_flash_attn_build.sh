#!/bin/bash
# Compile flash-attn 2.8.3 from source inside a CPU LSF job (no prebuilt wheel
# exists for torch 2.11/cu13; login node must not run heavy builds).
# Prereqs (done on login node): alab-rl has torch+vllm, cuda-nvcc 13.0 via conda,
# ninja/packaging/psutil via pip, and the sdist in third_party/wheels/.
# After this job is green, finish the env on the login node with:
#   conda run -n alab-rl pip install "openrlhf[vllm]==0.10.4" math-verify hf_transfer
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDIST="${REPO}/third_party/wheels/flash_attn-2.8.3.tar.gz"
LOG_DIR="${REPO}/results/logs/build"
mkdir -p "${LOG_DIR}"

if [ ! -f "${SDIST}" ]; then
  echo "Missing ${SDIST} — run on login node first:" >&2
  echo "  conda run -n alab-rl pip download flash-attn==2.8.3 --no-deps --no-build-isolation -d third_party/wheels" >&2
  exit 2
fi

# sm80 = A100, sm90 = H100 — restricting archs cuts compile time drastically.
# CUDA toolkit came in via vllm's pip deps (nvidia/cu13 tree holds bin/nvcc + headers).
CUDA13=/u/arushh/miniconda3/envs/alab-rl/lib/python3.11/site-packages/nvidia/cu13
# NVCC_THREADS=1: nvcc --threads has a temp-file race (missing .cpp1.ii) — parallelize
# via MAX_JOBS (ninja) only. Dedicated TMPDIR isolates intermediates on node-local disk.
CMD="cd ${REPO} && \
export CUDA_HOME=${CUDA13} PATH=${CUDA13}/bin:\$PATH && \
export TMPDIR=/tmp/fa_build_\${LSB_JOBID} && mkdir -p \$TMPDIR && trap 'rm -rf /tmp/fa_build_\${LSB_JOBID}' EXIT && \
export MAX_JOBS=16 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=\"80;90\" && \
conda run -n alab-rl pip install --no-build-isolation -v ${SDIST}"

echo "log_path=${LOG_DIR}/flash_attn_build.%J.out"
bsub -q normal \
     -J alab_flash_attn_build \
     -n 16 \
     -R "rusage[mem=128G]" \
     -R "span[hosts=1]" \
     -W 06:00 \
     -o "${LOG_DIR}/flash_attn_build.%J.out" \
     -e "${LOG_DIR}/flash_attn_build.%J.err" \
     bash -lc "${CMD}"
