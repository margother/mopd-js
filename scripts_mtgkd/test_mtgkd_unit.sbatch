#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --time=1:00:00
#SBATCH --qos=gpu-debug
#SBATCH -p a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=outputs/test-mtgkd-unit-%j.out
#SBATCH --error=errors/test-mtgkd-unit-%j.err

# Smoke test of the environment: runs the multi-teacher GKD unit tests (tiny models from the HF Hub)
# and the upstream single-teacher GKD tests with the venv and the local mopd-js clone.

set -euo pipefail

export VENV_PATH="${VENV_PATH:-/mnt/scratch-artemis/margot/venvs/mtgkd}"
export TRL_REPO="${TRL_REPO:-/mnt/home/margot/mopd-js}"
export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/${USER}/.hf}"
export TMPDIR="${TMPDIR:-/mnt/scratch-artemis/${USER}/tmp}"
export PYTHONNOUSERSITE=1
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TMPDIR}"

# Some tests build their inputs on CPU; CPU_ONLY=True hides the GPU so they run as written.
if [[ "${CPU_ONLY:-False}" == "True" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

source "${VENV_PATH}/bin/activate"
cd "${TRL_REPO}"

echo "[INFO] host=$(hostname) commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
python -c 'import torch, transformers, trl; print(f"[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__} trl={trl.__file__}")'

# PYTEST_K: optional pytest -k expression to run a subset of tests.
python -m pytest -q -p no:cacheprovider ${PYTEST_K:+-k "${PYTEST_K}"} \
  tests/experimental/test_multi_teacher_gkd_trainer.py \
  tests/experimental/test_gkd_trainer.py
