#!/bin/bash
#SBATCH --job-name=single-gkd-qwen25-2gpu
#SBATCH --time=2-00:00:00
#SBATCH --qos=gpu-medium
#SBATCH -p a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=outputs/single-gkd-qwen25-%j.out
#SBATCH --error=errors/single-gkd-qwen25-%j.err

set -euo pipefail

# Optional run config (first argument or CONFIG_FILE); its values override the defaults below.
CONFIG_FILE="${1:-${CONFIG_FILE:-}}"
if [[ -n "${CONFIG_FILE}" ]]; then
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Config file does not exist: ${CONFIG_FILE}" >&2
    exit 1
  fi
  echo "[INFO] Loading config: ${CONFIG_FILE}"
  source "${CONFIG_FILE}"
fi

if [[ "${USE_MODULE_LOAD:-False}" == "True" ]]; then
  module load "${PYTHON_MODULE:-python/3.12.5}"
fi

export VENV_PATH="${VENV_PATH:-/mnt/scratch-artemis/margot/venvs/mtgkd}"
export TRL_REPO="${TRL_REPO:-/mnt/home/margot/mopd-js}"
export EXPECTED_TRL_BRANCH="${EXPECTED_TRL_BRANCH:-}"
export RUN_ROOT="${RUN_ROOT:?RUN_ROOT must be set.}"
export DATASET_PATH="${DATASET_PATH:?DATASET_PATH must be set.}"
export TEACHER_MODEL="${TEACHER_MODEL:?TEACHER_MODEL must be set.}"

export BASE_MODEL="${BASE_MODEL:-/mnt/scratch-artemis/diogomiranda/Qwen/Qwen2.5-1.5B}"

export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/${USER}/.hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-/mnt/scratch-artemis/${USER}/.torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/scratch-artemis/${USER}/.cache}"
export TMPDIR="${TMPDIR:-/mnt/scratch-artemis/${USER}/tmp}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_EXPERIMENTAL_SILENCE=1
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "${RUN_ROOT}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${SLURM_SUBMIT_DIR:-$(pwd)}/outputs" \
  "${SLURM_SUBMIT_DIR:-$(pwd)}/errors"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Venv activate script does not exist: ${VENV_PATH}/bin/activate" >&2
  ls -la "${VENV_PATH}" "${VENV_PATH}/bin" >&2 || true
  exit 1
fi

if [[ ! -d "${TRL_REPO}" ]]; then
  echo "[ERROR] TRL_REPO does not exist: ${TRL_REPO}" >&2
  exit 1
fi

for path in "${BASE_MODEL}" "${TEACHER_MODEL}" "${DATASET_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing path: ${path}" >&2
    exit 1
  fi
done

source "${VENV_PATH}/bin/activate"
hash -r

if [[ -x "${VENV_PATH}/bin/python3" ]]; then
  VENV_PYTHON="${VENV_PATH}/bin/python3"
else
  VENV_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${VENV_PYTHON}" || ! -x "${VENV_PYTHON}" ]]; then
  echo "[ERROR] python3 was not found after activating ${VENV_PATH}" >&2
  exit 1
fi
echo "[INFO] python3=${VENV_PYTHON}"
"${VENV_PYTHON}" -c 'import sys; print("[INFO] executable=" + sys.executable)'

export PYTHONPATH="${TRL_REPO}:${PYTHONPATH:-}"
cd "${TRL_REPO}"

echo "[INFO] TRL_REPO=${TRL_REPO}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_TRL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  CURRENT_TRL_COMMIT="$(git rev-parse --short HEAD)"
  echo "[INFO] TRL branch=${CURRENT_TRL_BRANCH} commit=${CURRENT_TRL_COMMIT}"
  if [[ -n "${EXPECTED_TRL_BRANCH}" && "${CURRENT_TRL_BRANCH}" != "${EXPECTED_TRL_BRANCH}" ]]; then
    echo "[ERROR] Expected TRL branch ${EXPECTED_TRL_BRANCH}, got ${CURRENT_TRL_BRANCH}" >&2
    echo "[ERROR] Set EXPECTED_TRL_BRANCH='' to skip this check, or set TRL_REPO to the right checkout." >&2
    exit 1
  fi
fi

"${VENV_PYTHON}" - <<'PY'
import trl
from trl.experimental.gkd import GKDConfig, GKDTrainer

print(f"[INFO] trl imported from: {trl.__file__}")
print(f"[INFO] GKDConfig: {GKDConfig}")
print(f"[INFO] GKDTrainer: {GKDTrainer}")
PY

TRAIN_SCRIPT="${TMPDIR}/train_single_teacher_gkd_qwen25_${SLURM_JOB_ID:-manual}.py"

cat > "${TRAIN_SCRIPT}" <<'PY'
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from trl.experimental.gkd import GKDConfig, GKDTrainer


base_model = os.environ["BASE_MODEL"]
teacher_model = os.environ["TEACHER_MODEL"]
dataset = load_from_disk(os.environ["DATASET_PATH"])

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_prompt_completion(example, tokenizer, max_length):
    if "input_ids" in example:
        return example
    if "prompt" not in example or "completion" not in example:
        raise ValueError("Dataset examples must contain either input_ids or prompt/completion columns.")

    completion = example["completion"]
    if tokenizer.eos_token is not None and not completion.endswith(tokenizer.eos_token):
        completion = completion + tokenizer.eos_token

    prompt_ids = tokenizer(text=example["prompt"], add_special_tokens=False).input_ids
    completion_ids = tokenizer(text=completion, add_special_tokens=False).input_ids

    if max_length is not None and len(prompt_ids) + len(completion_ids) > max_length:
        if completion_ids and max_length > 1:
            max_prompt_tokens = min(len(prompt_ids), max_length - 1)
            prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
            completion_ids = completion_ids[: max_length - len(prompt_ids)]
        else:
            prompt_ids = prompt_ids[-max_length:]
            completion_ids = []

    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "prompts": prompt_ids,
    }


def prepare_dataset(dataset, tokenizer, max_length):
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split_name: prepare_dataset(split, tokenizer, max_length)
                for split_name, split in dataset.items()
            }
        )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    first_example = dataset[0]
    if "input_ids" in first_example:
        return dataset
    return dataset.map(
        tokenize_prompt_completion,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        desc="Tokenizing prompt-completion dataset",
        num_proc=int(os.environ.get("DATASET_NUM_PROC", "1")),
    )


max_length = int(os.environ.get("MAX_LENGTH", "512"))
dataset = prepare_dataset(dataset, tokenizer, max_length)

if isinstance(dataset, DatasetDict):
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else None
else:
    train_dataset = dataset
    eval_dataset = None

use_bf16 = os.environ.get("USE_BF16", "False") == "True"
teacher_dtype = "bfloat16" if use_bf16 else "float16"
student_dtype = os.environ.get("STUDENT_DTYPE", "bfloat16" if use_bf16 else "float32")
use_tf32 = os.environ.get("USE_TF32", "False") == "True"

args = GKDConfig(
    output_dir=os.environ["RUN_ROOT"],
    teacher_model_name_or_path=teacher_model,
    model_init_kwargs={"dtype": student_dtype},
    teacher_model_init_kwargs={"dtype": teacher_dtype},
    bf16=use_bf16,
    fp16=not use_bf16,
    tf32=use_tf32,
    per_device_train_batch_size=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")),
    per_device_eval_batch_size=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "1")),
    gradient_accumulation_steps=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8")),
    gradient_checkpointing=True,
    learning_rate=float(os.environ.get("LEARNING_RATE", "5e-6")),
    warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.03")),
    lr_scheduler_type=os.environ.get("LR_SCHEDULER_TYPE", "cosine"),
    optim=os.environ.get("OPTIM", "adamw_torch"),
    max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "1.0")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.0")),
    max_length=max_length,
    max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "64")),
    lmbda=float(os.environ.get("LMBDA", "0.5")),
    beta=float(os.environ.get("BETA", "0.5")),
    temperature=float(os.environ.get("TEMPERATURE", "0.9")),
    seq_kd=os.environ.get("SEQ_KD", "False") == "True",
    max_steps=int(os.environ.get("MAX_STEPS", "100")),
    num_train_epochs=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")),
    logging_steps=int(os.environ.get("LOGGING_STEPS", "1")),
    save_steps=int(os.environ.get("SAVE_STEPS", "50")),
    save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "2")),
    eval_strategy=os.environ.get("EVAL_STRATEGY", "steps" if eval_dataset is not None else "no"),
    eval_steps=int(os.environ.get("EVAL_STEPS", "50")),
    report_to=os.environ.get("REPORT_TO", "none"),
    dataloader_num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "0")),
    seed=int(os.environ.get("SEED", "42")),
)

trainer = GKDTrainer(
    model=base_model,
    teacher_model=teacher_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print(f"[INFO] teacher_model={teacher_model}")
print(f"[INFO] train_examples={len(train_dataset)}")
print(f"[INFO] eval_examples={len(eval_dataset) if eval_dataset is not None else 0}")

resume_from_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
if not resume_from_checkpoint and os.environ.get("AUTO_RESUME", "False") == "True":
    checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if checkpoints:
        resume_from_checkpoint = str(checkpoints[-1])

print(f"[INFO] resume_from_checkpoint={resume_from_checkpoint or ''}")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.save_model()
tokenizer.save_pretrained(args.output_dir)
PY

NUM_GPUS="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"

"${VENV_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" "${TRAIN_SCRIPT}"

deactivate
