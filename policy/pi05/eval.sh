#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Error: pi05 Python environment not found: ${PYTHON_BIN}" >&2
    echo "Run 'cd ${SCRIPT_DIR} && uv sync' to create it." >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/packages/openpi-client/src:${ROOT_DIR}/envs/curobo/src:${ROOT_DIR}:${PYTHONPATH:-}"

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${ROOT_DIR}"

PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON_BIN}" script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
