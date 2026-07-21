#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Error: pi05 Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/packages/openpi-client/src:${ROOT_DIR}/envs/curobo/src:${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/realman_remote_server.py" "$@"
