#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GT_DIR="${ROOT_DIR}/serving_core"
PYTHON_BIN="${PYTHON_BIN:-python}"

THINKER_SCRIPT="${GT_DIR}/server_thinker.py"
TALKER_SCRIPT="${GT_DIR}/server_talker.py"
ORCH_SCRIPT="${GT_DIR}/server_orchestrator.py"

LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/run"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

THINKER_LOG="${LOG_DIR}/thinker.log"
TALKER_LOG="${LOG_DIR}/talker.log"
ORCH_LOG="${LOG_DIR}/orchestrator.log"

THINKER_PID_FILE="${PID_DIR}/thinker.pid"
TALKER_PID_FILE="${PID_DIR}/talker.pid"
ORCH_PID_FILE="${PID_DIR}/orchestrator.pid"
THINKER_PORT="${THINKER_PORT:-19999}"
TALKER_PORT="${TALKER_PORT:-20000}"
ORCH_PORT="${ORCH_PORT:-21000}"
THINKER_MODEL="${THINKER_MODEL:-models/qwen3-omni-thinker}"
TALKER_MODEL="${TALKER_MODEL:-models/qwen3-omni-talker}"
TALKER_TEMPERATURE="${TALKER_TEMPERATURE:-1.0}"
TALKER_TOP_P="${TALKER_TOP_P:-0.9}"
TALKER_TOP_K="${TALKER_TOP_K:-50}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for script in "${THINKER_SCRIPT}" "${TALKER_SCRIPT}" "${ORCH_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "Missing script: ${script}" >&2
    exit 1
  fi
done

check_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "${name} already running (pid=${pid}). Stop it first or remove ${pid_file}." >&2
      exit 1
    fi
    rm -f "${pid_file}"
  fi
}

port_is_busy() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v port="${port}" 'NR > 1 { n = split($4, parts, ":"); if (parts[n] == port) { found = 1; exit } } END { exit(found ? 0 : 1) }'
    return $?
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${port}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

require_port_free() {
  local name="$1"
  local port="$2"
  if port_is_busy "${port}"; then
    echo "${name} port ${port} is already in use. Run stop_thinker_talker_stack.sh first." >&2
    exit 1
  fi
}

start_one() {
  local name="$1"
  local log_file="$2"
  local pid_file="$3"
  shift 3
  nohup setsid "$@" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  echo "started ${name} pid=${pid} log=${log_file}"
}

check_pid_file "thinker" "${THINKER_PID_FILE}"
check_pid_file "talker" "${TALKER_PID_FILE}"
check_pid_file "orchestrator" "${ORCH_PID_FILE}"
require_port_free "thinker" "${THINKER_PORT}"
require_port_free "talker" "${TALKER_PORT}"
require_port_free "orchestrator" "${ORCH_PORT}"

start_one \
  "thinker" \
  "${THINKER_LOG}" \
  "${THINKER_PID_FILE}" \
  env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    THINKER_PORT="${THINKER_PORT}" \
    THINKER_MODEL="${THINKER_MODEL}" \
    THINKER_GPU_MEM_UTIL="${THINKER_GPU_MEM_UTIL:-0.8}" \
    THINKER_HIDDEN_STORE_DIR=0 \
    "${PYTHON_BIN}" "${THINKER_SCRIPT}"

start_one \
  "talker" \
  "${TALKER_LOG}" \
  "${TALKER_PID_FILE}" \
  env \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    TALKER_PORT="${TALKER_PORT}" \
    TALKER_MODEL="${TALKER_MODEL}" \
    TALKER_TEMPERATURE="${TALKER_TEMPERATURE}" \
    TALKER_TOP_P="${TALKER_TOP_P}" \
    TALKER_TOP_K="${TALKER_TOP_K}" \
    TALKER_TP=4 \
    TALKER_GPU_MEM_UTIL="${TALKER_GPU_MEM_UTIL:-0.6}" \
    TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES=7 \
    TALKER_MTP_SUBPROC_TP=1 \
    TALKER_MTP_GPU_MEM_UTIL="${TALKER_MTP_GPU_MEM_UTIL:-0.12}" \
    TALKER_MTP_PROFILE=0 \
    "${PYTHON_BIN}" "${TALKER_SCRIPT}" --tp 4

start_one \
  "orchestrator" \
  "${ORCH_LOG}" \
  "${ORCH_PID_FILE}" \
  env \
    THINKER_INTERNAL_URL="http://127.0.0.1:${THINKER_PORT}/internal/chat_turn" \
    TALKER_INTERNAL_URL_BASE="http://127.0.0.1:${TALKER_PORT}/internal/talker/turn" \
    TALKER_DELETE_URL_BASE="http://127.0.0.1:${TALKER_PORT}/v1/talker/session" \
    ORCH_PORT="${ORCH_PORT}" \
    "${PYTHON_BIN}" "${ORCH_SCRIPT}"

cat <<EOF

stack started

logs:
  tail -f "${THINKER_LOG}"
  tail -f "${TALKER_LOG}"
  tail -f "${ORCH_LOG}"

stop:
  "${ROOT_DIR}/stop_thinker_talker_stack.sh"

simulate target:
  http://127.0.0.1:${ORCH_PORT}/v1/chat/completions
  ws://127.0.0.1:${ORCH_PORT}/v1/audio/stream/{session_id}

gpu layout:
  thinker -> physical 0,1,2,3
  talker main -> physical 4,5,6,7
  talker mtp subproc -> physical 7
EOF
