#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PID_DIR="${ROOT_DIR}/run"

THINKER_PID_FILE="${PID_DIR}/thinker.pid"
TALKER_PID_FILE="${PID_DIR}/talker.pid"
ORCH_PID_FILE="${PID_DIR}/orchestrator.pid"

THINKER_PORT="${THINKER_PORT:-19999}"
TALKER_PORT="${TALKER_PORT:-20000}"
ORCH_PORT="${ORCH_PORT:-21000}"
GPU_IDS="${GPU_IDS:-}"
GPU_RESET_THRESHOLD_MB="${GPU_RESET_THRESHOLD_MB:-1024}"

kill_pid_group() {
  local name="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  local pid
  pid="$(<"${pid_file}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    local pgid
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${pgid}" ]]; then
      echo "stopping ${name} pgid=${pgid}"
      kill -TERM -- "-${pgid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    else
      echo "stopping ${name} pid=${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi

  rm -f "${pid_file}"
}

kill_port_owner() {
  local name="$1"
  local port="$2"
  if command -v fuser >/dev/null 2>&1; then
    local pids
    pids="$(fuser -n tcp "${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "killing ${name} port=${port} pids=${pids}"
      fuser -k -TERM -n tcp "${port}" >/dev/null 2>&1 || true
      sleep 1
      fuser -k -KILL -n tcp "${port}" >/dev/null 2>&1 || true
    fi
  fi
}

gpu_memory_used_mb() {
  local gpu_id="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | awk 'NR == 1 { print int($1); found = 1 } END { if (!found) print 0 }'
}

list_gpu_ids() {
  if [[ -n "${GPU_IDS}" ]]; then
    echo "${GPU_IDS}" | tr ',' ' '
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}'
}

kill_gpu_device_holders() {
  local gpu_id="$1"
  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi
  local dev_path="/dev/nvidia${gpu_id}"
  if [[ -e "${dev_path}" ]]; then
    local holders
    holders="$(fuser "${dev_path}" 2>/dev/null || true)"
    if [[ -n "${holders}" ]]; then
      echo "killing gpu ${gpu_id} device holders: ${holders}"
      fuser -k -TERM "${dev_path}" >/dev/null 2>&1 || true
      sleep 1
      fuser -k -KILL "${dev_path}" >/dev/null 2>&1 || true
    fi
  fi
}

reset_gpu_if_stuck() {
  local gpu_id="$1"
  local used_mb
  used_mb="$(gpu_memory_used_mb "${gpu_id}")"
  if (( used_mb < GPU_RESET_THRESHOLD_MB )); then
    echo "gpu ${gpu_id} memory after stop: ${used_mb} MiB"
    return 0
  fi

  kill_gpu_device_holders "${gpu_id}"
  sleep 1
  used_mb="$(gpu_memory_used_mb "${gpu_id}")"
  if (( used_mb < GPU_RESET_THRESHOLD_MB )); then
    echo "gpu ${gpu_id} memory after device cleanup: ${used_mb} MiB"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu ${gpu_id} still holds ${used_mb} MiB, trying gpu reset"
    nvidia-smi --gpu-reset -i "${gpu_id}" >/dev/null 2>&1 || true
  fi

  sleep 1
  used_mb="$(gpu_memory_used_mb "${gpu_id}")"
  echo "gpu ${gpu_id} final memory: ${used_mb} MiB"
}

cleanup_all_gpus() {
  local gpu_id
  for gpu_id in $(list_gpu_ids); do
    [[ -n "${gpu_id}" ]] || continue
    echo "cleaning gpu ${gpu_id}"
    reset_gpu_if_stuck "${gpu_id}"
  done
}

kill_pid_group "thinker" "${THINKER_PID_FILE}"
kill_pid_group "talker" "${TALKER_PID_FILE}"
kill_pid_group "orchestrator" "${ORCH_PID_FILE}"

kill_port_owner "thinker" "${THINKER_PORT}"
kill_port_owner "talker" "${TALKER_PORT}"
kill_port_owner "orchestrator" "${ORCH_PORT}"
cleanup_all_gpus

echo "stack stopped"
