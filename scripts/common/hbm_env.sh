#!/usr/bin/env bash

# HBM read/write bandwidth is captured by a fully separate amdsmi-based sampler
# process (scripts/monitoring/rocm_hbm_sampler.py), started by
# scripts/servers/run_server_common.sh alongside the plain vLLM server (no
# profiler wraps the server itself -- see rocm_hbm_sampler.py's docstring for
# why: classic rocprof does capture real data but was confirmed 2026-08-18 to
# cause the recurring "Worker died unexpectedly" crash when used to wrap the
# server for its whole lifetime).

rivf26_find_vllm_pid() {
  local port=$1
  pgrep -f "vllm\.entrypoints\.cli\.main serve.*--port ${port}\b" | head -1
}

# vLLM's own Worker_TP*/EngineCore subprocesses set a custom process title
# (visible in `ps`'s CMD column as e.g. "VLLM::Worker_TP0") but carry no --port
# of their own, so rivf26_find_vllm_pid can never identify them. Observed
# directly on this host: when a worker crashes (the "Worker died unexpectedly"
# instability) and its APIServer exits from the resulting fatal error, the
# OTHER worker in the TP pair does not necessarily die with it -- it can be
# orphaned (reparented to PID 1) and keep running indefinitely (spin-waiting on
# a collective op with a partner that's gone), holding its full GPU memory
# allocation until something kills it explicitly. This is system-wide (workers
# carry no port to scope by) but restricted to orphans (PPID 1), so it will not
# touch a live sibling replica's still-supervised workers.
rivf26_reap_orphaned_vllm_workers() {
  local pid ppid
  for status_file in /proc/[0-9]*/status; do
    [[ -r "$status_file" ]] || continue
    ppid=$(awk '/^PPid:/{print $2}' "$status_file" 2>/dev/null)
    [[ "$ppid" == "1" ]] || continue
    pid=$(basename "$(dirname "$status_file")")
    if grep -q "VLLM::" "/proc/$pid/cmdline" 2>/dev/null; then
      echo "reaping orphaned vLLM worker: pid=$pid" >&2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

# Gracefully stop the vLLM server (found by port, not by the orchestrator's
# tracked PID, so this is safe under concurrent arms -- never process-group
# kills, which would risk a sibling replica sharing a process group in some
# launch patterns), falling back to a direct kill of server_pid if the vLLM
# child can never be identified (e.g. the server never started).
rivf26_stop_vllm_server() {
  local server_pid=$1
  local port=$2
  local timeout_iterations=${3:-120}
  local vllm_pid
  vllm_pid=$(rivf26_find_vllm_pid "$port")
  if [[ -n "$vllm_pid" ]]; then
    kill -TERM "$vllm_pid" 2>/dev/null || true
  else
    kill -TERM -- "-$server_pid" 2>/dev/null || true
  fi
  local _
  for _ in $(seq 1 "$timeout_iterations"); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  rivf26_reap_orphaned_vllm_workers
}

# Stop the amdsmi HBM sampler started by run_server_common.sh (PID file at
# $bulk_run_dir/raw/hbm_sampler.pid). SIGTERM lets it write final metadata
# (sample_rows, gpu_count) before exiting; safe to call even if HBM capture
# was disabled for this run (RIVF26_DISABLE_HBM_CAPTURE=1, no PID file).
rivf26_stop_hbm_sampler() {
  local bulk_run_dir=$1
  local pid_file="$bulk_run_dir/raw/hbm_sampler.pid"
  [[ -f "$pid_file" ]] || return 0
  local sampler_pid
  sampler_pid=$(cat "$pid_file" 2>/dev/null)
  [[ -n "$sampler_pid" ]] || return 0
  kill -TERM "$sampler_pid" 2>/dev/null || return 0
  local _
  for _ in $(seq 1 20); do
    kill -0 "$sampler_pid" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -KILL "$sampler_pid" 2>/dev/null || true
}
