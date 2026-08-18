#!/usr/bin/env bash

# Classic rocprof (v1) wraps the vLLM server for real HBM read/write bandwidth (see
# scripts/monitoring/parse_rocprof_hbm.py for why rocprofv3/amdsmi cannot be used on
# this host). rocprof's own bash wrapper only completes its post-processing step -- and
# therefore only writes the kernel-dispatch CSV -- if it is left running long enough to
# see its wrapped vLLM child exit on its own. A process-group-wide SIGTERM kills the
# rocprof wrapper and its child simultaneously, so the wrapper never gets to finish and
# HBM telemetry for the run is silently lost. Shutdown must instead signal only the
# vLLM child PID and let the rocprof wrapper (server_pid) detect that exit and finish
# by itself.

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

# Gracefully stop a rocprof-wrapped vLLM server and wait for rocprof's own
# post-processing to finish, falling back to a process-group kill if it hangs or the
# vLLM child can never be identified (e.g. the server never started).
rivf26_stop_rocprof_server() {
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
