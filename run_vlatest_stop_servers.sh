#!/usr/bin/env bash
# Stop vlatest GR00T inference servers by the known experiment ports.
#
# Usage:
#   bash run_vlatest_stop_servers.sh
#   bash run_vlatest_stop_servers.sh 5556 5557 5558

set -euo pipefail

ports=("$@")
if [[ "${#ports[@]}" -eq 0 ]]; then
    ports=(5556 5557 5558 5560 5561 5562)
fi

kill_process_tree() {
    local pid="${1:-}"
    local signal="${2:-TERM}"
    local child
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_process_tree "$child" "$signal"
    done
    kill -"$signal" "$pid" 2>/dev/null || true
}

stop_pid() {
    local pid="$1"
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    echo "Stopping pid=$pid"
    kill_process_tree "$pid" TERM
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force stopping pid=$pid"
        kill_process_tree "$pid" KILL
    fi
}

for port in "${ports[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
    else
        echo "Port $port: cannot inspect listeners because neither lsof nor fuser is installed"
        continue
    fi
    if [[ -z "$pids" ]]; then
        echo "Port $port: no listener"
        continue
    fi
    for pid in $pids; do
        echo "Port $port: listener pid=$pid"
        stop_pid "$pid"
    done
done

remaining="$(pgrep -f "private/vlatest/scripts/inference_service.py" 2>/dev/null || true)"
if [[ -n "$remaining" ]]; then
    echo "Stopping remaining vlatest inference_service.py processes"
    for pid in $remaining; do
        stop_pid "$pid"
    done
fi

echo "Done."
