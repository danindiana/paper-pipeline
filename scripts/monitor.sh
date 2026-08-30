#!/usr/bin/env bash
# scripts/monitor.sh — observe a running paper-pipeline batch at a glance.
#
# Shows: loaded Ollama model, per-GPU utilization/memory, batch progress
# (from --list), and a stuck heuristic. Adapted from the same-shaped tool
# in the author's ollama-delegate toolkit (peek.sh) — reimplemented
# standalone here since this is a separate public repo with no dependency
# on that project.
#
# Usage:
#   ./scripts/monitor.sh PAPERS_DIR                       # single snapshot
#   ./scripts/monitor.sh PAPERS_DIR --watch                # refresh every 5s
#   ./scripts/monitor.sh PAPERS_DIR --watch 10              # custom interval
#   ./scripts/monitor.sh PAPERS_DIR --db-path /path/to.db   # non-default DB

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# GPU utilization sum (across all GPUs, percent) below which we consider the
# pipeline potentially stuck, if it's also been running a while with no work.
STUCK_GPU_THRESHOLD=2
# Minimum seconds the paper-pipeline process must have been running before
# we flag low GPU util as suspicious (avoids false alarms during startup /
# PDF text extraction, which is CPU-bound and legitimately GPU-idle).
STUCK_MIN_ELAPSED=120

PAPERS_DIR=""
DB_PATH=""
WATCH=0
INTERVAL=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)
            WATCH=1
            shift
            [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]] && { INTERVAL="$1"; shift; }
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: monitor.sh PAPERS_DIR [--watch [interval_seconds]] [--db-path PATH]" >&2
            exit 0
            ;;
        *)
            if [[ -z "$PAPERS_DIR" ]]; then
                PAPERS_DIR="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$PAPERS_DIR" ]]; then
    echo "Usage: monitor.sh PAPERS_DIR [--watch [interval_seconds]] [--db-path PATH]" >&2
    exit 1
fi

PP_CMD=(paper-pipeline)
command -v paper-pipeline &>/dev/null || PP_CMD=(python3 -m paper_pipeline)
LIST_ARGS=("$PAPERS_DIR" --list)
[[ -n "$DB_PATH" ]] && LIST_ARGS+=(--db-path "$DB_PATH")

snapshot() {
    echo "━━━ paper-pipeline monitor — $(date '+%Y-%m-%d %H:%M:%S') ━━━"

    # --- Loaded Ollama model ---
    local ps_json count
    ps_json=$(curl -sf --max-time 3 "$OLLAMA_URL/api/ps" 2>/dev/null || echo '{"models":[]}')
    count=$(echo "$ps_json" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("models", [])))' 2>/dev/null || echo 0)

    if [[ "$count" -eq 0 ]]; then
        echo "  MODEL     : (none loaded)"
    else
        echo "$ps_json" | python3 -c '
import json, sys
m = json.load(sys.stdin)["models"][0]
name = m["name"]
vram_gb = m.get("size_vram", 0) / 1073741824
ctx = m.get("context_length", "?")
print("  MODEL     : {}  ({:.1f} GB VRAM, ctx={})".format(name, vram_gb, ctx))
'
    fi

    # --- GPU utilization ---
    local total_gpu_util=100
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU       :"
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader \
        | while IFS=',' read -r idx gname gpu_pct mem_used mem_total; do
            printf "    GPU%s %-22s  util: %-5s  mem: %s /%s\n" \
                "$idx" "$gname" "${gpu_pct// /}" "${mem_used// /}" "${mem_total// /}"
        done
        total_gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
            | awk '{sum += $1} END {print sum+0}')
    else
        echo "  GPU       : nvidia-smi not available"
    fi

    # --- Batch progress (from --list) ---
    local list_output
    if list_output=$("${PP_CMD[@]}" "${LIST_ARGS[@]}" 2>/dev/null); then
        local n_complete n_partial n_pending
        n_complete=$(grep -c '✅' <<< "$list_output" || true)
        n_partial=$(grep -c '⚠️' <<< "$list_output" || true)
        n_pending=$(grep -c '⬜' <<< "$list_output" || true)
        echo "  PROGRESS  : ${n_complete} complete / ${n_partial} partial / ${n_pending} not started (includes retryable failures)"
    else
        echo "  PROGRESS  : could not run --list against $PAPERS_DIR"
    fi

    # --- Process + stuck heuristic ---
    local pp_pid pp_elapsed
    pp_pid=$(pgrep -f "paper-pipeline|paper_pipeline" | head -1 || true)

    if [[ -z "$pp_pid" ]]; then
        echo "  PROCESS   : not running"
    else
        pp_elapsed=$(ps -p "$pp_pid" -o etimes= 2>/dev/null | tr -d ' ' || echo 0)
        echo "  PROCESS   : running (PID $pp_pid, up $((pp_elapsed/3600))h$(((pp_elapsed%3600)/60))m)"

        if [[ "${total_gpu_util%.*}" -le "$STUCK_GPU_THRESHOLD" && "$pp_elapsed" -ge "$STUCK_MIN_ELAPSED" ]]; then
            echo "  STUCK?    : ⚠ possibly — GPU util ${total_gpu_util}% with the process up ${pp_elapsed}s"
            echo "              (could also be legitimately CPU-bound: PDF/OCR extraction between LLM calls)"
            echo "              check: journalctl -u ollama --since '5 min ago' ; sudo systemctl restart ollama"
        else
            echo "  STUCK?    : no (GPU active, or process too new to judge)"
        fi
    fi

    echo
}

if [[ "$WATCH" -eq 1 ]]; then
    while true; do
        clear
        snapshot
        sleep "$INTERVAL"
    done
else
    snapshot
fi
