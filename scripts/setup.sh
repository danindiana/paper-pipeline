#!/usr/bin/env bash
# scripts/setup.sh — cold-start setup for paper-pipeline.
#
# Idempotent: safe to re-run. Checks before acting at every step, never
# overwrites an existing .venv, never installs/starts system services
# without telling you first.
#
# Usage:
#   ./scripts/setup.sh                # full setup, including model pulls
#   ./scripts/setup.sh --skip-models  # skip `ollama pull` (manage models yourself)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
SKIP_MODELS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1; shift ;;
        *) echo "Usage: setup.sh [--skip-models]" >&2; exit 1 ;;
    esac
done

echo "== paper-pipeline setup =="
echo "repo: $REPO_ROOT"
echo

# ── 1. System packages ───────────────────────────────────────────────────────
echo "-- Checking system packages (graphviz, tesseract) --"
NEED_APT_PKGS=()
command -v dot &>/dev/null || NEED_APT_PKGS+=("graphviz")
command -v tesseract &>/dev/null || NEED_APT_PKGS+=("tesseract-ocr" "tesseract-ocr-eng")

if [[ ${#NEED_APT_PKGS[@]} -gt 0 ]]; then
    echo "   missing: ${NEED_APT_PKGS[*]}"
    echo "   installing via apt (requires sudo) ..."
    sudo apt-get update -qq
    sudo apt-get install -y "${NEED_APT_PKGS[@]}"
else
    echo "   graphviz + tesseract already present"
fi
echo

# ── 2. Python interpreter probe ──────────────────────────────────────────────
# Some systems have a python3.X whose `venv` module produces a broken/partial
# pip (ensurepip silently falls back to user-site instead of the venv) --
# this bit this project's own maintainer during initial setup. Probe for an
# interpreter that both meets the >=3.11 floor AND actually produces a
# working venv+pip, rather than assuming the newest python3 is fine.
echo "-- Probing for a working Python >=3.11 interpreter --"

find_python() {
    local candidates=(python3.11 python3.12 python3.13 python3.14 python3)
    local test_dir
    for py in "${candidates[@]}"; do
        command -v "$py" &>/dev/null || continue
        local ver major minor
        ver=$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null) || continue
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -ne 3 || "$minor" -lt 11 ]]; then
            continue
        fi
        test_dir="$(mktemp -d)"
        if "$py" -m venv "$test_dir" &>/dev/null && "$test_dir/bin/pip" --version &>/dev/null; then
            rm -rf "$test_dir"
            echo "$py"
            return 0
        fi
        rm -rf "$test_dir"
        echo "   (skipping $py — venv/pip module not usable; likely missing python3-venv)" >&2
    done
    return 1
}

if ! PYTHON_BIN="$(find_python)"; then
    echo "ERROR: no usable Python >=3.11 with a working venv module found." >&2
    echo "  Fix: sudo apt install python3.11 python3.11-venv" >&2
    exit 1
fi
echo "   using: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
echo

# ── 3. Virtual environment ───────────────────────────────────────────────────
echo "-- Setting up .venv --"
if [[ -d .venv ]]; then
    echo "   .venv already exists — leaving it in place"
else
    "$PYTHON_BIN" -m venv .venv
    echo "   created .venv with $PYTHON_BIN"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest
echo "   package + pytest installed"
echo

# ── 4. Ollama reachability ───────────────────────────────────────────────────
echo "-- Checking Ollama at $OLLAMA_URL --"
if ! command -v ollama &>/dev/null; then
    echo "ERROR: 'ollama' binary not found." >&2
    echo "  Install: https://ollama.com/download" >&2
    exit 1
fi
if ! curl -sf --max-time 5 "$OLLAMA_URL/api/tags" >/dev/null; then
    echo "ERROR: Ollama installed but not reachable at $OLLAMA_URL." >&2
    echo "  Start it: sudo systemctl start ollama   (or: ollama serve)" >&2
    exit 1
fi
echo "   Ollama reachable"
echo

# ── 5. Model pull ────────────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" -eq 1 ]]; then
    echo "-- Skipping model pull (--skip-models) --"
else
    echo "-- Pulling required models (reads paper_pipeline.config.MODEL_TIERS) --"
    MODELS="$(python3 -c 'from paper_pipeline import config; print("\n".join(dict.fromkeys(config.MODEL_TIERS.values())))')"
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        echo "   ollama pull $model"
        ollama pull "$model"
    done <<< "$MODELS"
fi
echo

# ── 6. Verify ────────────────────────────────────────────────────────────────
echo "-- Running test suite --"
python -m pytest tests/ -q

echo
echo "== Setup complete =="
echo "Next steps — see cli_howto.md for the full runbook. Quick version:"
echo "  source .venv/bin/activate"
echo "  paper-pipeline /path/to/pdfs --list          # dry-run status check"
echo "  paper-pipeline /path/to/pdfs --paper one.pdf  # smoke-test a single paper first"
echo "  paper-pipeline /path/to/pdfs                  # full batch"
echo "  ./scripts/monitor.sh /path/to/pdfs --watch     # observe it while it runs"
echo
echo "Optional: visualize the evidence graph behind processed papers"
echo "(requires Docker + 'pip install neo4j' — see neo4j_viz/README.md):"
echo "  ./scripts/graph_viz.sh start --db-path /path/to/papers.db"
