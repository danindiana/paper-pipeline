#!/usr/bin/env bash
# scripts/graph_viz.sh — manage the optional Neo4j + cosmos.gl evidence
# graph viewer (neo4j_viz/). Entirely optional; the core pipeline never
# calls this.
#
# Usage:
#   ./scripts/graph_viz.sh start [--db-path PATH]   # bring up Neo4j + viewer
#   ./scripts/graph_viz.sh import --db-path PATH    # (re-)import only
#   ./scripts/graph_viz.sh stop                      # stop viewer + Neo4j
#   ./scripts/graph_viz.sh status                    # check what's running

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEO4J_VIZ_DIR="$REPO_ROOT/neo4j_viz"

COSMOS_PORT=8687
NEO4J_HTTP_PORT=7475

usage() {
    echo "Usage: graph_viz.sh {start|import|stop|status} [--db-path PATH]" >&2
    exit 1
}

CMD="${1:-}"
[[ $# -gt 0 ]] && shift
DB_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-path) DB_PATH="$2"; shift 2 ;;
        *) usage ;;
    esac
done

# Identify cosmos_server.py by the PORT it's bound to, not by process name --
# a sibling project's own neo4j_viz/cosmos_server.py can be running under the
# identical filename on a different port at the same time. Matching by name
# risks acting on the wrong project's process; matching by port cannot.
cosmos_server_pid() {
    lsof -ti :"$COSMOS_PORT" -sTCP:LISTEN 2>/dev/null || true
}

require_docker() {
    if ! command -v docker &>/dev/null; then
        echo "ERROR: docker not found. Install Docker to use the evidence graph viewer." >&2
        exit 1
    fi
}

require_neo4j_driver() {
    if ! python3 -c "import neo4j" &>/dev/null; then
        echo "ERROR: the 'neo4j' Python package is required." >&2
        echo "  pip install neo4j" >&2
        exit 1
    fi
}

cmd_status() {
    echo "-- Neo4j container --"
    local status
    status=$(docker ps --filter "name=paper-pipeline-neo4j" --format '{{.Status}}' 2>/dev/null || true)
    if [[ -n "$status" ]]; then
        echo "  paper-pipeline-neo4j: $status  (http://localhost:$NEO4J_HTTP_PORT)"
    else
        echo "  not running"
    fi

    echo "-- cosmos_server.py --"
    local pid
    pid=$(cosmos_server_pid)
    if [[ -n "$pid" ]]; then
        echo "  running (PID $pid) — http://localhost:$COSMOS_PORT/"
    else
        echo "  not running"
    fi
}

cmd_import() {
    require_neo4j_driver
    if [[ -z "$DB_PATH" ]]; then
        echo "ERROR: import requires --db-path PATH" >&2
        exit 1
    fi
    python3 "$NEO4J_VIZ_DIR/import_to_neo4j.py" --db-path "$DB_PATH"
}

cmd_start() {
    require_docker
    require_neo4j_driver

    echo "-- Starting Neo4j (paper-pipeline-neo4j) --"
    (cd "$NEO4J_VIZ_DIR" && docker compose up -d)

    echo "-- Waiting for Neo4j to accept connections --"
    for _ in $(seq 1 30); do
        curl -sf "http://localhost:$NEO4J_HTTP_PORT" >/dev/null 2>&1 && break
        sleep 2
    done

    if [[ -n "$DB_PATH" ]]; then
        echo "-- Importing evidence graph from $DB_PATH --"
        cmd_import
    else
        echo "-- No --db-path given, skipping import --"
        echo "   Run later with: ./scripts/graph_viz.sh import --db-path PATH"
    fi

    local pid
    pid=$(cosmos_server_pid)
    if [[ -n "$pid" ]]; then
        echo "-- cosmos_server.py already running (PID $pid) --"
    else
        echo "-- Starting cosmos_server.py --"
        nohup python3 "$NEO4J_VIZ_DIR/cosmos_server.py" \
            > "$NEO4J_VIZ_DIR/cosmos_server.log" 2>&1 &
        disown
        sleep 1
    fi

    echo
    echo "Evidence graph viewer: http://localhost:$COSMOS_PORT/"
}

cmd_stop() {
    local pid
    pid=$(cosmos_server_pid)
    if [[ -n "$pid" ]]; then
        echo "Stopping cosmos_server.py (PID $pid, port $COSMOS_PORT)"
        kill "$pid"
    else
        echo "cosmos_server.py not running"
    fi

    echo "Stopping Neo4j container"
    (cd "$NEO4J_VIZ_DIR" && docker compose down)
}

case "$CMD" in
    start) cmd_start ;;
    import) cmd_import ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    *) usage ;;
esac
