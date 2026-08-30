#!/usr/bin/env python3
"""
scripts/instances.py — what paper-pipeline is doing on this machine, right now.

Answers a different question than paper-pipeline-tui or scripts/monitor.sh,
both of which assume you already know which papers folder/database you care
about. This one discovers whatever `paper-pipeline` processes are actually
running system-wide -- possibly more than one, possibly from different
installations -- and reports, per instance: how long it's been running, its
root/source/database directories, live database entry counts, resource
usage (including any child processes like `dot`/`tesseract`), how to shut it
down gracefully, and what it depends on over the network.

Stdlib only, no dependencies, runs without an activated venv:
    python3 scripts/instances.py

Linux only (reads /proc directly), matching this project's existing
Ubuntu/Debian assumption throughout cli_howto.md.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# config.py and store.py import nothing beyond the stdlib and each other --
# confirmed by reading both files -- so this is safe without pymupdf/requests
# installed, i.e. without the project's own venv activated.
from paper_pipeline import config, store  # noqa: E402

GRAPH_VIZ_PORT = 8687  # must match scripts/graph_viz.sh's COSMOS_PORT
NEO4J_HTTP_PORT = 7475
CLK_TCK = os.sysconf("SC_CLK_TCK")


# ══════════════════════════════════════════════════════════════════════════
# Process discovery
# ══════════════════════════════════════════════════════════════════════════

def _read_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [p.decode(errors="replace") for p in raw.split(b"\0") if p]


def _is_paper_pipeline_instance(argv: list[str]) -> bool:
    """Exact match only -- never a substring. `paper-pipeline-tui` and
    `cosmos_server.py` must never be mistaken for this, the same lesson
    this session already learned the hard way from a Docker project-name
    collision and a process-name collision, applied here proactively."""
    if len(argv) < 2:
        return False
    if Path(argv[1]).name == "paper-pipeline":
        return True
    # `python3 -m paper_pipeline ...` -- Python rewrites argv[0]/[1] to the
    # resolved __main__.py path for -m invocations.
    return any(a.endswith("paper_pipeline/__main__.py") for a in argv)


def discover_instances() -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            argv = _read_cmdline(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if _is_paper_pipeline_instance(argv):
            pids.append(pid)
    return sorted(pids)


# ══════════════════════════════════════════════════════════════════════════
# Per-process introspection
# ══════════════════════════════════════════════════════════════════════════

def _stat_fields(pid: int) -> Optional[list[str]]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm (2nd field) can itself contain spaces (e.g. "kworker/u8:2-events
    # power queue"), so naive whitespace-splitting the whole line shifts
    # every later index. Extract pid/comm explicitly around the
    # parenthesised command name instead.
    open_paren = text.index("(")
    close_paren = text.rindex(")")
    pid_field = text[:open_paren].strip()
    comm_field = text[open_paren + 1 : close_paren]
    rest = text[close_paren + 1 :].split()
    return [pid_field, comm_field] + rest


def uptime_seconds(pid: int) -> Optional[float]:
    fields = _stat_fields(pid)
    if fields is None:
        return None
    starttime_ticks = int(fields[21])
    boot_uptime = float(Path("/proc/uptime").read_text().split()[0])
    return boot_uptime - (starttime_ticks / CLK_TCK)


def cpu_times(pid: int) -> Optional[float]:
    """Total CPU seconds (utime+stime) consumed so far -- for computing a
    short-window CPU% via two samples, not a point-in-time percentage."""
    fields = _stat_fields(pid)
    if fields is None:
        return None
    utime, stime = int(fields[13]), int(fields[14])
    return (utime + stime) / CLK_TCK


def rss_bytes(pid: int) -> Optional[int]:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def children_of(pid: int) -> list[tuple[int, str]]:
    """(child_pid, comm) for every process whose PPID is `pid`, right now."""
    kids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cpid = int(entry.name)
        fields = _stat_fields(cpid)
        if fields is None:
            continue
        comm = fields[1].strip("()")
        ppid = int(fields[3])
        if ppid == pid:
            kids.append((cpid, comm))
    return kids


def cpu_percent_over(pids: list[int], window: float = 0.2) -> dict[int, float]:
    before = {p: cpu_times(p) for p in pids}
    time.sleep(window)
    after = {p: cpu_times(p) for p in pids}
    result = {}
    for p in pids:
        if before.get(p) is None or after.get(p) is None:
            continue
        result[p] = max(0.0, (after[p] - before[p]) / window * 100)
    return result


def read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}
    env = {}
    for part in raw.split(b"\0"):
        if b"=" in part:
            k, _, v = part.partition(b"=")
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def parse_pipeline_args(argv: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (papers_dir, db_path) as given on the command line, or None
    for whichever was omitted (meaning that instance is using its default)."""
    papers_dir = None
    db_path = None
    rest = argv[2:]  # skip [python, .../paper-pipeline]
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--db-path" and i + 1 < len(rest):
            db_path = rest[i + 1]
            i += 2
            continue
        if not arg.startswith("--") and papers_dir is None:
            papers_dir = arg
        i += 1
    return papers_dir, db_path


def resolve_db_path_for_instance(cli_value: Optional[str], env: dict[str, str]) -> Path:
    """Same resolution order as store.resolve_db_path(), but honoring the
    specific process's own environment rather than this script's."""
    if cli_value:
        return Path(os.path.expandvars(cli_value)).expanduser()
    env_value = env.get(config.DB_PATH_ENV_VAR)
    if env_value:
        return Path(os.path.expandvars(env_value)).expanduser()
    return config.DEFAULT_DB_PATH


def db_report(db_path: Path) -> str:
    if not db_path.exists():
        return "unreadable: file does not exist"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as exc:
        return f"unreadable: {exc}"
    try:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_complete = 0
        for row in conn.execute("SELECT sections_completed FROM papers"):
            sections = set(json.loads(row["sections_completed"] or "[]"))
            if store.ALL_SECTIONS.issubset(sections):
                n_complete += 1
        return f"{total} papers ({n_complete} complete, {total - n_complete} partial/not-started)"
    except sqlite3.OperationalError as exc:
        return f"unreadable: {exc}"
    finally:
        conn.close()


def _http_get_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception:
        return None


def ollama_report(env: dict[str, str]) -> str:
    url = env.get("OLLAMA_URL", "http://localhost:11434")
    ps = _http_get_json(f"{url}/api/ps")
    if ps is None:
        return f"{url} — unreachable"
    models = ps.get("models", [])
    if not models:
        return f"{url} — reachable, no model currently loaded"
    m = models[0]
    vram_gb = m.get("size_vram", 0) / 1_073_741_824
    return f"{url} — reachable, {m.get('name')} loaded ({vram_gb:.1f} GB VRAM)"


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}m{s}s" if h else f"{m}m{s}s"


# ══════════════════════════════════════════════════════════════════════════
# Related services (not per-instance -- machine-wide networking context)
# ══════════════════════════════════════════════════════════════════════════

def related_services_report() -> list[str]:
    lines = []

    neo4j_status = "not running"
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=paper-pipeline-neo4j", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            neo4j_status = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    lines.append(f"  Neo4j (evidence graph)   127.0.0.1:{NEO4J_HTTP_PORT}  — {neo4j_status}")

    viewer_status = "not running"
    try:
        with socket.create_connection(("127.0.0.1", GRAPH_VIZ_PORT), timeout=1):
            viewer_status = f"reachable at http://localhost:{GRAPH_VIZ_PORT}/"
    except OSError:
        pass
    lines.append(f"  cosmos.gl viewer         127.0.0.1:{GRAPH_VIZ_PORT}  — {viewer_status}")

    return lines


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    pids = discover_instances()

    print("=" * 72)
    print("paper-pipeline instances")
    print("=" * 72)

    if not pids:
        print("\n  No paper-pipeline instances currently running.\n")
    else:
        cpu_pcts = cpu_percent_over(pids)
        for pid in pids:
            argv = _read_cmdline(pid)
            env = read_environ(pid)
            papers_dir_arg, db_path_arg = parse_pipeline_args(argv)
            db_path = resolve_db_path_for_instance(db_path_arg, env)
            kids = children_of(pid)

            print(f"\n── PID {pid} " + "─" * 50)
            print(f"  Uptime         : {_fmt_duration(uptime_seconds(pid))}")
            try:
                print(f"  Root directory : {os.readlink(f'/proc/{pid}/cwd')}")
            except OSError:
                print("  Root directory : (unavailable)")
            print(f"  PDF source     : {papers_dir_arg or f'{config.DEFAULT_PAPERS_DIR} (default)'}")
            print(f"  Database       : {db_path}"
                  + ("" if db_path_arg else " (default/env, not --db-path)"))
            print(f"  Database rows  : {db_report(db_path)}")
            print(f"  CPU / memory   : {cpu_pcts.get(pid, 0.0):.1f}% CPU, "
                  f"{_fmt_bytes(rss_bytes(pid))} RSS")
            if kids:
                for cpid, comm in kids:
                    print(f"    ↳ child {comm} (PID {cpid}): "
                          f"{_fmt_bytes(rss_bytes(cpid))} RSS")
            else:
                print("    (no child processes right now -- dot/tesseract "
                      "only run briefly during diagram/OCR steps)")
            print(f"  Ollama         : {ollama_report(env)}")
            print(f"  Graceful stop  : kill -TERM {pid}")
            print("                   (finishes the current section, then exits "
                  "cleanly; a second")
            print("                   SIGINT/SIGTERM instead force-exits "
                  "immediately -- not graceful)")

    print("\n" + "=" * 72)
    print("Related services (not paper-pipeline itself)")
    print("=" * 72 + "\n")
    for line in related_services_report():
        print(line)
    print()


if __name__ == "__main__":
    main()
