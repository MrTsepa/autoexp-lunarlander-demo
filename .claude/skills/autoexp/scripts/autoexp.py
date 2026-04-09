#!/usr/bin/env python3
"""autoexp — ML experiment toolkit for autonomous agents. Single-file, zero dependencies."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUTOEXP_DIR = ".autoexp"
DB_FILE = "experiments.db"
CONFIG_FILE = "config.yaml"
PROGRAM_FILE = "program.md"

DEFAULT_CONFIG_TEMPLATE = """\
# autoexp project configuration

# Files the agent can modify (glob patterns)
editable_files:
  - "configs/*.yaml"
  - "src/**/*.py"

# Files the agent must NEVER modify (glob patterns)
locked_files:
  - "eval.py"
  - ".autoexp/*"

# Abort rules — kill training if these conditions are met
abort_rules: []
  # - pattern: "loss"
  #   condition: "above"
  #   threshold: 100.0
  #   after_lines: 500
"""

DEFAULT_PROGRAM_TEMPLATE = """\
# Research Program

## Goal
<!-- What are you trying to achieve? Be specific: metric, target value, constraints. -->

## Current Best
<!-- What's the best result so far? Include experiment ID if available. -->

## Search Space
<!-- What should the agent explore? Architecture, hyperparameters, reward design, etc. -->

## Constraints
<!-- What must NOT change? Eval protocol, environment code, etc. -->

## Known Insights
<!-- What have you already learned? Don't re-discover these. -->
"""

# ---------------------------------------------------------------------------
# Config (simple YAML subset parser — no pyyaml dependency)
# ---------------------------------------------------------------------------

def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal YAML subset: top-level keys with scalar or list values."""
    result = {}
    current_key = None
    current_list = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under current key
        if stripped.startswith("- ") and current_key is not None:
            val = stripped[2:].strip().strip('"').strip("'")
            if current_list is None:
                current_list = []
            current_list.append(val)
            result[current_key] = current_list
            continue

        # Top-level key
        if ":" in stripped and not stripped.startswith("-"):
            if current_list is not None:
                current_list = None
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            current_key = key

            if val == "[]":
                result[key] = []
                current_list = None
            elif val:
                # Scalar value
                val = val.strip('"').strip("'")
                if val.lower() == "true":
                    result[key] = True
                elif val.lower() == "false":
                    result[key] = False
                else:
                    try:
                        result[key] = int(val)
                    except ValueError:
                        try:
                            result[key] = float(val)
                        except ValueError:
                            result[key] = val
                current_list = None
            else:
                current_list = []
                result[key] = current_list

    return result


def load_config() -> dict:
    path = Path(AUTOEXP_DIR) / CONFIG_FILE
    if not path.exists():
        return {"editable_files": ["*"], "locked_files": [], "abort_rules": []}
    cfg = _parse_simple_yaml(path.read_text())
    cfg.setdefault("editable_files", ["*"])
    cfg.setdefault("locked_files", [])
    cfg.setdefault("abort_rules", [])
    return cfg


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    commit_sha TEXT NOT NULL,
    hypothesis TEXT,
    status TEXT DEFAULT 'created',
    started_at TEXT,
    finished_at TEXT,
    abort_reason TEXT,
    train_command TEXT,
    eval_command TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT REFERENCES experiments(id),
    metric_name TEXT,
    value REAL,
    source TEXT,
    recorded_at TEXT,
    PRIMARY KEY (experiment_id, metric_name, recorded_at)
);
CREATE TABLE IF NOT EXISTS evaluations (
    experiment_id TEXT REFERENCES experiments(id),
    eval_name TEXT,
    score REAL,
    raw_output TEXT,
    evaluated_at TEXT
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _row_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _db():
    conn = sqlite3.connect(str(Path(AUTOEXP_DIR) / DB_FILE))
    conn.row_factory = _row_factory
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _next_id(conn):
    row = conn.execute("SELECT COUNT(*) as n FROM experiments").fetchone()
    return f"auto_{row['n'] + 1:03d}"


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def _git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _head_sha():
    r = _git("rev-parse", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _changed_files():
    files = set()
    for cmd in (["diff", "--name-only", "HEAD"], ["diff", "--name-only", "--cached"],
                ["ls-files", "--others", "--exclude-standard"]):
        r = _git(*cmd, check=False)
        for line in r.stdout.strip().splitlines():
            if line:
                files.add(line)
    return sorted(files)


def _commit(message, files=None):
    if files:
        _git("add", *files)
    else:
        _git("add", "-A")
    _git("commit", "-m", message)
    return _head_sha()


def _revert():
    r = _git("revert", "HEAD", "--no-edit", check=False)
    return (r.returncode == 0, r.stderr.strip())


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def _matches(filepath, patterns):
    return any(fnmatch(filepath, p) for p in patterns)


def _validate_files(files, config):
    editable = config.get("editable_files", ["*"])
    locked = config.get("locked_files", [])

    for fp in files:
        if _matches(fp, locked):
            return False, f"locked:{fp}"
        if editable != ["*"] and not _matches(fp, editable):
            return False, f"not_editable:{fp}"
        if not Path(fp).exists():
            return False, f"not_found:{fp}"
        if fp.endswith(".py"):
            content = Path(fp).read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                return False, f"syntax_error:{fp}:{e.msg}:{e.lineno}"
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / Path(fp).name
                tmp.write_text(content)
                r = subprocess.run([sys.executable, "-m", "py_compile", str(tmp)],
                                   capture_output=True, text=True, check=False)
                if r.returncode != 0:
                    return False, f"compile_error:{fp}:{r.stderr[:300]}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class TrainResult(NamedTuple):
    status: str
    exit_code: int | None
    duration_seconds: float
    stdout_tail: str
    abort_reason: str | None


def _parse_timeout(s):
    s = s.strip().lower()
    if s.endswith("h"): return int(float(s[:-1]) * 3600)
    if s.endswith("m"): return int(float(s[:-1]) * 60)
    if s.endswith("s"): return int(float(s[:-1]))
    return int(s)


def _run_training(command, timeout=None, abort_rules=None, on_line=None):
    timeout_secs = _parse_timeout(str(timeout)) if timeout else None
    abort_rules = abort_rules or []
    tail, line_num, abort_reason = [], 0, None

    start = time.time()
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            line_num += 1
            tail.append(line)
            if len(tail) > 100:
                tail.pop(0)
            if on_line:
                on_line(line)
            # Check abort rules
            for rule in abort_rules:
                if line_num < rule.get("after_lines", 0):
                    continue
                pat = rule.get("pattern", "")
                m = re.search(rf"{pat}\s*[=:]\s*([\d.eE+-]+)", line)
                if not m:
                    continue
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                cond, thresh = rule.get("condition", "below"), rule.get("threshold", 0.0)
                if (cond == "below" and val < thresh) or (cond == "above" and val > thresh):
                    abort_reason = f"{pat}={val} vs {thresh}"
                    proc.terminate()
                    proc.wait(timeout=10)
                    break
            if abort_reason:
                break
            if timeout_secs and (time.time() - start) > timeout_secs:
                abort_reason = f"timeout after {timeout_secs}s"
                proc.terminate()
                proc.wait(timeout=10)
                break
        else:
            proc.wait()
    except Exception:
        proc.kill()
        proc.wait()

    duration = time.time() - start
    if abort_reason and "timeout" in abort_reason:
        status = "timeout"
    elif abort_reason:
        status = "aborted"
    elif proc.returncode != 0:
        status = "crashed"
    else:
        status = "completed"

    return TrainResult(status, proc.returncode, round(duration, 1), "\n".join(tail), abort_reason)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _generate_report(conn):
    experiments = conn.execute("SELECT * FROM experiments ORDER BY started_at DESC").fetchall()
    if not experiments:
        return "# Experiment Log\n\nNo experiments recorded yet.\n"

    by_status = {}
    for exp in experiments:
        s = exp["status"]
        by_status[s] = by_status.get(s, 0) + 1

    bests = conn.execute(
        """SELECT eval_name, MAX(score) as best, experiment_id FROM evaluations
           JOIN experiments ON experiments.id = evaluations.experiment_id
           WHERE experiments.status = 'completed' GROUP BY eval_name"""
    ).fetchall()

    lines = ["# Experiment Log", "", "Generated by autoexp.", ""]
    lines.append("## Summary")
    lines.append(f"- **Total experiments:** {len(experiments)}")
    lines.append(f"- **By status:** {' | '.join(f'{s}: {n}' for s, n in sorted(by_status.items()))}")
    for b in bests:
        lines.append(f"- **Best {b['eval_name']}:** {b['best']} ({b['experiment_id']})")
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    lines.append("| ID | Hypothesis | Status | Duration | Evals |")
    lines.append("|----|-----------|--------|----------|-------|")

    for exp in experiments:
        hypothesis = (exp["hypothesis"] or "")[:60]
        duration = ""
        if exp["started_at"] and exp["finished_at"]:
            try:
                secs = (datetime.fromisoformat(exp["finished_at"]) -
                        datetime.fromisoformat(exp["started_at"])).total_seconds()
                duration = f"{secs/3600:.1f}h" if secs > 3600 else f"{secs/60:.0f}m" if secs > 60 else f"{secs:.0f}s"
            except (ValueError, TypeError):
                pass
        evals = conn.execute("SELECT eval_name, score FROM evaluations WHERE experiment_id = ?",
                             (exp["id"],)).fetchall()
        eval_str = ", ".join(f"{e['eval_name']}={e['score']:.3f}" for e in evals) if evals else "-"
        lines.append(f"| {exp['id']} | {hypothesis} | {exp['status']} | {duration} | {eval_str} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_init(_args):
    d = Path(AUTOEXP_DIR)
    d.mkdir(exist_ok=True)
    for name, content in [(CONFIG_FILE, DEFAULT_CONFIG_TEMPLATE), (PROGRAM_FILE, DEFAULT_PROGRAM_TEMPLATE)]:
        dest = d / name
        if not dest.exists():
            dest.write_text(content)
            print(f"Created {dest}")
        else:
            print(f"Already exists: {dest}")
    conn = _db()
    conn.close()
    print(f"Database: {d / DB_FILE}")
    print("\nDone. Edit .autoexp/program.md with your research goal.")


def cmd_status(_args):
    conn = _db()
    running = conn.execute("SELECT * FROM experiments WHERE status='running' ORDER BY started_at DESC").fetchall()
    if running:
        print("RUNNING:")
        for exp in running:
            print(f"  {exp['id']}: {exp['hypothesis']}")
            if exp["train_command"]:
                print(f"    command: {exp['train_command']}")
            print(f"    started: {exp['started_at']}")
    else:
        print("No experiments running.")
    recent = conn.execute("SELECT * FROM experiments ORDER BY started_at DESC LIMIT 3").fetchall()
    if recent:
        print("\nRecent:")
        for exp in recent:
            print(f"  {exp['id']}: [{exp['status']}] {exp['hypothesis']}")
    conn.close()


def cmd_validate(args):
    ok, reason = _validate_files(args.files, load_config())
    if ok:
        print(f"OK: {len(args.files)} file(s) valid")
    else:
        print(f"FAIL: {reason}", file=sys.stderr)
        sys.exit(1)


def cmd_commit(args):
    conn = _db()
    files = args.files or _changed_files()
    files = [f for f in files if not f.startswith(".autoexp/")]
    if not files:
        print("No changes to commit.", file=sys.stderr)
        sys.exit(1)
    config = load_config()
    ok, reason = _validate_files([f for f in files if Path(f).exists()], config)
    if not ok:
        print(f"Validation failed: {reason}", file=sys.stderr)
        sys.exit(1)
    exp_id = _next_id(conn)
    sha = _commit(f"[{exp_id}] {args.hypothesis}", files)
    conn.execute("INSERT INTO experiments (id, commit_sha, hypothesis, status, started_at) VALUES (?, ?, ?, 'created', ?)",
                 (exp_id, sha, args.hypothesis, _now()))
    conn.commit()
    print(f"Experiment {exp_id} committed ({sha[:8]})")
    print(f"  hypothesis: {args.hypothesis}")
    print(f"  files: {', '.join(files)}")
    conn.close()


def cmd_train(args):
    conn = _db()
    config = load_config()
    exp_id = args.experiment
    if not exp_id:
        row = conn.execute("SELECT id FROM experiments ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            print("No experiments found. Run commit first.", file=sys.stderr)
            sys.exit(1)
        exp_id = row["id"]
    conn.execute("UPDATE experiments SET status='running', train_command=? WHERE id=?", (args.command, exp_id))
    conn.commit()
    print(f"Training {exp_id}: {args.command}")
    if args.timeout:
        print(f"  timeout: {args.timeout}")
    result = _run_training(args.command, timeout=args.timeout,
                           abort_rules=config.get("abort_rules", []),
                           on_line=lambda l: print(f"  | {l}"))
    conn.execute("UPDATE experiments SET status=?, finished_at=?, abort_reason=? WHERE id=?",
                 (result.status, _now(), result.abort_reason, exp_id))
    conn.commit()
    print(f"\nResult: {result.status} ({result.duration_seconds}s)")
    if result.abort_reason:
        print(f"  abort reason: {result.abort_reason}")
    if result.exit_code and result.exit_code != 0:
        print(f"  exit code: {result.exit_code}")
    conn.close()


def cmd_eval(args):
    conn = _db()
    exp_id = args.experiment
    if not exp_id:
        row = conn.execute("SELECT id FROM experiments ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            print("No experiments found.", file=sys.stderr)
            sys.exit(1)
        exp_id = row["id"]
    print(f"Evaluating {exp_id}: {args.command}")
    r = subprocess.run(args.command, shell=True, capture_output=True, text=True)
    output = r.stdout + r.stderr
    print(output)
    scores_found = 0
    for line in output.splitlines():
        for match in re.finditer(r"(\w+)\s*[=:]\s*([\d.eE+-]+)", line):
            name, value = match.group(1), match.group(2)
            try:
                score = float(value)
                conn.execute("INSERT INTO evaluations (experiment_id, eval_name, score, raw_output, evaluated_at) VALUES (?,?,?,?,?)",
                             (exp_id, name, score, output, _now()))
                conn.execute("INSERT OR REPLACE INTO metrics (experiment_id, metric_name, value, source, recorded_at) VALUES (?,?,?,?,?)",
                             (exp_id, name, score, "eval", _now()))
                scores_found += 1
            except ValueError:
                continue
    if scores_found:
        conn.commit()
        print(f"\nRecorded {scores_found} metric(s) for {exp_id}")
    else:
        conn.execute("INSERT INTO evaluations (experiment_id, eval_name, score, raw_output, evaluated_at) VALUES (?,?,?,?,?)",
                     (exp_id, "raw", 0.0, output, _now()))
        conn.commit()
        print(f"\nNo metrics extracted. Raw output saved for {exp_id}")
    conn.execute("UPDATE experiments SET eval_command=? WHERE id=?", (args.command, exp_id))
    conn.commit()
    conn.close()


def cmd_revert(args):
    conn = _db()
    exp_id = args.experiment
    if not exp_id:
        row = conn.execute("SELECT id FROM experiments ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            print("No experiments to revert.", file=sys.stderr)
            sys.exit(1)
        exp_id = row["id"]
    ok, msg = _revert()
    if ok:
        conn.execute("UPDATE experiments SET status='discarded', finished_at=? WHERE id=?", (_now(), exp_id))
        conn.commit()
        print(f"Reverted {exp_id}")
    else:
        print(f"Revert failed: {msg}", file=sys.stderr)
        sys.exit(1)
    conn.close()


def cmd_results(args):
    conn = _db()
    if args.best:
        row = conn.execute(
            """SELECT e.*, m.value as best_value FROM experiments e
               JOIN metrics m ON e.id = m.experiment_id
               WHERE m.metric_name = ? AND e.status = 'completed'
               ORDER BY m.value DESC LIMIT 1""", (args.best,)).fetchone()
        if row:
            if args.json:
                print(json.dumps(row, indent=2))
            else:
                print(f"Best {args.best}: {row.get('best_value')} ({row['id']})")
                print(f"  hypothesis: {row['hypothesis']}")
        else:
            print(f"No completed experiments with metric '{args.best}'")
        conn.close()
        return
    experiments = conn.execute("SELECT * FROM experiments ORDER BY started_at DESC LIMIT ?",
                               (args.last,)).fetchall()
    if args.json:
        for exp in experiments:
            exp["evaluations"] = conn.execute("SELECT * FROM evaluations WHERE experiment_id=?", (exp["id"],)).fetchall()
            exp["metrics"] = conn.execute("SELECT * FROM metrics WHERE experiment_id=?", (exp["id"],)).fetchall()
        print(json.dumps(experiments, indent=2))
    else:
        if not experiments:
            print("No experiments recorded.")
        for exp in experiments:
            evals = conn.execute("SELECT eval_name, score FROM evaluations WHERE experiment_id=?", (exp["id"],)).fetchall()
            eval_str = ", ".join(f"{e['eval_name']}={e['score']:.3f}" for e in evals)
            print(f"{exp['id']} [{exp['status']}] {exp['hypothesis']}")
            if eval_str:
                print(f"  evals: {eval_str}")
    conn.close()


def cmd_report(_args):
    conn = _db()
    print(_generate_report(conn))
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog="autoexp", description="ML experiment toolkit for autonomous agents")
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("init", help="Initialize .autoexp/ in current project")
    sub.add_parser("status", help="Show current experiment state")

    p = sub.add_parser("validate", help="Validate files")
    p.add_argument("files", nargs="+")

    p = sub.add_parser("commit", help="Commit changes as experiment")
    p.add_argument("hypothesis", help="Experiment hypothesis")
    p.add_argument("--files", nargs="*", default=None)

    p = sub.add_parser("train", help="Run training with monitoring")
    p.add_argument("command", help="Training command to run")
    p.add_argument("--timeout", default=None, help="Timeout (e.g. 20m, 2h, 30s)")
    p.add_argument("--experiment", default=None, help="Experiment ID")

    p = sub.add_parser("eval", help="Run evaluation")
    p.add_argument("command", help="Evaluation command to run")
    p.add_argument("--experiment", default=None, help="Experiment ID")

    p = sub.add_parser("revert", help="Revert last experiment")
    p.add_argument("--experiment", default=None, help="Experiment ID")

    p = sub.add_parser("results", help="Query experiments")
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--best", default=None, help="Show best by metric name")
    p.add_argument("--json", action="store_true")

    sub.add_parser("report", help="Generate RESEARCH.md")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    cmds = {"init": cmd_init, "status": cmd_status, "validate": cmd_validate,
            "commit": cmd_commit, "train": cmd_train, "eval": cmd_eval,
            "revert": cmd_revert, "results": cmd_results, "report": cmd_report}
    cmds[args.subcommand](args)


if __name__ == "__main__":
    main()
