#!/usr/bin/env python3
"""Bounded, resumable Verilator checks for prepare_real_gemm.py fixtures (stdlib only)."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
COUNTERS = ("command_cycles", "operand_cycles", "input_starved_cycles",
            "output_cycles", "output_stall_cycles", "other_cycles")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def selection(raw):
    return set(raw.split(",")) if raw else None


def load_metrics(path, task):
    rows = list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"Invalid result row count: {path}")
    metrics = rows[0]
    if metrics["id"] != task["id"]:
        raise ValueError(f"Result id mismatch: {path}")
    for key in ("m", "n", "k", "cycles", "macs", "result_fires", *COUNTERS):
        metrics[key] = int(metrics[key])
        if metrics[key] < 0:
            raise ValueError(f"Negative metric {key}: {path}")
    metrics["wall_seconds"] = float(metrics["wall_seconds"])
    m, n, k = (task[key] for key in ("m", "n", "k"))
    if ([metrics[key] for key in ("m", "n", "k")] != [m, n, k]
            or metrics["command_cycles"] != 1
            or metrics["operand_cycles"] != k
            or metrics["result_fires"] != m * n
            or metrics["macs"] != m * n * k
            or sum(metrics[key] for key in COUNTERS) != metrics["cycles"]):
        raise ValueError(f"Inconsistent phase/shape counters: {path}")
    metrics["useful_mac_utilization"] = metrics["macs"] / (512 * metrics["cycles"])
    metrics["output_fraction"] = metrics["output_cycles"] / metrics["cycles"]
    metrics["active_lane_fraction"] = m * n / 512
    return metrics


def bounded_make(arguments):
    """SpinalHDL passes host CPU count to make; override it with the run's cap."""
    jobs = int(os.environ["REAL_GEMM_BUILD_JOBS"])
    if not 1 <= jobs <= 6:
        raise ValueError("REAL_GEMM_BUILD_JOBS must be 1..6")
    filtered = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "-j":
            index += 1
            if index < len(arguments) and arguments[index].isdigit():
                index += 1
            continue
        if not value.startswith("-j"):
            filtered.append(value)
        index += 1
    command = [*shlex.split(os.environ.get("REAL_GEMM_MAKE", "make")), *filtered, f"-j{jobs}"]
    print(f"RealGemm build: jobs={jobs}; {shlex.join(command)}", flush=True)
    os.execvp(command[0], command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "out/real-gemm/manifest.json")
    parser.add_argument("--max-tasks", type=int, default=2, help="At most this many unfinished tiles per invocation")
    parser.add_argument("--max-seconds", type=float, default=180, help="Wall deadline for the single build/simulation subprocess")
    parser.add_argument("--threads", type=int, choices=range(1, 7), default=1,
                        help="Verilator evaluation threads (default: 1; current small RTL cannot schedule 2/6)")
    parser.add_argument("--build-jobs", type=int, choices=range(1, 7), default=6,
                        help="C++ build job cap, independent of evaluation threads (default: 6)")
    parser.add_argument("--layers", help="Comma-separated transformer layer indices")
    parser.add_argument("--projections", help="Comma-separated projection names")
    parser.add_argument("--stages", help="smoke,tile (filter only, never changes fixture dimensions)")
    parser.add_argument("--input-gap", type=int, default=0)
    parser.add_argument("--output-stall", type=int, default=0)
    parser.add_argument("--status", action="store_true", help="Validate artifacts and report without launching simulation")
    args = parser.parse_args()
    if args.max_tasks < 1 or args.max_seconds <= 0 or min(args.input_gap, args.output_stall) < 0:
        parser.error("max-tasks/max-seconds must be positive; gap/stall must be nonnegative")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or not manifest.get("tasks"):
        raise ValueError("Expected a version 1 manifest with tasks")
    layers, projections, stages = map(selection, (args.layers, args.projections, args.stages))
    tasks = [t for t in manifest["tasks"]
             if (layers is None or str(t["layer"]) in layers)
             and (projections is None or t["projection"] in projections)
             and (stages is None or t["stage"] in stages)]
    if not tasks:
        parser.error("No tasks match the filters")
    ids = [t["id"] for t in manifest["tasks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs in manifest")
    fixtures = {}
    for task in tasks:
        path = (manifest_path.parent / task["fixture"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or path.stem != task["id"]:
            raise ValueError(f"Unsafe fixture path/id: {task['id']}")
        if digest(path) != task["sha256"]:
            raise ValueError(f"Fixture checksum mismatch (regenerate, do not resume): {path}")
        fixtures[task["id"]] = path
    # Bind resume to the model/input manifest, every RTL/driver source, tool version,
    # build settings and stall policy. Never reuse a PASS from a different design.
    verilator_version = subprocess.check_output(["verilator", "--version"], text=True).strip()
    identity = {"manifest_sha256": digest(manifest_path), "verilator": verilator_version,
                "input_gap": args.input_gap, "output_stall": args.output_stall,
                "threads": args.threads, "build_jobs": args.build_jobs,
                "make_command": os.environ.get("SPINAL_MAKE_CMD", "make"),
                "environment": {key: os.environ.get(key) for key in
                                ("SIM_OPT_LEVEL", "SPINAL_VERILATOR_FLAGS", "JAVA_HOME")},
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in sorted([
                    *ROOT.glob("src/**/*.scala"), ROOT / "build.mill", ROOT / ".mill-version", Path(__file__).resolve()])}}
    run_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    run_dir = manifest_path.parent / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir = run_dir / "results"
    result_dir.mkdir(exist_ok=True)
    with (run_dir / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(run_dir / "identity.json", identity)
        state_path = run_dir / "checkpoint.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"version": 1, "passed": {}}

        def collect():
            for task in tasks:
                result = result_dir / (task["id"] + ".tsv")
                previous = state["passed"].get(task["id"])
                if previous and not result.exists():
                    raise ValueError(f"Previously passed result is missing: {result}")
                if result.exists():
                    metric = load_metrics(result, task)
                    if previous and (previous["result_sha256"] != digest(result)
                                     or previous["fixture_sha256"] != task["sha256"]):
                        raise ValueError(f"Previously passed result/fixture was modified: {result}")
                    state["passed"][task["id"]] = {
                        "fixture_sha256": task["sha256"], "result_sha256": digest(result), "metrics": metric}
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        collect()  # Also recovers tiles completed before an interruption/timeout.
        pending = [t for t in tasks if t["id"] not in state["passed"]]
        print(f"Run {run_key}: selected={len(tasks)} passed={len(tasks)-len(pending)} pending={len(pending)}", flush=True)
        exit_code = 0
        if pending and not args.status:
            batch = pending[:args.max_tasks]
            command = ["mill", "--no-server", "channel.runMain", "npu.sim.RealGemmCheck",
                       "--output", str(result_dir), "--input-gap", str(args.input_gap),
                       "--output-stall", str(args.output_stall), *[str(fixtures[t["id"]]) for t in batch]]
            child_env = os.environ.copy()
            child_env["SIM_VERILATOR_THREADS"] = str(args.threads)
            child_env["REAL_GEMM_BUILD_JOBS"] = str(args.build_jobs)
            child_env["REAL_GEMM_MAKE"] = os.environ.get("SPINAL_MAKE_CMD", "make")
            child_env["SPINAL_MAKE_CMD"] = shlex.join(
                [sys.executable, str(Path(__file__).resolve()), "--make"])
            log_path = run_dir / f"attempt-{time.time_ns()}.log"
            print(f"Launching {len(batch)} task(s), threads={args.threads}, build-jobs={args.build_jobs}, "
                  f"deadline={args.max_seconds:g}s; log={log_path}", flush=True)
            state["last_attempt"] = {"tasks": [t["id"] for t in batch], "log": str(log_path), "status": "running"}
            atomic_json(state_path, state)
            started = time.monotonic()
            try:
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, env=child_env)
                    try:
                        exit_code = process.wait(timeout=args.max_seconds)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        exit_code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
                        print("Stopped at invocation boundary; completed tile artifacts are retained.", flush=True)
            finally:
                state["last_attempt"]["status"] = "passed" if exit_code == 0 else f"exit-{exit_code}"
                state["last_attempt"]["wall_seconds"] = time.monotonic() - started
                collect()
            if exit_code:
                print(f"Simulation failed/stopped (exit={exit_code}); inspect {log_path}", file=sys.stderr)
            elif any(t["id"] not in state["passed"] for t in batch):
                raise ValueError("Simulator exited successfully without all requested tile results")
        passed = [t for t in tasks if t["id"] in state["passed"]]
        report_tasks = []
        print("id | shape(m,n,k) | cycles | input-gap | output-stall | useful-MAC% | emit%")
        for task in passed:
            metric = state["passed"][task["id"]]["metrics"]
            print(f"{task['id']} | {metric['m']},{metric['n']},{metric['k']} | {metric['cycles']} | "
                  f"{metric['input_starved_cycles']} | {metric['output_stall_cycles']} | "
                  f"{100*metric['useful_mac_utilization']:.2f} | {100*metric['output_fraction']:.2f}")
            report_tasks.append({"task": task, "metrics": metric})
        report = {"run_key": run_key, "selected": len(tasks), "passed": len(passed),
                  "pending": len(tasks) - len(passed), "tasks": report_tasks,
                  "threads": args.threads, "build_jobs": args.build_jobs,
                  "scope": "Bit-exact INT8 GEMM tiles from real FP-model activations; not full-model RTL inference",
                  "counter_meaning": "Disjoint interface cycles attributed to the issuing tile (cmd through last operand beat; a batch's last tile also owns its result tail); useful_MAC/(16*32*cycles), not wall-clock throughput",
                  "bottleneck_basis": "Accumulation overlaps the previous tile's emission via a result bank; a tile's last operand beat waits until that bank drains; one output element/cycle; underfilled tiles waste physical lanes. No DDR/DMA model."}
        atomic_json(run_dir / "report.json", report)
        print(f"Checkpoint: {state_path}\nReport: {run_dir / 'report.json'}")
        print(f"Selected coverage: {len(passed)}/{len(tasks)} PASS; {len(tasks)-len(passed)} pending. "
              "Re-run the same command to continue; --max-tasks limits work per invocation.")
        return exit_code


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--make"]:
            bounded_make(sys.argv[2:])
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
