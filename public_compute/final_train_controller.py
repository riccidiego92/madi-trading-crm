from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfl_public.p43_final_train_sharded import (
    CALIBRATION_REPS,
    FINAL_REPS,
    aggregate_calibration,
    aggregate_final,
)

SHARD_REPS = 2000
CANDIDATES_PER_BATCH = 5

PLAN_PATH = Path("public_compute/results/final_train_plan.json")
QUEUE_PATH = Path("public_compute/queue/final_train_active_batch.json")
STATE_PATH = Path("public_compute/results/final_train/state.json")
RESULT_ROOT = Path("public_compute/results/final_train")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _starts(start: int, stop: int) -> list[int]:
    if (stop - start) % SHARD_REPS:
        raise ValueError("shard size must exactly divide phase range")
    return list(range(start, stop, SHARD_REPS))


def _queue(phase: str, candidates: list[int], generation: int) -> dict:
    if phase == "calibration":
        starts = _starts(0, CALIBRATION_REPS)
    elif phase == "null":
        starts = _starts(CALIBRATION_REPS, FINAL_REPS)
    else:
        raise ValueError(phase)
    tasks = [
        {
            "candidate_index": int(idx),
            "rep_start": int(start),
            "rep_count": SHARD_REPS,
        }
        for idx in candidates
        for start in starts
    ]
    if len(tasks) > 256:
        raise ValueError(f"matrix too large: {len(tasks)}")
    return {
        "schema": "wfl-p43-final-train-active-batch-1",
        "phase": phase,
        "generation": int(generation),
        "candidate_indices": [int(x) for x in candidates],
        "candidate_count": len(candidates),
        "shard_reps": SHARD_REPS,
        "tasks": tasks,
        "scientific_settings": {
            "total_reps_per_candidate": FINAL_REPS,
            "calibration_reps": CALIBRATION_REPS,
            "null_reps": FINAL_REPS - CALIBRATION_REPS,
            "alpha_final": 0.05,
            "train_only": True,
            "validation_accessed": False,
            "holdout_accessed": False,
        },
    }


def _plan_candidates() -> list[int]:
    plan = _load(PLAN_PATH)
    if plan.get("schema") != "wfl-public-p43-final-train-plan-1":
        raise SystemExit("bad FINAL_TRAIN plan schema")
    if plan.get("status") == "TERMINAL_ALL_FAMILIES_REJECTED":
        return []
    if plan.get("status") != "FINAL_TRAIN_PLAN_FROZEN":
        raise SystemExit("FINAL_TRAIN plan not frozen")
    return [int(x) for x in plan["final_train"]["candidate_indices"]]


def _final_path(idx: int) -> Path:
    return RESULT_ROOT / "final" / f"candidate_{idx}.json"


def _cal_path(idx: int) -> Path:
    return RESULT_ROOT / "calibration" / f"candidate_{idx}.json"


def _cal_dir(idx: int) -> Path:
    return RESULT_ROOT / "calibration_shards" / f"candidate_{idx}"


def _null_dir(idx: int) -> Path:
    return RESULT_ROOT / "null_shards" / f"candidate_{idx}"


def _schedule_next(plan_candidates: list[int], generation: int) -> None:
    remaining = [x for x in plan_candidates if not _final_path(x).exists()]
    if not remaining:
        _write(STATE_PATH, {
            "schema": "wfl-p43-final-train-state-1",
            "status": "FINAL_TRAIN_COMPLETE",
            "candidate_count": len(plan_candidates),
            "last_generation": int(generation) - 1,
        })
        return

    # Never recompute a completed 50k calibration. This is especially
    # important after a retry queue, whose candidate_indices contain only the
    # previously incomplete candidates.
    ready_for_null = [x for x in remaining if _cal_path(x).exists()]
    if ready_for_null:
        batch = ready_for_null[:CANDIDATES_PER_BATCH]
        _write(QUEUE_PATH, _queue("null", batch, generation))
        _write(STATE_PATH, {
            "schema": "wfl-p43-final-train-state-1",
            "status": "NULL_BATCH_QUEUED",
            "generation": int(generation),
            "candidate_indices": batch,
            "completed_final_count": len(plan_candidates) - len(remaining),
            "planned_final_count": len(plan_candidates),
            "resumed_from_existing_calibration": True,
        })
        return

    batch = remaining[:CANDIDATES_PER_BATCH]
    _write(QUEUE_PATH, _queue("calibration", batch, generation))
    _write(STATE_PATH, {
        "schema": "wfl-p43-final-train-state-1",
        "status": "CALIBRATION_BATCH_QUEUED",
        "generation": int(generation),
        "candidate_indices": batch,
        "completed_final_count": len(plan_candidates) - len(remaining),
        "planned_final_count": len(plan_candidates),
    })


def bootstrap() -> None:
    candidates = _plan_candidates()
    if not candidates:
        _write(STATE_PATH, {
            "schema": "wfl-p43-final-train-state-1",
            "status": "TERMINAL_NO_FINAL_TRAIN_CANDIDATES",
        })
        return

    _schedule_next(candidates, 1)


def advance() -> None:
    plan_candidates = _plan_candidates()
    if not QUEUE_PATH.exists():
        bootstrap()
        return

    q = _load(QUEUE_PATH)
    if q.get("schema") != "wfl-p43-final-train-active-batch-1":
        raise SystemExit("bad active batch schema")
    phase = q["phase"]
    generation = int(q["generation"])
    candidates = [int(x) for x in q["candidate_indices"]]

    if phase == "calibration":
        incomplete = []
        for idx in candidates:
            paths = sorted(_cal_dir(idx).glob("shard_*.json"))
            if len(paths) != CALIBRATION_REPS // SHARD_REPS:
                incomplete.append(idx)
                continue
            shards = [_load(p) for p in paths]
            cal = aggregate_calibration(idx, shards)
            _write(_cal_path(idx), cal)
        if incomplete:
            tasks = []
            for idx in incomplete:
                have = {
                    int(p.stem.split("_")[-1])
                    for p in _cal_dir(idx).glob("shard_*.json")
                }
                for start in _starts(0, CALIBRATION_REPS):
                    if start not in have:
                        tasks.append({
                            "candidate_index": idx,
                            "rep_start": start,
                            "rep_count": SHARD_REPS,
                        })
            retry = _queue("calibration", incomplete, generation + 1)
            retry["tasks"] = tasks
            retry["retry_only_missing_shards"] = True
            _write(QUEUE_PATH, retry)
            _write(STATE_PATH, {
                "schema": "wfl-p43-final-train-state-1",
                "status": "CALIBRATION_RETRY_QUEUED",
                "generation": generation + 1,
                "candidate_indices": candidates,
                "incomplete_candidates": incomplete,
                "missing_shard_count": len(tasks),
            })
            return
        _schedule_next(plan_candidates, generation + 1)
        return

    if phase == "null":
        incomplete = []
        for idx in candidates:
            cal_path = _cal_path(idx)
            paths = sorted(_null_dir(idx).glob("shard_*.json"))
            if not cal_path.exists() or len(paths) != (FINAL_REPS - CALIBRATION_REPS) // SHARD_REPS:
                incomplete.append(idx)
                continue
            cal = _load(cal_path)
            shards = [_load(p) for p in paths]
            final = aggregate_final(idx, cal, shards)
            _write(_final_path(idx), final)
        if incomplete:
            tasks = []
            for idx in incomplete:
                have = {
                    int(p.stem.split("_")[-1])
                    for p in _null_dir(idx).glob("shard_*.json")
                }
                for start in _starts(CALIBRATION_REPS, FINAL_REPS):
                    if start not in have:
                        tasks.append({
                            "candidate_index": idx,
                            "rep_start": start,
                            "rep_count": SHARD_REPS,
                        })
            retry = _queue("null", incomplete, generation + 1)
            retry["tasks"] = tasks
            retry["retry_only_missing_shards"] = True
            _write(QUEUE_PATH, retry)
            _write(STATE_PATH, {
                "schema": "wfl-p43-final-train-state-1",
                "status": "NULL_RETRY_QUEUED",
                "generation": generation + 1,
                "candidate_indices": candidates,
                "incomplete_candidates": incomplete,
                "missing_shard_count": len(tasks),
            })
            return

        _schedule_next(plan_candidates, generation + 1)
        return

    raise SystemExit(f"unknown phase {phase}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bootstrap", "advance"])
    args = ap.parse_args()
    if args.command == "bootstrap":
        bootstrap()
    else:
        advance()


if __name__ == "__main__":
    main()
