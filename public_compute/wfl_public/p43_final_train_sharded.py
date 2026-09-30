from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .p43_p42_train_screen import (
    FINAL_ALPHA,
    FINAL_REPS_MIN,
    TRAIN_SEED_ROOT,
    TRIAGE_ALPHA,
    _feature_vector,
    candidate_signature,
    observed_complete_vector,
    registry_rows,
    simulate_candidate,
    train_frame_from_full_history,
)

FINAL_REPS = 100_000
CALIBRATION_REPS = FINAL_REPS // 2
NULL_REPS = FINAL_REPS - CALIBRATION_REPS


def schema_sha(names: list[str]) -> str:
    return sha256("\0".join(names).encode("utf-8")).hexdigest()


def load_train_data(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    parts = [pd.read_csv(p) for p in sorted(root.glob("wfl_train_*.csv"))]
    if not parts:
        raise ValueError("no TRAIN data files found")
    return train_frame_from_full_history(pd.concat(parts, ignore_index=True))


def row_for_index(candidate_index: int) -> dict:
    rows = registry_rows()
    idx = int(candidate_index)
    if not 0 <= idx < len(rows):
        raise ValueError("candidate index out of range")
    return rows[idx]


def _replicate(task: tuple[dict, int, int]) -> tuple[str, np.ndarray]:
    row, n_contests, seed = task
    frame = simulate_candidate(row, n_contests=n_contests, experiment_seed=seed)
    names, vec = _feature_vector(frame)
    return schema_sha(names), np.asarray(vec, dtype=float)


def calibration_shard(
    candidate_index: int,
    rep_start: int,
    rep_count: int,
    *,
    data_dir: str | Path = "public_compute/data",
    workers: int | None = None,
) -> dict:
    start = int(rep_start)
    count = int(rep_count)
    stop = start + count
    if start < 0 or count <= 0 or stop > CALIBRATION_REPS:
        raise ValueError("calibration shard must lie inside [0, 50000)")

    row = row_for_index(candidate_index)
    train = load_train_data(data_dir)
    names, _ = observed_complete_vector(train)
    expected_schema = schema_sha(names)
    worker_count = max(1, int(workers or min(4, max(1, os.cpu_count() or 1))))

    sums = np.zeros(len(names), dtype=float)
    sums_sq = np.zeros(len(names), dtype=float)
    tasks = (
        (row, len(train), int(TRAIN_SEED_ROOT) + r)
        for r in range(start, stop)
    )
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for got_schema, vec in pool.map(_replicate, tasks, chunksize=1):
            if got_schema != expected_schema:
                raise AssertionError("feature schema changed in calibration shard")
            sums += vec
            sums_sq += vec * vec

    return {
        "schema": "wfl-p43-final-train-calibration-shard-1",
        "candidate_index": int(candidate_index),
        "candidate_id": row["candidate_id"],
        "candidate_signature_sha256": candidate_signature(row),
        "n_real_contests": int(len(train)),
        "feature_schema_sha256": expected_schema,
        "feature_count": len(names),
        "rep_start": start,
        "rep_count": count,
        "rep_stop_exclusive": stop,
        "seed_start": int(TRAIN_SEED_ROOT) + start,
        "seed_stop_exclusive": int(TRAIN_SEED_ROOT) + stop,
        "sum": sums.tolist(),
        "sum_sq": sums_sq.tolist(),
        "guardrails": {
            "stage": "train",
            "validation_accessed": False,
            "holdout_accessed": False,
            "scientific_settings_unchanged": True,
        },
    }


def _hash(payload: dict) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def aggregate_calibration(
    candidate_index: int,
    shards: list[dict],
    *,
    data_dir: str | Path = "public_compute/data",
) -> dict:
    if not shards:
        raise ValueError("no calibration shards")
    row = row_for_index(candidate_index)
    train = load_train_data(data_dir)
    names, obs = observed_complete_vector(train)
    expected_schema = schema_sha(names)
    expected_sig = candidate_signature(row)

    ordered = sorted(shards, key=lambda x: int(x["rep_start"]))
    cursor = 0
    sums = np.zeros(len(names), dtype=float)
    sums_sq = np.zeros(len(names), dtype=float)

    for s in ordered:
        if s.get("schema") != "wfl-p43-final-train-calibration-shard-1":
            raise ValueError("bad calibration shard schema")
        if int(s["candidate_index"]) != int(candidate_index):
            raise ValueError("mixed candidate indices")
        if s["candidate_id"] != row["candidate_id"]:
            raise ValueError("candidate id mismatch")
        if s["candidate_signature_sha256"] != expected_sig:
            raise ValueError("candidate signature mismatch")
        if s["feature_schema_sha256"] != expected_schema:
            raise ValueError("feature schema mismatch")
        if int(s["rep_start"]) != cursor:
            raise ValueError("calibration shards must be contiguous and non-overlapping")
        cursor = int(s["rep_stop_exclusive"])
        sums += np.asarray(s["sum"], dtype=float)
        sums_sq += np.asarray(s["sum_sq"], dtype=float)

    if cursor != CALIBRATION_REPS:
        raise ValueError(f"calibration coverage incomplete: {cursor}/{CALIBRATION_REPS}")

    n = float(CALIBRATION_REPS)
    mean = sums / n
    var = (sums_sq - (sums * sums) / n) / float(CALIBRATION_REPS - 1)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    keep = np.isfinite(sd) & (sd > 1e-12)
    if not np.any(keep):
        raise ValueError("all calibration coordinates have zero/invalid SD")

    kept_idx = np.flatnonzero(keep)
    obs_z = (obs[keep] - mean[keep]) / sd[keep]
    obs_t = float(np.max(np.abs(obs_z)))
    kept_names = [names[int(i)] for i in kept_idx]
    order = np.argsort(np.abs(obs_z))[::-1]
    top = [
        {"feature": kept_names[int(j)], "z": float(obs_z[int(j)])}
        for j in order[:20]
    ]

    out = {
        "schema": "wfl-p43-final-train-calibration-freeze-1",
        "candidate_index": int(candidate_index),
        "candidate_id": row["candidate_id"],
        "candidate_signature_sha256": expected_sig,
        "n_real_contests": int(len(train)),
        "simulation_reps_total": FINAL_REPS,
        "calibration_reps": CALIBRATION_REPS,
        "null_reps": NULL_REPS,
        "feature_schema_sha256": expected_schema,
        "retained_features": int(np.sum(keep)),
        "excluded_zero_sd_features": int(np.sum(~keep)),
        "kept_indices": kept_idx.tolist(),
        "mean_kept": mean[keep].tolist(),
        "sd_kept": sd[keep].tolist(),
        "observed_max_abs_z": obs_t,
        "top_observed_standardized_residuals": top,
        "aggregation_note": (
            "Calibration shards use the exact frozen replicate seeds and sufficient "
            "statistics; aggregation order is shard-order deterministic."
        ),
        "guardrails": {
            "stage": "train",
            "validation_accessed": False,
            "holdout_accessed": False,
            "calibration_complete_50000": True,
            "no_post_result_retuning": True,
        },
    }
    out["calibration_freeze_sha256"] = _hash(out)
    return out


def null_shard(
    candidate_index: int,
    rep_start: int,
    rep_count: int,
    calibration: dict,
    *,
    data_dir: str | Path = "public_compute/data",
    workers: int | None = None,
) -> dict:
    start = int(rep_start)
    count = int(rep_count)
    stop = start + count
    if start < CALIBRATION_REPS or count <= 0 or stop > FINAL_REPS:
        raise ValueError("null shard must lie inside [50000, 100000)")

    row = row_for_index(candidate_index)
    train = load_train_data(data_dir)
    names, _ = observed_complete_vector(train)
    expected_schema = schema_sha(names)
    expected_sig = candidate_signature(row)

    if calibration.get("schema") != "wfl-p43-final-train-calibration-freeze-1":
        raise ValueError("bad calibration freeze schema")
    if int(calibration["candidate_index"]) != int(candidate_index):
        raise ValueError("calibration candidate mismatch")
    if calibration["candidate_signature_sha256"] != expected_sig:
        raise ValueError("calibration signature mismatch")
    if calibration["feature_schema_sha256"] != expected_schema:
        raise ValueError("calibration schema mismatch")

    kept_idx = np.asarray(calibration["kept_indices"], dtype=int)
    mean = np.asarray(calibration["mean_kept"], dtype=float)
    sd = np.asarray(calibration["sd_kept"], dtype=float)
    obs_t = float(calibration["observed_max_abs_z"])
    worker_count = max(1, int(workers or min(4, max(1, os.cpu_count() or 1))))

    t_values: list[float] = []
    exceed = 0
    tasks = (
        (row, len(train), int(TRAIN_SEED_ROOT) + r)
        for r in range(start, stop)
    )
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for got_schema, vec in pool.map(_replicate, tasks, chunksize=1):
            if got_schema != expected_schema:
                raise AssertionError("feature schema changed in null shard")
            z = (vec[kept_idx] - mean) / sd
            t = float(np.max(np.abs(z)))
            t_values.append(t)
            exceed += int(t >= obs_t - 1e-15)

    out = {
        "schema": "wfl-p43-final-train-null-shard-1",
        "candidate_index": int(candidate_index),
        "candidate_id": row["candidate_id"],
        "candidate_signature_sha256": expected_sig,
        "calibration_freeze_sha256": calibration["calibration_freeze_sha256"],
        "feature_schema_sha256": expected_schema,
        "rep_start": start,
        "rep_count": count,
        "rep_stop_exclusive": stop,
        "seed_start": int(TRAIN_SEED_ROOT) + start,
        "seed_stop_exclusive": int(TRAIN_SEED_ROOT) + stop,
        "exceed_count": int(exceed),
        "t_values": t_values,
        "guardrails": {
            "stage": "train",
            "validation_accessed": False,
            "holdout_accessed": False,
            "scientific_settings_unchanged": True,
        },
    }
    out["null_shard_sha256"] = _hash(out)
    return out


def aggregate_final(
    candidate_index: int,
    calibration: dict,
    null_shards: list[dict],
) -> dict:
    if not null_shards:
        raise ValueError("no null shards")
    row = row_for_index(candidate_index)
    expected_sig = candidate_signature(row)

    ordered = sorted(null_shards, key=lambda x: int(x["rep_start"]))
    cursor = CALIBRATION_REPS
    t_values: list[float] = []
    exceed = 0
    for s in ordered:
        if s.get("schema") != "wfl-p43-final-train-null-shard-1":
            raise ValueError("bad null shard schema")
        if int(s["candidate_index"]) != int(candidate_index):
            raise ValueError("mixed candidate indices")
        if s["candidate_signature_sha256"] != expected_sig:
            raise ValueError("candidate signature mismatch")
        if s["calibration_freeze_sha256"] != calibration["calibration_freeze_sha256"]:
            raise ValueError("calibration freeze mismatch")
        if int(s["rep_start"]) != cursor:
            raise ValueError("null shards must be contiguous and non-overlapping")
        cursor = int(s["rep_stop_exclusive"])
        vals = [float(x) for x in s["t_values"]]
        if len(vals) != int(s["rep_count"]):
            raise ValueError("null shard t-value count mismatch")
        t_values.extend(vals)
        exceed += int(s["exceed_count"])

    if cursor != FINAL_REPS or len(t_values) != NULL_REPS:
        raise ValueError("null coverage incomplete")

    sim_t = np.asarray(t_values, dtype=float)
    p = float((1 + exceed) / (NULL_REPS + 1))
    result = {
        "schema": "wfl-p43-p42-train-candidate-evaluation-1",
        "stage": "train",
        "n_real_contests": int(calibration["n_real_contests"]),
        "simulation_reps": FINAL_REPS,
        "candidate_id": row["candidate_id"],
        "core_family_key": row["core_family_key"],
        "candidate_identity": row["candidate_identity"],
        "candidate_signature_sha256": expected_sig,
        "config": row["config"],
        "test": {
            "simulation_reps_total": FINAL_REPS,
            "normalization_reps": CALIBRATION_REPS,
            "null_evaluation_reps": NULL_REPS,
            "retained_features": int(calibration["retained_features"]),
            "excluded_zero_sd_features": int(calibration["excluded_zero_sd_features"]),
            "observed_max_abs_z": float(calibration["observed_max_abs_z"]),
            "simulated_max_abs_z_q95": float(np.quantile(sim_t, 0.95)),
            "simulated_max_abs_z_q99": float(np.quantile(sim_t, 0.99)),
            "p_max_stat": p,
            "alpha_final": FINAL_ALPHA,
            "rejected_at_final_alpha": bool(p <= FINAL_ALPHA),
            "rejected_at_triage_alpha": bool(p <= TRIAGE_ALPHA),
            "top_observed_standardized_residuals": calibration[
                "top_observed_standardized_residuals"
            ],
        },
        "evidence_grade": (
            "FINAL_TRAIN_ELIGIBLE_REPLICATE_COUNT"
            if FINAL_REPS >= FINAL_REPS_MIN
            else "DEVELOPMENT_OR_CONFIRMATION_ONLY"
        ),
        "guardrails": {
            "validation_accessed": False,
            "holdout_accessed": False,
            "internal_synthetic_state_compared_to_real": False,
            "operational_seed_search": False,
            "non_rejection_is_compatibility_only": True,
            "sharded_execution_only": True,
            "no_post_result_retuning": True,
        },
        "execution": {
            "mode": "deterministic_sharded_sufficient_statistics",
            "calibration_freeze_sha256": calibration["calibration_freeze_sha256"],
            "null_shard_count": len(ordered),
        },
    }
    return result
