from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import os

import numpy as np
import pandas as pd

from .p43_p42_train_screen import (
    FINAL_ALPHA,
    FINAL_REPS_MIN,
    TRAIN_SEED_ROOT,
    TRIAGE_ALPHA,
    _feature_vector,
    candidate_signature,
    registry_by_id,
    simulate_candidate,
)


def _schema_sha(names: list[str]) -> str:
    return sha256("\0".join(names).encode("utf-8")).hexdigest()


def _replicate(task: tuple[dict, int, int]) -> tuple[str, np.ndarray]:
    row, n_contests, seed = task
    frame = simulate_candidate(row, n_contests=n_contests, experiment_seed=seed)
    names, vec = _feature_vector(frame)
    return _schema_sha(names), np.asarray(vec, dtype=float)


def parallel_train_evaluation(
    real_train: pd.DataFrame,
    candidate_id: str,
    reps: int,
    *,
    seed_root: int = TRAIN_SEED_ROOT,
    observed: tuple[list[str], np.ndarray] | None = None,
    registry_seed_root: int = 2026092642,
    workers: int | None = None,
) -> dict:
    rows = registry_by_id(registry_seed_root)
    try:
        row = rows[str(candidate_id)]
    except KeyError as exc:
        raise KeyError(f"unknown P42 candidate_id {candidate_id}") from exc

    names_obs, obs = (
        _feature_vector(real_train)
        if observed is None
        else observed
    )
    expected_schema = _schema_sha(names_obs)

    reps = int(reps)
    if reps < 4:
        raise ValueError("reps must be >=4")
    split = reps // 2
    if split < 2:
        raise ValueError("normalization split must contain at least 2 replicates")

    worker_count = int(workers or min(4, max(1, os.cpu_count() or 1)))
    worker_count = max(1, worker_count)

    sums = np.zeros(len(obs), dtype=float)
    sums_sq = np.zeros(len(obs), dtype=float)

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        cal_tasks = (
            (row, len(real_train), int(seed_root) + r)
            for r in range(split)
        )
        # executor.map preserves input order. Aggregate in that same order so
        # floating-point accumulation matches the reference sequential loop.
        for schema, vec in pool.map(_replicate, cal_tasks, chunksize=1):
            if schema != expected_schema:
                raise AssertionError("P42 feature schema changed across calibration")
            sums += vec
            sums_sq += vec * vec

        mean = sums / float(split)
        var = (sums_sq - (sums * sums) / float(split)) / float(split - 1)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        keep = np.isfinite(sd) & (sd > 1e-12)
        if not np.any(keep):
            raise ValueError("all P42 calibration coordinates have zero/invalid SD")

        obs_z = (obs[keep] - mean[keep]) / sd[keep]
        obs_t = float(np.max(np.abs(obs_z)))

        null_count = reps - split
        sim_t = np.empty(null_count, dtype=float)
        exceed = 0
        null_tasks = (
            (row, len(real_train), int(seed_root) + r)
            for r in range(split, reps)
        )
        for j, (schema, vec) in enumerate(
            pool.map(_replicate, null_tasks, chunksize=1)
        ):
            if schema != expected_schema:
                raise AssertionError("P42 feature schema changed in null evaluation")
            z = (vec[keep] - mean[keep]) / sd[keep]
            t = float(np.max(np.abs(z)))
            sim_t[j] = t
            exceed += int(t >= obs_t - 1e-15)

    p = float((1 + exceed) / (null_count + 1))
    kept_names = [n for n, use in zip(names_obs, keep) if use]
    order = np.argsort(np.abs(obs_z))[::-1]
    top = [
        {"feature": kept_names[int(j)], "z": float(obs_z[int(j)])}
        for j in order[:20]
    ]

    return {
        "schema": "wfl-p43-p42-train-candidate-evaluation-1",
        "stage": "train",
        "n_real_contests": int(len(real_train)),
        "simulation_reps": reps,
        "candidate_id": row["candidate_id"],
        "core_family_key": row["core_family_key"],
        "candidate_identity": row["candidate_identity"],
        "candidate_signature_sha256": candidate_signature(row),
        "config": row["config"],
        "test": {
            "simulation_reps_total": reps,
            "normalization_reps": split,
            "null_evaluation_reps": null_count,
            "retained_features": int(np.sum(keep)),
            "excluded_zero_sd_features": int(np.sum(~keep)),
            "observed_max_abs_z": obs_t,
            "simulated_max_abs_z_q95": float(np.quantile(sim_t, 0.95)),
            "simulated_max_abs_z_q99": float(np.quantile(sim_t, 0.99)),
            "p_max_stat": p,
            "alpha_final": FINAL_ALPHA,
            "rejected_at_final_alpha": bool(p <= FINAL_ALPHA),
            "rejected_at_triage_alpha": bool(p <= TRIAGE_ALPHA),
            "top_observed_standardized_residuals": top,
        },
        "evidence_grade": (
            "FINAL_TRAIN_ELIGIBLE_REPLICATE_COUNT"
            if reps >= FINAL_REPS_MIN
            else "DEVELOPMENT_OR_CONFIRMATION_ONLY"
        ),
        "guardrails": {
            "validation_accessed": False,
            "holdout_accessed": False,
            "internal_synthetic_state_compared_to_real": False,
            "operational_seed_search": False,
            "non_rejection_is_compatibility_only": True,
            "parallel_execution_only": True,
        },
        "execution": {
            "workers": worker_count,
            "replicate_order_preserved_for_aggregation": True,
        },
    }
