from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .p23_discriminator import stage_frame
from .p28_discriminability_v2 import attach_public_schedule
from .p29_metrics import complete_fingerprint, complete_fingerprint_vector
from .p42_phased_draw_machine import PhasedMachineConfig, phase_grid
from .p42_fast import simulate_phased_fast as simulate_phased
from .p46_calendar_alignment import calendar_alignment_offset

TRAIN_SEED_ROOT = 202609264301
TRIAGE_ALPHA = 0.01
FINAL_ALPHA = 0.05
FINAL_REPS_MIN = 100_000


def core_family_key(config: PhasedMachineConfig) -> str:
    return f"{config.mapper}|{config.stream_mode}|{config.reseed_mode}"


def registry_rows(seed_root: int = 2026092642) -> list[dict]:
    rows = []
    for cfg in phase_grid(seed_root):
        rows.append({
            "candidate_id": cfg.candidate_id(),
            "family": "P42_PHASED_HMAC_DRBG",
            "core_family_key": core_family_key(cfg),
            "candidate_identity": cfg.identity(),
            "config": cfg.to_dict(),
        })
    rows.sort(key=lambda r: (r["core_family_key"], r["candidate_identity"]))
    return rows


def registry_by_id(seed_root: int = 2026092642) -> dict[str, dict]:
    return {r["candidate_id"]: r for r in registry_rows(seed_root)}


def family_members(
    family_key: str,
    seed_root: int = 2026092642,
) -> list[dict]:
    out = [
        r for r in registry_rows(seed_root)
        if r["core_family_key"] == str(family_key)
    ]
    if len(out) != 16:
        raise ValueError(
            f"expected exactly 16 P42 phase members for {family_key}, got {len(out)}"
        )
    return out


def _config(row: dict, experiment_seed: int) -> PhasedMachineConfig:
    data = dict(row["config"])
    data["experiment_seed"] = int(experiment_seed)
    return PhasedMachineConfig(**data)


def simulate_candidate(
    row: dict,
    n_contests: int,
    experiment_seed: int,
) -> pd.DataFrame:
    cfg = _config(row, experiment_seed)
    calendar_offset = calendar_alignment_offset(cfg)
    frame = simulate_phased(int(n_contests) + calendar_offset, cfg)
    if calendar_offset:
        frame = frame.iloc[calendar_offset:].reset_index(drop=True)
    frame = frame.iloc[:int(n_contests)].reset_index(drop=True)
    return attach_public_schedule(
        frame,
        start_date="2013-01-21",
        launch_partial_first_day=True,
    )


def _feature_vector(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    return complete_fingerprint_vector(complete_fingerprint(frame))


def observed_complete_vector(real_train: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    return _feature_vector(real_train)


def candidate_signature(row: dict) -> str:
    payload = {
        "candidate_id": row["candidate_id"],
        "core_family_key": row["core_family_key"],
        "candidate_identity": row["candidate_identity"],
        "config": row["config"],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def streaming_train_evaluation(
    real_train: pd.DataFrame,
    candidate_id: str,
    reps: int,
    *,
    seed_root: int = TRAIN_SEED_ROOT,
    observed: tuple[list[str], np.ndarray] | None = None,
    registry_seed_root: int = 2026092642,
) -> dict:
    rows = registry_by_id(registry_seed_root)
    try:
        row = rows[str(candidate_id)]
    except KeyError as exc:
        raise KeyError(f"unknown P42 candidate_id {candidate_id}") from exc

    names_obs, obs = (
        observed_complete_vector(real_train)
        if observed is None
        else observed
    )

    reps = int(reps)
    if reps < 4:
        raise ValueError("reps must be >=4")
    split = reps // 2
    if split < 2:
        raise ValueError("normalization split must contain at least 2 replicates")

    sums = np.zeros(len(obs), dtype=float)
    sums_sq = np.zeros(len(obs), dtype=float)
    names_ref = None

    for r in range(split):
        frame = simulate_candidate(
            row,
            n_contests=len(real_train),
            experiment_seed=int(seed_root) + r,
        )
        names, vec = _feature_vector(frame)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise AssertionError("P42 feature schema changed across calibration")
        sums += vec
        sums_sq += vec * vec

    if names_ref != names_obs:
        raise ValueError("observed/P42 complete fingerprint schemas differ")

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
    for j, r in enumerate(range(split, reps)):
        frame = simulate_candidate(
            row,
            n_contests=len(real_train),
            experiment_seed=int(seed_root) + r,
        )
        names, vec = _feature_vector(frame)
        if names != names_ref:
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
        },
    }


def classify_candidate(result: dict, phase: str) -> str:
    p = float(result["test"]["p_max_stat"])
    reps = int(result["simulation_reps"])
    if phase == "development_triage":
        return (
            "TRIAGE_MISMATCH_REQUIRES_MEMBER_CONFIRMATION"
            if p <= TRIAGE_ALPHA
            else "TRIAGE_COMPATIBLE_REQUIRES_MEMBER_CONFIRMATION"
        )
    if phase == "member_confirmation":
        return (
            "CONFIRMATION_REJECTED_MEMBER"
            if p <= TRIAGE_ALPHA
            else "CONFIRMATION_SURVIVOR_REQUIRES_FINAL_TRAIN"
        )
    if phase == "final_train":
        if reps < FINAL_REPS_MIN:
            return "FINAL_TRAIN_INELIGIBLE_TOO_FEW_REPLICATES"
        return (
            "FINAL_TRAIN_REJECTED_AS_INCOMPATIBLE"
            if p <= FINAL_ALPHA
            else "FINAL_TRAIN_COMPATIBLE_ELIGIBLE_FOR_FREEZE"
        )
    raise ValueError(f"unknown phase {phase}")


def classify_family(
    family_results: list[dict],
    phase: str,
) -> dict:
    if not family_results:
        raise ValueError("empty family results")
    family_key = family_results[0]["core_family_key"]
    if any(r["core_family_key"] != family_key for r in family_results):
        raise ValueError("mixed core families")

    expected_ids = {
        r["candidate_id"] for r in family_members(family_key)
    }
    got_ids = {r["candidate_id"] for r in family_results}

    if phase in ("member_confirmation", "final_train") and got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        return {
            "core_family_key": family_key,
            "classification": "FAMILY_INCOMPLETE_FAIL_CLOSED",
            "missing_candidate_ids": missing,
            "extra_candidate_ids": extra,
        }

    labels = [classify_candidate(r, phase) for r in family_results]
    if phase == "development_triage":
        rejected_flags = [
            r["test"]["p_max_stat"] <= TRIAGE_ALPHA for r in family_results
        ]
    elif phase == "member_confirmation":
        rejected_flags = [
            r["test"]["p_max_stat"] <= TRIAGE_ALPHA for r in family_results
        ]
    elif phase == "final_train":
        if any(int(r["simulation_reps"]) < FINAL_REPS_MIN for r in family_results):
            return {
                "core_family_key": family_key,
                "classification": "FAMILY_FINAL_INELIGIBLE_TOO_FEW_REPLICATES",
                "member_classifications": labels,
            }
        rejected_flags = [
            r["test"]["p_max_stat"] <= FINAL_ALPHA for r in family_results
        ]
    else:
        raise ValueError(phase)

    return {
        "core_family_key": family_key,
        "member_count": len(family_results),
        "member_classifications": labels,
        "all_members_rejected": bool(all(rejected_flags)),
        "classification": (
            "FAMILY_REJECTED_ALL_MEMBERS_INCOMPATIBLE"
            if all(rejected_flags)
            else "FAMILY_RETAINED_COMPATIBLE_CLASS"
        ),
        "guardrail": "family rejection requires every frozen phase member to reject",
    }


def train_frame_from_full_history(df: pd.DataFrame) -> pd.DataFrame:
    return stage_frame(df, "train", unlock_holdout=False)
