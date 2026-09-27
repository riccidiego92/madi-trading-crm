from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import sqrt

import numpy as np
import pandas as pd

from .analysis import MAIN_COLS, inclusion_matrix
from .p17_fingerprint import FULL_MASK, MODULI, STATE_COUNT, canonical_mask, canonical_state_table
from .p23_metrics import FIXED_LAGS, fingerprint, fingerprint_vector

PAIR_INDEX = tuple(combinations(range(20), 2))
PROJECTION_DIM = 32
PROJECTION_SEED = 20260925
BIN_COUNT = 256
VERIFIED_CLASSICO_RNG_BREAKPOINTS: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _p17_rank_mapping() -> dict[int, int]:
    _, mapping = canonical_state_table()
    return mapping


def _mask_from_main(values: np.ndarray) -> int:
    mask = 0
    for raw in values:
        n = int(raw)
        if not 1 <= n <= 20:
            raise ValueError(f"main number out of range {n}")
        mask |= 1 << (n - 1)
    if mask.bit_count() != 10:
        raise ValueError("expected 10 unique main numbers")
    return mask


@lru_cache(maxsize=1)
def _p17_rank_lookup() -> np.ndarray:
    ordered, _ = canonical_state_table()
    lookup = np.full(1 << 20, -1, dtype=np.int32)
    lookup[ordered.astype(np.int64)] = np.arange(len(ordered), dtype=np.int32)
    return lookup


def p17_partition_ranks(df: pd.DataFrame) -> np.ndarray:
    arr = np.sort(df[MAIN_COLS].to_numpy(dtype=np.int64), axis=1)
    if arr.shape != (len(df), 10):
        raise ValueError("expected 10 Main columns")
    if len(arr):
        if int(arr.min()) < 1 or int(arr.max()) > 20:
            raise ValueError("main number out of range")
        if np.any(np.diff(arr, axis=1) == 0):
            raise ValueError("expected 10 unique main numbers")
    masks = np.sum(
        np.left_shift(np.int64(1), arr - 1),
        axis=1,
        dtype=np.int64,
    )
    canonical = np.minimum(masks, np.bitwise_xor(FULL_MASK, masks))
    out = _p17_rank_lookup()[canonical]
    if np.any(out < 0):
        raise AssertionError("canonical mask missing from P17 lookup")
    return out.astype(np.int64)


def _pearson_residual_family(cats: np.ndarray, all_cats: np.ndarray, k: int) -> dict:
    n = len(cats)
    observed = np.bincount(cats, minlength=k).astype(float)
    state_counts = np.bincount(all_cats, minlength=k).astype(float)
    expected = n * state_counts / float(STATE_COUNT)
    residuals = np.zeros(k, dtype=float)
    keep = expected > 0
    residuals[keep] = (observed[keep] - expected[keep]) / np.sqrt(expected[keep])
    chi2 = float(np.sum(residuals[keep] ** 2))
    dfree = max(1, int(np.sum(keep)) - 1)
    return {
        "cells": int(k),
        "chi2": chi2,
        "df": dfree,
        "chi2_per_df": chi2 / float(dfree),
        "residuals": residuals.tolist(),
    }


def rank_residue_fingerprint(df: pd.DataFrame) -> dict:
    ranks = p17_partition_ranks(df)
    all_ranks = np.arange(STATE_COUNT, dtype=np.int64)
    bins = (ranks * BIN_COUNT) // STATE_COUNT
    all_bins = (all_ranks * BIN_COUNT) // STATE_COUNT
    out = {
        "BIN256": _pearson_residual_family(bins, all_bins, BIN_COUNT),
        "moduli": {},
    }
    for modulus in MODULI:
        out["moduli"][str(modulus)] = _pearson_residual_family(
            ranks % modulus,
            all_ranks % modulus,
            int(modulus),
        )
    return out


@lru_cache(maxsize=1)
def fixed_projection() -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    signs = rng.integers(
        0, 2, size=(len(PAIR_INDEX), PROJECTION_DIM), dtype=np.int8
    ).astype(float)
    signs = signs * 2.0 - 1.0
    return signs / np.sqrt(float(len(PAIR_INDEX)))


def projected_main_features(df: pd.DataFrame) -> np.ndarray:
    x = inclusion_matrix(df)
    pairs = np.asarray(PAIR_INDEX, dtype=np.int64)
    same_side = (x[:, pairs[:, 0]] == x[:, pairs[:, 1]]).astype(float)
    raw = same_side @ fixed_projection()
    mean = raw.mean(axis=0)
    sd = raw.std(axis=0, ddof=0)
    keep = sd > 1e-12
    if not np.all(keep):
        raise ValueError("degenerate fixed projected Main feature")
    return (raw - mean) / sd


def main_marginal_fingerprint(df: pd.DataFrame) -> dict:
    """Exact-side 10-of-20 marginal inclusion diagnostics.

    Under the exact-uniform 10-of-20 null each number has inclusion probability
    1/2, with fixed-row-sum covariance. The global statistic uses the 19-dim
    sum-zero covariance eigenvalue N*5/19.
    """
    x = inclusion_matrix(df).astype(float)
    n = len(x)
    if n <= 0:
        raise ValueError("empty frame")
    counts = x.sum(axis=0)
    rates = counts / float(n)
    expected = n / 2.0
    raw = float(np.sum((counts - expected) ** 2))
    eig = n * 5.0 / 19.0
    chi2_equivalent = raw / eig if eig > 0 else 0.0
    return {
        "counts": {str(i + 1): int(v) for i, v in enumerate(counts)},
        "rates": {str(i + 1): float(v) for i, v in enumerate(rates)},
        "global_chi2_equivalent": float(chi2_equivalent),
        "df": 19,
        "expected_rate": 0.5,
    }


def _numerone_labels(df: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(df["numerone"], errors="raise").to_numpy(dtype=np.int64) - 1
    if np.any((y < 0) | (y >= 20)):
        raise ValueError("Numerone outside 1..20")
    return y


def projected_main_numerone_omnibus(z: np.ndarray, labels: np.ndarray) -> float:
    if len(z) != len(labels) or len(labels) == 0:
        raise ValueError("projected features and labels must align")
    counts = np.bincount(labels, minlength=20).astype(float)
    p = counts / float(len(labels))
    active = (p > 0) & (p < 1)
    if not np.any(active):
        return 0.0
    scale = np.sqrt(p[active] * (1.0 - p[active]))
    onehot = np.zeros((len(labels), 20), dtype=float)
    onehot[np.arange(len(labels)), labels] = 1.0
    centered = onehot[:, active] - p[active][None, :]
    corr = (z.T @ centered) / float(len(labels))
    corr = corr / scale[None, :]
    return float(np.sum(corr * corr))


def _cramers_v(a: np.ndarray, b: np.ndarray, a_levels: int, b_levels: int) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    table = np.bincount(
        a.astype(np.int64) * int(b_levels) + b.astype(np.int64),
        minlength=int(a_levels) * int(b_levels),
    ).reshape(int(a_levels), int(b_levels)).astype(float)
    keep_r = table.sum(axis=1) > 0
    keep_c = table.sum(axis=0) > 0
    table = table[np.ix_(keep_r, keep_c)]
    n = float(table.sum())
    if n <= 0 or min(table.shape) <= 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    mask = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[mask] / expected[mask]))
    denom = n * float(min(table.shape[0] - 1, table.shape[1] - 1))
    return sqrt(chi2 / denom) if denom > 0 else 0.0


def _public_slots(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "date" in df.columns and "time" in df.columns:
        hour = pd.to_datetime(
            df["time"].astype(str), format="%H:%M", errors="raise"
        ).dt.hour.to_numpy(dtype=np.int64)
        dow = pd.to_datetime(df["date"].astype(str), errors="raise").dt.dayofweek.to_numpy(
            dtype=np.int64
        )
        return hour, dow
    if "synthetic_day" in df.columns and "synthetic_contest_in_day" in df.columns:
        hour = 7 + pd.to_numeric(
            df["synthetic_contest_in_day"], errors="raise"
        ).to_numpy(dtype=np.int64)
        dow = (
            pd.to_numeric(df["synthetic_day"], errors="raise").to_numpy(dtype=np.int64)
            % 7
        )
        return hour, dow
    raise ValueError("frame lacks public time-slot metadata")


def _main_group_dependence(
    df: pd.DataFrame, group: np.ndarray, group_min: int, group_count: int
) -> dict:
    x = inclusion_matrix(df)
    g = group.astype(np.int64) - int(group_min)
    values = np.empty(20, dtype=float)
    for j in range(20):
        values[j] = _cramers_v(g, x[:, j].astype(np.int64), group_count, 2)
    return {
        "mean_cramers_v": float(np.mean(values)),
        "max_cramers_v": float(np.max(values)),
        "per_number": values.tolist(),
    }


def extended_observables(df: pd.DataFrame) -> dict:
    z = projected_main_features(df)
    labels = _numerone_labels(df)
    hour, dow = _public_slots(df)

    numerone_serial = {}
    lagged_main_numerone = {}
    for lag in FIXED_LAGS:
        if lag >= len(df):
            continue
        numerone_serial[str(lag)] = {
            "cramers_v": _cramers_v(labels[:-lag], labels[lag:], 20, 20)
        }
        lagged_main_numerone[str(lag)] = {
            "omnibus": projected_main_numerone_omnibus(z[:-lag], labels[lag:])
        }

    hour_index = hour - 7
    if np.any((hour_index < 0) | (hour_index >= 17)):
        raise ValueError("hour outside fixed 07..23 stratification")

    return {
        "main_marginal": main_marginal_fingerprint(df),
        "rank_residue": rank_residue_fingerprint(df),
        "same_contest_main_numerone": {
            "omnibus": projected_main_numerone_omnibus(z, labels),
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
        },
        "numerone_serial": numerone_serial,
        "lagged_main_numerone": lagged_main_numerone,
        "stratification": {
            "hour_numerone_cramers_v": _cramers_v(hour_index, labels, 17, 20),
            "weekday_numerone_cramers_v": _cramers_v(dow, labels, 7, 20),
            "hour_main": _main_group_dependence(df, hour, 7, 17),
            "weekday_main": _main_group_dependence(df, dow, 0, 7),
        },
        "breakpoints": {
            "verified_classico_rng_boundaries": list(VERIFIED_CLASSICO_RNG_BREAKPOINTS),
            "status": "NO_VERIFIED_CLASSICO_RNG_BOUNDARY",
        },
    }


def complete_fingerprint(df: pd.DataFrame) -> dict:
    return {
        "schema": "wfl-p29-complete-fingerprint-1",
        "base": fingerprint(df),
        "extended": extended_observables(df),
        "guardrails": {
            "no_regulatory_or_concession_date_as_rng_boundary": True,
            "no_post_result_endpoint_choice": True,
            "operational_seed_search": False,
        },
    }


def complete_fingerprint_vector(report: dict) -> tuple[list[str], np.ndarray]:
    base_names, base_values = fingerprint_vector(report["base"])
    names = list(base_names)
    values = [float(x) for x in base_values]

    main_marginal = report["extended"]["main_marginal"]
    names.append("main_marginal.global_chi2_equivalent")
    values.append(float(main_marginal["global_chi2_equivalent"]))
    for number in range(1, 21):
        names.append(f"main_marginal.rate.{number}")
        values.append(float(main_marginal["rates"][str(number)]))

    residue = report["extended"]["rank_residue"]
    families = [("BIN256", residue["BIN256"])]
    families.extend(
        (f"MOD_{m}", residue["moduli"][str(m)]) for m in MODULI
    )
    for label, row in families:
        names.append(f"rank_residue.{label}.chi2_per_df")
        values.append(float(row["chi2_per_df"]))
        for cell, raw in enumerate(row["residuals"]):
            names.append(f"rank_residue.{label}.residual.{cell}")
            values.append(float(raw))

    names.append("main_numerone.same_contest.omnibus")
    values.append(
        float(report["extended"]["same_contest_main_numerone"]["omnibus"])
    )

    for lag in FIXED_LAGS:
        key = str(lag)
        if key in report["extended"]["numerone_serial"]:
            names.append(f"numerone_serial.lag{lag}.cramers_v")
            values.append(
                float(report["extended"]["numerone_serial"][key]["cramers_v"])
            )
        if key in report["extended"]["lagged_main_numerone"]:
            names.append(f"main_to_numerone.lag{lag}.omnibus")
            values.append(
                float(report["extended"]["lagged_main_numerone"][key]["omnibus"])
            )

    strat = report["extended"]["stratification"]
    for field in (
        "hour_numerone_cramers_v",
        "weekday_numerone_cramers_v",
    ):
        names.append(f"stratification.{field}")
        values.append(float(strat[field]))

    for prefix in ("hour_main", "weekday_main"):
        for field in ("mean_cramers_v", "max_cramers_v"):
            names.append(f"stratification.{prefix}.{field}")
            values.append(float(strat[prefix][field]))

    return names, np.asarray(values, dtype=float)
