from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from .analysis import MAIN_COLS

FULL_MASK = (1 << 20) - 1
STATE_COUNT = comb(20, 10) // 2
MODULI = (2, 3, 5, 7, 11, 16, 31, 64, 127, 256)
STATIC_ENDPOINTS = ("BIN256",) + tuple(f"MOD_{m}" for m in MODULI)
SERIAL_LAGS = (1, 17)
WINDOWS = {
    "W1_DISCOVERY": ("2013-01-21", "2019-12-31"),
    "W2_CONFIRMATION": ("2020-01-01", "2026-09-24"),
}


def _mask_from_numbers(numbers: tuple[int, ...]) -> int:
    mask = 0
    for n in numbers:
        if not 1 <= int(n) <= 20:
            raise ValueError(f"main number out of range: {n}")
        mask |= 1 << (int(n) - 1)
    if mask.bit_count() != 10:
        raise ValueError(f"expected 10 distinct numbers, got {numbers}")
    return mask


def canonical_mask(mask: int) -> int:
    comp = FULL_MASK ^ int(mask)
    return min(int(mask), comp)


def canonical_state_table() -> tuple[np.ndarray, dict[int, int]]:
    states = set()
    for combo in combinations(range(1, 21), 10):
        states.add(canonical_mask(_mask_from_numbers(combo)))
    ordered = np.asarray(sorted(states), dtype=np.int64)
    if len(ordered) != STATE_COUNT:
        raise AssertionError(f"expected {STATE_COUNT} states, got {len(ordered)}")
    return ordered, {int(mask): i for i, mask in enumerate(ordered)}


def partition_ranks(df: pd.DataFrame) -> np.ndarray:
    _, mapping = canonical_state_table()
    ranks = np.empty(len(df), dtype=np.int64)
    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        nums = tuple(int(row[c]) for c in MAIN_COLS)
        mask = canonical_mask(_mask_from_numbers(nums))
        ranks[i] = mapping[mask]
    return ranks


def _expected_counts_for_categories(n: int, category_index: np.ndarray, k: int) -> np.ndarray:
    state_counts = np.bincount(category_index, minlength=k).astype(float)
    probs = state_counts / float(STATE_COUNT)
    return probs * float(n)


def _static_endpoint(ranks: np.ndarray, endpoint: str) -> dict:
    n = len(ranks)
    all_ranks = np.arange(STATE_COUNT, dtype=np.int64)
    if endpoint == "BIN256":
        k = 256
        cats = (ranks * k) // STATE_COUNT
        all_cats = (all_ranks * k) // STATE_COUNT
    elif endpoint.startswith("MOD_"):
        k = int(endpoint.split("_", 1)[1])
        cats = ranks % k
        all_cats = all_ranks % k
    else:
        raise ValueError(endpoint)

    observed = np.bincount(cats, minlength=k).astype(float)
    expected = _expected_counts_for_categories(n, all_cats, k)
    stat, p = chisquare(observed, f_exp=expected)
    residuals = (observed - expected) / np.sqrt(expected)

    return {
        "endpoint": endpoint,
        "cells": int(k),
        "n": int(n),
        "chi_square": float(stat),
        "p": float(p),
        "residuals": residuals.tolist(),
    }


def static_family(ranks: np.ndarray) -> dict:
    rows = [_static_endpoint(ranks, ep) for ep in STATIC_ENDPOINTS]
    family_size = len(rows)
    for row in rows:
        row["bonferroni_family_size"] = family_size
        row["p_corrected"] = min(1.0, row["p"] * family_size)
    return {"family_size": family_size, "endpoints": rows}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("invalid correlation vectors")
    xx = x.astype(float)
    yy = y.astype(float)
    xx -= xx.mean()
    yy -= yy.mean()
    den = np.sqrt(np.dot(xx, xx) * np.dot(yy, yy))
    if den == 0:
        return 0.0
    return float(np.dot(xx, yy) / den)


def _circular_shift_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = x.astype(float) - float(np.mean(x))
    yy = y.astype(float) - float(np.mean(y))
    den = np.sqrt(np.dot(xx, xx) * np.dot(yy, yy))
    if den == 0:
        return np.zeros(len(xx), dtype=float)
    fx = np.fft.rfft(xx)
    fy = np.fft.rfft(yy)
    corr = np.fft.irfft(np.conj(fx) * fy, n=len(xx))
    return corr / den


def serial_family(
    ranks: np.ndarray,
    reps: int = 4999,
    seed: int = 20260925,
) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    norm = ranks.astype(float) / float(STATE_COUNT - 1)

    for lag in SERIAL_LAGS:
        x = norm[:-lag]
        y = norm[lag:]
        observed = _pearson(x, y)
        shift_corr = _circular_shift_correlations(x, y)
        possible = np.arange(1, len(shift_corr), dtype=np.int64)
        if len(possible) == 0:
            raise ValueError("not enough data for serial null")
        offsets = rng.choice(possible, size=reps, replace=True)
        sims = shift_corr[offsets]
        center = float(np.mean(sims))
        extreme = int(np.sum(np.abs(sims - center) >= abs(observed - center) - 1e-15))
        p = float((1 + extreme) / (reps + 1))
        rows.append({
            "lag": int(lag),
            "observed_correlation": observed,
            "null_mean": center,
            "null_q025": float(np.quantile(sims, 0.025)),
            "null_q975": float(np.quantile(sims, 0.975)),
            "p": p,
        })

    for row in rows:
        row["bonferroni_family_size"] = len(rows)
        row["p_corrected"] = min(1.0, row["p"] * len(rows))
    return {"reps": int(reps), "endpoints": rows}


def _collision_stats(draws: np.ndarray) -> tuple[int, int, int]:
    counts = np.bincount(draws, minlength=STATE_COUNT)
    occupied = counts[counts > 0]
    pairs = int(np.sum(occupied * (occupied - 1) // 2))
    unique = int(len(occupied))
    max_mult = int(occupied.max()) if len(occupied) else 0
    return pairs, unique, max_mult


def collision_family(
    ranks: np.ndarray,
    reps: int = 4999,
    seed: int = 20260927,
    batch_size: int = 32,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = _collision_stats(ranks)
    sims = np.empty((reps, 3), dtype=float)

    done = 0
    while done < reps:
        b = min(batch_size, reps - done)
        block = rng.integers(0, STATE_COUNT, size=(b, len(ranks)), dtype=np.int32)
        for j in range(b):
            sims[done + j] = _collision_stats(block[j])
        done += b

    names = ("COLLISION_PAIRS", "UNIQUE_STATES", "MAX_MULTIPLICITY")
    rows = []
    for idx, name in enumerate(names):
        sim = sims[:, idx]
        obs = float(observed[idx])
        null_mean = float(np.mean(sim))
        if name == "MAX_MULTIPLICITY":
            extreme = int(np.sum(sim >= obs - 1e-15))
            p = float((1 + extreme) / (reps + 1))
            direction = "upper_only"
        else:
            extreme = int(np.sum(np.abs(sim - null_mean) >= abs(obs - null_mean) - 1e-15))
            p = float((1 + extreme) / (reps + 1))
            direction = "two_sided"
        rows.append({
            "endpoint": name,
            "observed": obs,
            "null_mean": null_mean,
            "null_q025": float(np.quantile(sim, 0.025)),
            "null_q975": float(np.quantile(sim, 0.975)),
            "p": p,
            "direction_test": direction,
            "effect_sign": int(np.sign(obs - null_mean)),
        })

    for row in rows:
        row["bonferroni_family_size"] = len(rows)
        row["p_corrected"] = min(1.0, row["p"] * len(rows))
    return {"reps": int(reps), "endpoints": rows}


def _window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    z = df.copy()
    dates = pd.to_datetime(z["date"], errors="raise")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return z.loc[mask].sort_values(["date", "time", "contest"]).reset_index(drop=True)


def _index_by(rows: list[dict], key: str) -> dict:
    return {str(row[key]): row for row in rows}


def replication_decision(w1: dict, w2: dict) -> dict:
    replicated = []

    s1 = _index_by(w1["static"]["endpoints"], "endpoint")
    s2 = _index_by(w2["static"]["endpoints"], "endpoint")
    static_checks = []
    for endpoint in STATIC_ENDPOINTS:
        a = s1[endpoint]
        b = s2[endpoint]
        corr = float(np.corrcoef(a["residuals"], b["residuals"])[0, 1])
        passed = (
            a["p_corrected"] <= 0.05
            and b["p_corrected"] <= 0.05
            and np.isfinite(corr)
            and corr > 0
        )
        row = {
            "endpoint": endpoint,
            "w1_p_corrected": a["p_corrected"],
            "w2_p_corrected": b["p_corrected"],
            "residual_correlation": corr,
            "replicated": bool(passed),
        }
        static_checks.append(row)
        if passed:
            replicated.append(f"STATIC:{endpoint}")

    q1 = {int(r["lag"]): r for r in w1["serial"]["endpoints"]}
    q2 = {int(r["lag"]): r for r in w2["serial"]["endpoints"]}
    serial_checks = []
    for lag in SERIAL_LAGS:
        a = q1[lag]
        b = q2[lag]
        same_sign = (
            np.sign(a["observed_correlation"]) == np.sign(b["observed_correlation"])
            and np.sign(a["observed_correlation"]) != 0
        )
        passed = (
            a["p_corrected"] <= 0.05
            and b["p_corrected"] <= 0.05
            and same_sign
        )
        row = {
            "lag": int(lag),
            "w1_p_corrected": a["p_corrected"],
            "w2_p_corrected": b["p_corrected"],
            "same_sign": bool(same_sign),
            "replicated": bool(passed),
        }
        serial_checks.append(row)
        if passed:
            replicated.append(f"SERIAL:LAG_{lag}")

    c1 = _index_by(w1["collisions"]["endpoints"], "endpoint")
    c2 = _index_by(w2["collisions"]["endpoints"], "endpoint")
    collision_checks = []
    for endpoint in ("COLLISION_PAIRS", "UNIQUE_STATES", "MAX_MULTIPLICITY"):
        a = c1[endpoint]
        b = c2[endpoint]
        if endpoint == "MAX_MULTIPLICITY":
            direction_ok = a["effect_sign"] > 0 and b["effect_sign"] > 0
        else:
            direction_ok = a["effect_sign"] == b["effect_sign"] and a["effect_sign"] != 0
        passed = (
            a["p_corrected"] <= 0.05
            and b["p_corrected"] <= 0.05
            and direction_ok
        )
        row = {
            "endpoint": endpoint,
            "w1_p_corrected": a["p_corrected"],
            "w2_p_corrected": b["p_corrected"],
            "direction_compatible": bool(direction_ok),
            "replicated": bool(passed),
        }
        collision_checks.append(row)
        if passed:
            replicated.append(f"COLLISION:{endpoint}")

    return {
        "static": static_checks,
        "serial": serial_checks,
        "collisions": collision_checks,
        "replicated_endpoints": replicated,
        "status": (
            "SIGNAL_REQUIRES_FORWARD_CONFIRMATION"
            if replicated
            else "NO_COMPLEMENTARY_STATE_FINGERPRINT_DEMONSTRATED"
        ),
    }


def p17_report(
    df: pd.DataFrame,
    serial_reps: int = 4999,
    collision_reps: int = 4999,
    seed: int = 20260925,
) -> dict:
    windows = {}
    for i, (name, (start, end)) in enumerate(WINDOWS.items()):
        frame = _window(df, start, end)
        ranks = partition_ranks(frame)
        windows[name] = {
            "start": start,
            "end": end,
            "n": int(len(frame)),
            "static": static_family(ranks),
            "serial": serial_family(ranks, reps=serial_reps, seed=seed + 10 * i),
            "collisions": collision_family(
                ranks,
                reps=collision_reps,
                seed=seed + 100 + 10 * i,
            ),
        }

    decision = replication_decision(
        windows["W1_DISCOVERY"],
        windows["W2_CONFIRMATION"],
    )
    return {
        "schema": "wfl-p17-complementary-state-fingerprint-1",
        "state_count": STATE_COUNT,
        "canonicalization": "min(mask, full_mask XOR mask), then integer-sorted contiguous rank",
        "windows": windows,
        "decision": decision,
        "status": decision["status"],
        "guardrails": {
            "no_v1_p11_retuning": True,
            "historical_signal_requires_forward_confirmation": True,
            "rng_version_inference_from_statistics_alone": False,
        },
    }
