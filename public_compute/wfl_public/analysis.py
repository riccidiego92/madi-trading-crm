from __future__ import annotations

from collections import Counter
from math import comb
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, chisquare

from .combinatorics import (
    N_MAIN, N_COMPLEMENT_PAIRS, JACKPOT_ODDS_EUR2,
    combination_rank, canonical_pair_rank, joint_jackpot_state,
    pair_match_count, symmetric_pair_match_probability,
)

MAIN_COLS = [f"n{i}" for i in range(1, 11)]


def inclusion_matrix(df: pd.DataFrame) -> np.ndarray:
    vals = df[MAIN_COLS].to_numpy(dtype=int)
    m = np.zeros((len(df), 20), dtype=np.int8)
    if len(df):
        rows = np.arange(len(df), dtype=np.int64)[:, None]
        m[rows, vals - 1] = 1
    return m


def marginal_uniformity(df: pd.DataFrame) -> dict:
    """Global marginal test respecting fixed row sum of 10.

    Under the 10-of-20 null, each indicator has p=.5 and pairwise covariance is negative.
    On the 19-dimensional sum-zero subspace, covariance eigenvalue per draw is 5/19.
    Therefore sum_i(count_i-N/2)^2 / (N*5/19) is asymptotically chi-square_19.
    """
    x = inclusion_matrix(df)
    n = len(x)
    counts = x.sum(axis=0)
    expected = n / 2.0
    stat_raw = float(np.sum((counts - expected) ** 2))
    eig = n * 5.0 / 19.0
    stat = stat_raw / eig if eig > 0 else np.nan
    p = float(chi2.sf(stat, 19)) if n else np.nan
    return {
        "n_draws": n,
        "counts": {str(i + 1): int(v) for i, v in enumerate(counts)},
        "expected_each": expected,
        "chi2_equivalent": stat,
        "df": 19,
        "p_value_asymptotic": p,
    }


def numerone_uniformity(df: pd.DataFrame) -> dict:
    counts = np.bincount(df["numerone"].to_numpy(dtype=int), minlength=21)[1:21]
    if len(df) == 0:
        return {"n_draws": 0}
    test = chisquare(counts, f_exp=np.repeat(len(df) / 20.0, 20))
    return {
        "n_draws": int(len(df)),
        "counts": {str(i + 1): int(v) for i, v in enumerate(counts)},
        "chi2": float(test.statistic),
        "df": 19,
        "p_value": float(test.pvalue),
    }


def numerone_membership(df: pd.DataFrame) -> dict:
    hits = 0
    for _, r in df.iterrows():
        main = {int(r[c]) for c in MAIN_COLS}
        hits += int(int(r["numerone"]) in main)
    n = len(df)
    p = float(binomtest(hits, n, p=0.5, alternative="two-sided").pvalue) if n else np.nan
    return {"n_draws": n, "hits": hits, "rate": hits / n if n else np.nan, "null_rate": 0.5, "p_value": p}


def overlap_null_probs() -> dict[int, float]:
    den = comb(20, 10)
    return {k: comb(10, k) * comb(10, 10 - k) / den for k in range(11)}


def consecutive_overlap(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"n_transitions": 0}
    z = df.copy()
    z["ts"] = pd.to_datetime(z["date"].astype(str) + " " + z["time"].astype(str))
    z = z.sort_values("ts")
    vals = [set(map(int, row)) for row in z[MAIN_COLS].to_numpy()]
    overlaps = [len(a & b) for a, b in zip(vals[:-1], vals[1:])]
    obs = Counter(overlaps)
    probs = overlap_null_probs()
    return {
        "n_transitions": len(overlaps),
        "mean_overlap": float(np.mean(overlaps)),
        "expected_mean": 5.0,
        "observed": {str(k): int(obs.get(k, 0)) for k in range(11)},
        "expected_probability": {str(k): probs[k] for k in range(11)},
    }


def _bh_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n, dtype=float)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        running = min(running, p[idx] * n / rank)
        out[idx] = min(1.0, running)
    return out.tolist()


def hour_effects(df: pd.DataFrame) -> dict:
    z = df.copy()
    z["hour"] = pd.to_datetime(z["time"].astype(str), format="%H:%M", errors="coerce").dt.hour
    x = inclusion_matrix(z)
    hours = sorted(int(h) for h in z["hour"].dropna().unique())
    pvals: list[float] = []
    stats: list[dict] = []
    for j in range(20):
        table = []
        for h in hours:
            mask = z["hour"].to_numpy() == h
            yes = int(x[mask, j].sum())
            no = int(mask.sum() - yes)
            table.append([yes, no])
        arr = np.asarray(table, dtype=float)
        if arr.shape[0] < 2 or np.any(arr.sum(axis=0) == 0):
            p = 1.0
            stat = 0.0
        else:
            from scipy.stats import chi2_contingency
            stat, p, _, _ = chi2_contingency(arr, correction=False)
        pvals.append(float(p))
        stats.append({"number": j + 1, "chi2": float(stat), "p_raw": float(p)})
    adj = _bh_adjust(pvals)
    for s, q in zip(stats, adj):
        s["q_bh"] = q
    return {"hours": hours, "number_tests": stats}


def pair_cooccurrence(df: pd.DataFrame) -> dict:
    x = inclusion_matrix(df)
    n = len(x)
    p_pair = 10 * 9 / (20 * 19)
    mean = n * p_pair
    var = n * p_pair * (1 - p_pair)
    rows: list[dict] = []
    pvals: list[float] = []
    for i in range(20):
        for j in range(i + 1, 20):
            obs = int(np.sum(x[:, i] * x[:, j]))
            z = (obs - mean) / np.sqrt(var) if var > 0 else 0.0
            p = float(chi2.sf(z * z, 1))
            rows.append({"a": i + 1, "b": j + 1, "observed": obs, "expected": mean, "z": float(z), "p_raw": p})
            pvals.append(p)
    adj = _bh_adjust(pvals)
    for r, q in zip(rows, adj):
        r["q_bh"] = q
    rows.sort(key=lambda r: r["q_bh"])
    return {"null_pair_probability": p_pair, "top": rows[:25], "n_tests": len(rows)}


def triple_cooccurrence(df: pd.DataFrame) -> dict:
    from itertools import combinations
    from scipy.stats import binomtest

    x = inclusion_matrix(df)
    n = len(x)
    p_tri = (10 / 20) * (9 / 19) * (8 / 18)
    rows: list[dict] = []
    pvals: list[float] = []
    for a, b, c in combinations(range(20), 3):
        obs = int(np.sum(x[:, a] * x[:, b] * x[:, c]))
        p = float(binomtest(obs, n, p=p_tri, alternative="two-sided").pvalue) if n else 1.0
        rows.append({
            "a": a + 1,
            "b": b + 1,
            "c": c + 1,
            "observed": obs,
            "expected": n * p_tri,
            "p_raw": p,
        })
        pvals.append(p)
    adj = _bh_adjust(pvals)
    m = len(rows)
    for r, q in zip(rows, adj):
        r["q_bh"] = q
        r["p_bonferroni"] = min(1.0, r["p_raw"] * m)
    rows.sort(key=lambda r: (r["p_bonferroni"], r["p_raw"]))
    return {"null_triple_probability": p_tri, "top": rows[:25], "n_tests": m}


def lag_overlap(df: pd.DataFrame, lags: Iterable[int] = (1, 2, 3, 4, 5, 17, 34, 119)) -> dict:
    z = df.copy()
    z["ts"] = pd.to_datetime(z["date"].astype(str) + " " + z["time"].astype(str))
    z = z.sort_values("ts")
    sets = [set(map(int, r)) for r in z[MAIN_COLS].to_numpy()]
    out = []
    for lag in lags:
        lag = int(lag)
        if lag <= 0 or lag >= len(sets):
            continue
        v = [len(sets[i] & sets[i - lag]) for i in range(lag, len(sets))]
        out.append({"lag": lag, "n": len(v), "mean_overlap": float(np.mean(v)), "null_mean": 5.0})
    return {"lags": out}



def _occupancy_summary(states: list[int], space_size: int) -> dict:
    counts = Counter(states)
    freq_of_freq = Counter(counts.values())
    collisions = sum(v * (v - 1) // 2 for v in counts.values())
    n = len(states)
    return {
        "n": int(n),
        "space_size": int(space_size),
        "distinct": int(len(counts)),
        "unseen": int(space_size - len(counts)),
        "singletons": int(freq_of_freq.get(1, 0)),
        "doubletons": int(freq_of_freq.get(2, 0)),
        "triple_plus_states": int(sum(v for k, v in freq_of_freq.items() if k >= 3)),
        "max_occupancy": int(max(counts.values(), default=0)),
        "collision_pairs": int(collisions),
        "null_expected_collision_pairs": float(n * (n - 1) / (2.0 * space_size)),
    }


def state_space_occupancy(df: pd.DataFrame) -> dict:
    exact_states: list[int] = []
    pair_states: list[int] = []
    joint_states: list[int] = []
    for _, row in df.iterrows():
        main = tuple(int(row[c]) for c in MAIN_COLS)
        num = int(row["numerone"])
        exact_states.append(combination_rank(main))
        pair_states.append(canonical_pair_rank(main))
        joint_states.append(joint_jackpot_state(main, num))

    def repeat_lags(states: list[int], lags=(1, 2, 3, 4, 5, 17, 34, 119)) -> list[dict]:
        out = []
        a = np.asarray(states, dtype=np.int64)
        for lag in lags:
            if lag >= len(a):
                continue
            observed = int(np.sum(a[lag:] == a[:-lag]))
            out.append({"lag": int(lag), "observed": observed, "n_pairs": int(len(a) - lag)})
        return out

    return {
        "exact_10set": {
            **_occupancy_summary(exact_states, N_MAIN),
            "repeat_lags": repeat_lags(exact_states),
        },
        "complement_pair": {
            **_occupancy_summary(pair_states, N_COMPLEMENT_PAIRS),
            "repeat_lags": repeat_lags(pair_states),
        },
        "joint_pair_numerone": {
            **_occupancy_summary(joint_states, JACKPOT_ODDS_EUR2),
            "repeat_lags": repeat_lags(joint_states),
        },
    }

def baseline_report(df: pd.DataFrame) -> dict:
    return {
        "marginal_uniformity": marginal_uniformity(df),
        "numerone_uniformity": numerone_uniformity(df),
        "numerone_membership": numerone_membership(df),
        "consecutive_overlap": consecutive_overlap(df),
        "hour_effects": hour_effects(df),
        "pair_cooccurrence": pair_cooccurrence(df),
        "triple_cooccurrence": triple_cooccurrence(df),
        "lag_overlap": lag_overlap(df),
        "state_space_occupancy": state_space_occupancy(df),
        "null_symmetric_pair_match": {str(m): symmetric_pair_match_probability(m) for m in range(5, 11)},
    }
