from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import comb
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis import MAIN_COLS
from .combinatorics import N_COMPLEMENT_PAIRS, N_MAIN, canonical_pair_rank, combination_rank

FIXED_LAGS = (1, 17, 24, 168)
OCCUPANCY_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)


def _main_tuple(row: pd.Series) -> tuple[int, ...]:
    vals = tuple(sorted(int(row[c]) for c in MAIN_COLS))
    if len(vals) != 10 or len(set(vals)) != 10 or vals[0] < 1 or vals[-1] > 20:
        raise ValueError(f"invalid main result {vals}")
    return vals


@lru_cache(maxsize=1)
def _combination_rank_terms() -> np.ndarray:
    table = np.zeros((10, 21), dtype=np.int64)
    for position in range(1, 11):
        for value in range(1, 21):
            table[position - 1, value] = (
                comb(value - 1, position)
                if value - 1 >= position
                else 0
            )
    return table


def _validated_main_array(df: pd.DataFrame) -> np.ndarray:
    vals = np.sort(df[MAIN_COLS].to_numpy(dtype=np.int64), axis=1)
    if vals.shape != (len(df), 10):
        raise ValueError("expected 10 Main columns")
    if len(vals):
        if int(vals.min()) < 1 or int(vals.max()) > 20:
            raise ValueError("main number out of range")
        if np.any(np.diff(vals, axis=1) == 0):
            raise ValueError("main result must contain 10 unique numbers")
    return vals


def state_vectors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    main = _validated_main_array(df)
    if len(main):
        positions = np.arange(10, dtype=np.int64)[:, None]
        exact = _combination_rank_terms()[positions, main.T].sum(axis=0).astype(np.int64)
    else:
        exact = np.empty(0, dtype=np.int64)
    partition = np.minimum(exact, (N_MAIN - 1) - exact).astype(np.int64)
    numerone = pd.to_numeric(df["numerone"], errors="raise").to_numpy(dtype=np.int64)
    if np.any((numerone < 1) | (numerone > 20)):
        raise ValueError("Numerone out of range")
    return {"exact": exact, "partition": partition, "numerone": numerone}


def _checkpoints(n: int, fractions: Iterable[float]) -> list[int]:
    if n <= 0:
        return []
    points = {n}
    for f in fractions:
        f = float(f)
        if not 0 < f <= 1:
            raise ValueError("occupancy fractions must lie in (0,1]")
        points.add(max(1, min(n, int(round(n * f)))))
    return sorted(points)


def occupancy_trajectory(
    states: np.ndarray,
    space_size: int,
    fractions: Iterable[float] = OCCUPANCY_FRACTIONS,
) -> list[dict]:
    checkpoints = set(_checkpoints(len(states), fractions))
    seen: Counter[int] = Counter()
    distinct = 0
    collision_pairs = 0
    out: list[dict] = []
    for i, value in enumerate(states, start=1):
        old = seen[int(value)]
        collision_pairs += old
        if old == 0:
            distinct += 1
        seen[int(value)] = old + 1
        if i in checkpoints:
            expected_distinct = space_size * (1.0 - (1.0 - 1.0 / space_size) ** i)
            expected_pairs = i * (i - 1) / (2.0 * space_size)
            out.append({
                "n": int(i),
                "distinct": int(distinct),
                "unseen": int(space_size - distinct),
                "collision_pairs": int(collision_pairs),
                "uniform_expected_distinct": float(expected_distinct),
                "uniform_expected_collision_pairs": float(expected_pairs),
            })
    return out


def multiplicity_histogram(states: np.ndarray) -> dict[str, int]:
    counts = Counter(int(x) for x in states)
    hist = Counter(counts.values())
    return {
        "0": 0,
        "1": int(hist.get(1, 0)),
        "2": int(hist.get(2, 0)),
        "3": int(hist.get(3, 0)),
        "4": int(hist.get(4, 0)),
        "5_plus": int(sum(v for k, v in hist.items() if k >= 5)),
        "max": int(max(counts.values(), default=0)),
    }


def waiting_time_summary(states: np.ndarray) -> dict:
    last: dict[int, int] = {}
    gaps: list[int] = []
    for i, raw in enumerate(states):
        state = int(raw)
        if state in last:
            gaps.append(i - last[state])
        last[state] = i
    if not gaps:
        return {
            "repeat_intervals": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "min": None,
            "max": None,
        }
    a = np.asarray(gaps, dtype=float)
    return {
        "repeat_intervals": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q10": float(np.quantile(a, 0.10)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "q90": float(np.quantile(a, 0.90)),
        "min": int(np.min(a)),
        "max": int(np.max(a)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    a = x.astype(float)
    b = y.astype(float)
    a -= float(a.mean())
    b -= float(b.mean())
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def transition_fingerprint(
    exact: np.ndarray,
    partition: np.ndarray,
    numerone: np.ndarray,
    lags: Iterable[int] = FIXED_LAGS,
) -> list[dict]:
    rows: list[dict] = []
    for lag_raw in lags:
        lag = int(lag_raw)
        if lag <= 0 or lag >= len(exact):
            continue
        e1, e2 = exact[:-lag], exact[lag:]
        p1, p2 = partition[:-lag], partition[lag:]
        n1, n2 = numerone[:-lag], numerone[lag:]
        rows.append({
            "lag": lag,
            "n_pairs": int(len(e1)),
            "exact_equal_rate": float(np.mean(e1 == e2)),
            "partition_equal_rate": float(np.mean(p1 == p2)),
            "numerone_equal_rate": float(np.mean(n1 == n2)),
            "exact_rank_correlation": _pearson(e1 / float(N_MAIN - 1), e2 / float(N_MAIN - 1)),
            "partition_rank_correlation": _pearson(
                p1 / float(N_COMPLEMENT_PAIRS - 1),
                p2 / float(N_COMPLEMENT_PAIRS - 1),
            ),
        })
    return rows


def numerone_fingerprint(df: pd.DataFrame, numerone: np.ndarray) -> dict:
    counts = np.bincount(numerone, minlength=21)[1:21]
    main = _validated_main_array(df)
    membership = int(np.sum(np.any(main == numerone[:, None], axis=1))) if len(df) else 0
    n = len(df)
    return {
        "counts": {str(i + 1): int(v) for i, v in enumerate(counts)},
        "rates": {str(i + 1): float(v / n) if n else 0.0 for i, v in enumerate(counts)},
        "main_membership_hits": int(membership),
        "main_membership_rate": float(membership / n) if n else None,
        "uniform_expected_membership_rate": 0.5,
    }


def fingerprint(df: pd.DataFrame) -> dict:
    vec = state_vectors(df)
    exact = vec["exact"]
    partition = vec["partition"]
    numerone = vec["numerone"]
    return {
        "schema": "wfl-p23-fingerprint-1",
        "n": int(len(df)),
        "exact": {
            "space_size": N_MAIN,
            "trajectory": occupancy_trajectory(exact, N_MAIN),
            "multiplicity": multiplicity_histogram(exact),
            "waiting_times": waiting_time_summary(exact),
        },
        "partition": {
            "space_size": N_COMPLEMENT_PAIRS,
            "trajectory": occupancy_trajectory(partition, N_COMPLEMENT_PAIRS),
            "multiplicity": multiplicity_histogram(partition),
            "waiting_times": waiting_time_summary(partition),
        },
        "transitions": transition_fingerprint(exact, partition, numerone),
        "numerone": numerone_fingerprint(df, numerone),
        "guardrail": "descriptive fingerprint only; no candidate promotion from this function",
    }


def fingerprint_vector(report: dict) -> tuple[list[str], np.ndarray]:
    """Flatten the frozen P23 fingerprint into a deterministic scalar vector."""
    names: list[str] = []
    values: list[float] = []

    for space in ("exact", "partition"):
        for row in report[space]["trajectory"]:
            n = int(row["n"])
            for field in ("distinct", "collision_pairs"):
                names.append(f"{space}.trajectory.n{n}.{field}")
                values.append(float(row[field]))

        mult = report[space]["multiplicity"]
        for field in ("1", "2", "3", "4", "5_plus", "max"):
            names.append(f"{space}.multiplicity.{field}")
            values.append(float(mult[field]))

        wait = report[space]["waiting_times"]
        for field in ("repeat_intervals", "mean", "median", "q10", "q25", "q75", "q90", "min", "max"):
            names.append(f"{space}.waiting.{field}")
            raw = wait[field]
            values.append(-1.0 if raw is None else float(raw))

    by_lag = {int(row["lag"]): row for row in report["transitions"]}
    for lag in FIXED_LAGS:
        if lag not in by_lag:
            continue
        row = by_lag[lag]
        for field in (
            "exact_equal_rate",
            "partition_equal_rate",
            "numerone_equal_rate",
            "exact_rank_correlation",
            "partition_rank_correlation",
        ):
            names.append(f"transition.lag{lag}.{field}")
            values.append(float(row[field]))

    for n in range(1, 21):
        names.append(f"numerone.rate.{n}")
        values.append(float(report["numerone"]["rates"][str(n)]))
    names.append("numerone.main_membership_rate")
    membership = report["numerone"]["main_membership_rate"]
    values.append(-1.0 if membership is None else float(membership))

    return names, np.asarray(values, dtype=float)
