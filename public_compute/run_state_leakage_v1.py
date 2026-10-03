from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "state_leakage_v1_config.json"
DATA_DIR = ROOT / "data"
FULL_MASK = (1 << 20) - 1
PARTITION_STATES = 184756 // 2


def load_train() -> pd.DataFrame:
    frames = []
    for year in range(2013, 2019):
        p = DATA_DIR / f"wfl_train_{year}.csv"
        frames.append(pd.read_csv(p, dtype={"time": str}))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["time"] = df["time"].astype(str).str.zfill(5)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="raise").dt.hour
    return df.sort_values(["date", "hour", "contest"]).reset_index(drop=True)


def complete_days(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_cols = [f"n{i}" for i in range(1, 11)]
    days = []
    masks = []
    nums = []
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("hour")
        if len(g) != 17 or g["hour"].tolist() != list(range(7, 24)):
            continue
        arr = g[main_cols].to_numpy(dtype=np.int64)
        row_masks = np.zeros(17, dtype=np.uint32)
        for j in range(10):
            row_masks |= np.left_shift(np.uint32(1), (arr[:, j] - 1).astype(np.uint32))
        if np.any(np.fromiter((int(x).bit_count() for x in row_masks), dtype=np.int16) != 10):
            raise SystemExit("invalid main mask")
        days.append(np.datetime64(d.date()))
        masks.append(row_masks)
        nums.append(g["numerone"].to_numpy(dtype=np.int16))
    return np.asarray(days), np.stack(masks), np.stack(nums)


def bitcount_table() -> np.ndarray:
    x = np.arange(1 << 20, dtype=np.uint32)
    out = np.zeros(len(x), dtype=np.uint8)
    for shift in range(20):
        out += ((x >> np.uint32(shift)) & np.uint32(1)).astype(np.uint8)
    return out


def canonical_partition(m: np.ndarray) -> np.ndarray:
    c = np.bitwise_xor(m, np.uint32(FULL_MASK))
    return np.minimum(m, c)


def lag_rows(days, masks, nums, lags, bits):
    by_date = {d: i for i, d in enumerate(days)}
    result = []
    one_day = np.timedelta64(1, "D")
    for lag in lags:
        left = []
        right = []
        for i, d in enumerate(days):
            j = by_date.get(d - int(lag) * one_day)
            if j is not None:
                left.append(j)
                right.append(i)
        if not left:
            continue
        a = masks[np.asarray(left)].reshape(-1)
        b = masks[np.asarray(right)].reshape(-1)
        na = nums[np.asarray(left)].reshape(-1)
        nb = nums[np.asarray(right)].reshape(-1)
        n = len(a)
        overlap = bits[np.bitwise_and(a, b)].astype(float)
        exact_partition = canonical_partition(a) == canonical_partition(b)
        result.append({
            "day_lag": int(lag),
            "pair_count": int(n),
            "mean_main_overlap": float(overlap.mean()),
            "mean_overlap_minus_5": float(overlap.mean() - 5.0),
            "numerone_equal_count": int(np.sum(na == nb)),
            "numerone_equal_rate": float(np.mean(na == nb)),
            "exact_partition_collision_count": int(np.sum(exact_partition)),
            "exact_partition_collision_rate": float(np.mean(exact_partition)),
        })
    return result


def boundary_signature(masks, nums, bits):
    if len(masks) < 2:
        raise SystemExit("not enough complete days")
    within_a = masks[:, :-1].reshape(-1)
    within_b = masks[:, 1:].reshape(-1)
    boundary_a = masks[:-1, -1]
    boundary_b = masks[1:, 0]
    within_na = nums[:, :-1].reshape(-1)
    within_nb = nums[:, 1:].reshape(-1)
    boundary_na = nums[:-1, -1]
    boundary_nb = nums[1:, 0]
    wov = bits[np.bitwise_and(within_a, within_b)].astype(float)
    bov = bits[np.bitwise_and(boundary_a, boundary_b)].astype(float)
    return {
        "within_day_transition_count": int(len(wov)),
        "boundary_transition_count": int(len(bov)),
        "within_mean_overlap": float(wov.mean()),
        "boundary_mean_overlap": float(bov.mean()),
        "boundary_minus_within_overlap": float(bov.mean() - wov.mean()),
        "within_numerone_equal_rate": float(np.mean(within_na == within_nb)),
        "boundary_numerone_equal_rate": float(np.mean(boundary_na == boundary_nb)),
        "boundary_minus_within_numerone_equal": float(
            np.mean(boundary_na == boundary_nb) - np.mean(within_na == within_nb)
        ),
    }


def month_groups(days):
    months = days.astype("datetime64[M]")
    groups = []
    for m in np.unique(months):
        groups.append(np.where(months == m)[0])
    return groups


def permute_days_within_month(masks, nums, groups, rng):
    order = np.arange(len(masks))
    for idx in groups:
        order[idx] = rng.permutation(idx)
    return masks[order], nums[order]


def max_stats(rows):
    if not rows:
        return {"overlap": 0.0, "numerone": 0.0, "exact_collisions": 0}
    return {
        "overlap": float(max(abs(r["mean_overlap_minus_5"]) for r in rows)),
        "numerone": float(max(abs(r["numerone_equal_rate"] - 0.05) for r in rows)),
        "exact_collisions": int(max(r["exact_partition_collision_count"] for r in rows)),
    }


def empirical_p(observed, null_values, direction="ge"):
    if direction != "ge":
        raise ValueError(direction)
    return float((1 + sum(v >= observed for v in null_values)) / (1 + len(null_values)))


def main(out_path: str) -> None:
    cfg = json.loads(CFG_PATH.read_text())
    if cfg["scope"] != "TRAIN_ONLY_2013_2018":
        raise SystemExit("scope changed")
    if cfg["guardrails"]["validation_accessed"] is not False or cfg["guardrails"]["holdout_accessed"] is not False:
        raise SystemExit("guardrail breach")

    df = load_train()
    days, masks, nums = complete_days(df)
    bits = bitcount_table()
    lags = [int(x) for x in cfg["same_slot_day_lags"]]

    observed_rows = lag_rows(days, masks, nums, lags, bits)
    observed_max = max_stats(observed_rows)
    observed_boundary = boundary_signature(masks, nums, bits)

    reps = int(cfg["permutation"]["reps"])
    rng = np.random.default_rng(int(cfg["permutation"]["seed"]))
    groups = month_groups(days)

    null_overlap = []
    null_numerone = []
    null_exact = []
    null_boundary_overlap = []
    null_boundary_numerone = []

    for _ in range(reps):
        pm, pn = permute_days_within_month(masks, nums, groups, rng)
        rows = lag_rows(days, pm, pn, lags, bits)
        mx = max_stats(rows)
        null_overlap.append(mx["overlap"])
        null_numerone.append(mx["numerone"])
        null_exact.append(mx["exact_collisions"])
        b = boundary_signature(pm, pn, bits)
        null_boundary_overlap.append(abs(b["boundary_minus_within_overlap"]))
        null_boundary_numerone.append(abs(b["boundary_minus_within_numerone_equal"]))

    pvals = {
        "same_slot_overlap_max_abs": empirical_p(observed_max["overlap"], null_overlap),
        "same_slot_numerone_max_abs": empirical_p(observed_max["numerone"], null_numerone),
        "same_slot_exact_partition_collision_max": empirical_p(observed_max["exact_collisions"], null_exact),
        "day_boundary_overlap_abs": empirical_p(
            abs(observed_boundary["boundary_minus_within_overlap"]), null_boundary_overlap
        ),
        "day_boundary_numerone_abs": empirical_p(
            abs(observed_boundary["boundary_minus_within_numerone_equal"]), null_boundary_numerone
        ),
    }

    min_p = min(pvals.values())
    status = (
        "TRAIN_STRUCTURAL_SIGNAL_REQUIRES_REPLICATION"
        if min_p <= 0.01
        else "NO_STRONG_TRAIN_STATE_LEAKAGE_SIGNAL"
    )

    payload = {
        "schema": "wfl-state-leakage-v1-result",
        "status": status,
        "config": cfg,
        "data": {
            "input_rows_2013_2018": int(len(df)),
            "complete_17_draw_days": int(len(days)),
            "first_complete_day": str(days[0]),
            "last_complete_day": str(days[-1]),
            "validation_accessed": False,
            "holdout_accessed": False,
        },
        "null_reference": {
            "expected_main_overlap": 5.0,
            "expected_numerone_equal_rate": 0.05,
            "expected_exact_complement_invariant_partition_collision_rate": 1.0 / PARTITION_STATES,
            "partition_state_count": PARTITION_STATES,
        },
        "observed_by_same_slot_day_lag": observed_rows,
        "observed_max_stats": observed_max,
        "day_boundary_reseed_signature": observed_boundary,
        "familywise_empirical_p": pvals,
        "interpretation": (
            "This is a TRAIN-only structural diagnostic. A small p-value would indicate "
            "serial/calendar structure worth independent replication, not recovered RNG state "
            "and not predictive capability. No seed/state search is performed."
        ),
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)
