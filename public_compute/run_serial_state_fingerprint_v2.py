from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "serial_state_fingerprint_v2_config.json"
DATA_DIR = ROOT / "data"
FULL_MASK = (1 << 20) - 1


def load_train() -> pd.DataFrame:
    frames = [pd.read_csv(DATA_DIR / f"wfl_train_{y}.csv", dtype={"time": str}) for y in range(2013, 2019)]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["time"] = df["time"].astype(str).str.zfill(5)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="raise").dt.hour
    return df.sort_values(["date", "hour", "contest"]).reset_index(drop=True)


def complete_days(df: pd.DataFrame):
    main_cols = [f"n{i}" for i in range(1, 11)]
    days, masks, nums = [], [], []
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("hour")
        if len(g) != 17 or g["hour"].tolist() != list(range(7, 24)):
            continue
        arr = g[main_cols].to_numpy(dtype=np.int64)
        row_masks = np.zeros(17, dtype=np.uint32)
        for j in range(10):
            row_masks |= np.left_shift(np.uint32(1), (arr[:, j] - 1).astype(np.uint32))
        if any(int(x).bit_count() != 10 for x in row_masks):
            raise SystemExit("invalid Main mask")
        days.append(np.datetime64(d.date()))
        masks.append(row_masks)
        nums.append(g["numerone"].to_numpy(dtype=np.int16))
    return np.asarray(days), np.stack(masks), np.stack(nums)


def canonical_masks(m):
    c = np.bitwise_xor(m, np.uint32(FULL_MASK))
    return np.minimum(m, c)


def state_table():
    vals = set()
    for comb in itertools.combinations(range(20), 10):
        mask = 0
        for j in comb:
            mask |= 1 << j
        vals.add(min(mask, FULL_MASK ^ mask))
    out = np.asarray(sorted(vals), dtype=np.uint32)
    if len(out) != 92378:
        raise SystemExit(f"unexpected partition state count {len(out)}")
    return out


def ranks_for_masks(masks_flat, states):
    canon = canonical_masks(masks_flat)
    ranks = np.searchsorted(states, canon).astype(np.int64)
    if np.any(states[ranks] != canon):
        raise SystemExit("canonical partition lookup failure")
    return ranks, canon


def bitcount_table():
    x = np.arange(1 << 20, dtype=np.uint32)
    out = np.zeros(len(x), dtype=np.uint8)
    for shift in range(20):
        out += ((x >> np.uint32(shift)) & np.uint32(1)).astype(np.uint8)
    return out


def cramers_v(a, b, k):
    tab = np.bincount(a.astype(np.int64) * k + b.astype(np.int64), minlength=k*k).reshape(k, k).astype(float)
    rs = tab.sum(axis=1)
    cs = tab.sum(axis=0)
    keep_r = rs > 0
    keep_c = cs > 0
    tab = tab[np.ix_(keep_r, keep_c)]
    n = float(tab.sum())
    if n <= 0 or min(tab.shape) <= 1:
        return 0.0
    expected = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
    mask = expected > 0
    chi2 = float(np.sum(((tab - expected) ** 2)[mask] / expected[mask]))
    denom = n * float(min(tab.shape[0] - 1, tab.shape[1] - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def evaluate(masks_flat, nums_flat, states, bits, lags, moduli):
    ranks, canon = ranks_for_masks(masks_flat, states)
    rows = []
    for lag in lags:
        if lag >= len(ranks):
            continue
        ra, rb = ranks[:-lag], ranks[lag:]
        ca, cb = canon[:-lag], canon[lag:]
        ma, mb = masks_flat[:-lag], masks_flat[lag:]
        na, nb = nums_flat[:-lag] - 1, nums_flat[lag:] - 1
        overlap = bits[np.bitwise_and(ma, mb)].astype(float)
        residue = {}
        for mod in moduli:
            residue[str(mod)] = cramers_v(ra % mod, rb % mod, mod)
        rows.append({
            "contest_lag": int(lag),
            "pair_count": int(len(ra)),
            "partition_residue_cramers_v": residue,
            "numerone_cramers_v": cramers_v(na, nb, 20),
            "mean_main_overlap": float(overlap.mean()),
            "mean_overlap_minus_5": float(overlap.mean() - 5.0),
            "exact_partition_collision_count": int(np.sum(ca == cb)),
            "exact_partition_collision_rate": float(np.mean(ca == cb)),
        })
    residue_top = None
    for row in rows:
        for mod, v in row["partition_residue_cramers_v"].items():
            item = {"contest_lag": row["contest_lag"], "modulus": int(mod), "cramers_v": float(v)}
            if residue_top is None or item["cramers_v"] > residue_top["cramers_v"]:
                residue_top = item
    top_num = max(rows, key=lambda r: r["numerone_cramers_v"])
    top_ov = max(rows, key=lambda r: abs(r["mean_overlap_minus_5"]))
    top_col = max(rows, key=lambda r: r["exact_partition_collision_count"])
    maxima = {
        "partition_residue_cramers_v": float(residue_top["cramers_v"]),
        "numerone_cramers_v": float(top_num["numerone_cramers_v"]),
        "abs_overlap_minus_5": float(abs(top_ov["mean_overlap_minus_5"])),
        "exact_partition_collision_count": int(top_col["exact_partition_collision_count"]),
    }
    tops = {
        "partition_residue": residue_top,
        "numerone": {"contest_lag": int(top_num["contest_lag"]), "cramers_v": float(top_num["numerone_cramers_v"])},
        "overlap": {"contest_lag": int(top_ov["contest_lag"]), "mean_overlap_minus_5": float(top_ov["mean_overlap_minus_5"])},
        "exact_collision": {"contest_lag": int(top_col["contest_lag"]), "count": int(top_col["exact_partition_collision_count"])},
    }
    return rows, maxima, tops


def month_groups(days):
    months = days.astype("datetime64[M]")
    return [np.where(months == m)[0] for m in np.unique(months)]


def permute_slot_stratified(masks, nums, groups, rng):
    pm = masks.copy()
    pn = nums.copy()
    for idx in groups:
        for slot in range(17):
            order = rng.permutation(idx)
            pm[idx, slot] = masks[order, slot]
            pn[idx, slot] = nums[order, slot]
    return pm, pn


def empirical_p(obs, null):
    return float((1 + sum(v >= obs for v in null)) / (1 + len(null)))


def main(out_path):
    cfg = json.loads(CFG_PATH.read_text())
    guard = cfg["guardrails"]
    if guard["validation_accessed"] is not False or guard["holdout_accessed"] is not False:
        raise SystemExit("guardrail breach")

    df = load_train()
    days, masks, nums = complete_days(df)
    states = state_table()
    bits = bitcount_table()
    lags = [int(x) for x in cfg["contest_lags"]]
    moduli = [int(x) for x in cfg["partition_rank_moduli"]]

    obs_rows, obs_max, obs_tops = evaluate(masks.reshape(-1), nums.reshape(-1), states, bits, lags, moduli)

    reps = int(cfg["permutation"]["reps"])
    rng = np.random.default_rng(int(cfg["permutation"]["seed"]))
    groups = month_groups(days)
    null = {k: [] for k in obs_max}

    for _ in range(reps):
        pm, pn = permute_slot_stratified(masks, nums, groups, rng)
        _, mx, _ = evaluate(pm.reshape(-1), pn.reshape(-1), states, bits, lags, moduli)
        for k, v in mx.items():
            null[k].append(v)

    pvals = {k: empirical_p(obs_max[k], null[k]) for k in obs_max}
    threshold = float(cfg["promotion_threshold"])
    min_p = min(pvals.values())
    status = "TRAIN_SERIAL_SIGNAL_REQUIRES_INDEPENDENT_REPLICATION" if min_p <= threshold else "NO_STRONG_TRAIN_SERIAL_STATE_FINGERPRINT"

    payload = {
        "schema": "wfl-serial-state-fingerprint-v2-result",
        "status": status,
        "config": cfg,
        "data": {
            "input_rows_2013_2018": int(len(df)),
            "complete_17_draw_days": int(len(days)),
            "flattened_contests": int(masks.size),
            "first_complete_day": str(days[0]),
            "last_complete_day": str(days[-1]),
            "validation_accessed": False,
            "holdout_accessed": False
        },
        "observed_max_stats": obs_max,
        "observed_top_locations": obs_tops,
        "familywise_empirical_p": pvals,
        "observed_by_contest_lag": obs_rows,
        "null_quantiles": {
            k: {
                "q90": float(np.quantile(v, .90)),
                "q95": float(np.quantile(v, .95)),
                "q99": float(np.quantile(v, .99))
            } for k, v in null.items()
        },
        "interpretation": "TRAIN-only cryptanalytic-style serial fingerprint diagnostic. A small familywise p-value would indicate output-state dependence worth preregistered replication, not RNG-state recovery and not predictive edge."
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)
