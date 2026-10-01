from __future__ import annotations

from math import comb
import json
from pathlib import Path

import pandas as pd

from wfl_public.p42_fast import simulate_phased_fast
from wfl_public.p42_phased_draw_machine import PhasedMachineConfig
from wfl_public.p46_calendar_alignment import calendar_alignment_offset

DATA_ROOT = Path("public_compute/data")
CANDIDATE = Path("public_compute/results/member_confirmation/candidate_16.json")
OUT = Path("public_compute/results/hourly_twin_train_prize_backtest_candidate_16.json")
MAIN_COLS = [f"n{i}" for i in range(1, 11)]


def classify(k: int, num_hit: bool) -> tuple[str, int]:
    sym = max(int(k), 10 - int(k))
    if sym == 10 and num_hit:
        return "WFL_RENDITA", 8
    if sym == 10:
        return "CAT_1", 7
    if sym == 9 and num_hit:
        return "CAT_2", 6
    if sym == 9:
        return "CAT_3", 5
    if sym == 8 and num_hit:
        return "CAT_4", 4
    if sym == 7 and num_hit:
        return "CAT_5", 3
    if sym == 8:
        return "CAT_6", 2
    if sym == 7:
        return "CAT_7", 1
    return "NO_PRIZE", 0


def random_reference() -> dict:
    total = comb(20, 10)

    def p(k: int) -> float:
        return comb(10, k) * comb(10, 10-k) / total

    sym = {
        s: (p(s) + p(10-s) if s != 5 else p(5))
        for s in range(5, 11)
    }
    hit = 1 / 20
    miss = 19 / 20
    cats = {
        "WFL_RENDITA": sym[10] * hit,
        "CAT_1": sym[10] * miss,
        "CAT_2": sym[9] * hit,
        "CAT_3": sym[9] * miss,
        "CAT_4": sym[8] * hit,
        "CAT_5": sym[7] * hit,
        "CAT_6": sym[8] * miss,
        "CAT_7": sym[7] * miss,
    }
    return {
        "category_probability": cats,
        "any_prize_probability": sum(cats.values()),
        "symmetric_overlap_probability": {str(k): v for k, v in sym.items()},
    }


def main() -> None:
    parts = [pd.read_csv(p) for p in sorted(DATA_ROOT.glob("wfl_train_*.csv"))]
    if not parts:
        raise SystemExit("no TRAIN data")
    real = pd.concat(parts, ignore_index=True)
    real = real.sort_values(["date", "time", "contest"]).reset_index(drop=True)

    candidate = json.loads(CANDIDATE.read_text())
    cfg = PhasedMachineConfig(**candidate["config"])
    offset = calendar_alignment_offset(cfg)
    synthetic = simulate_phased_fast(len(real) + offset, cfg)
    if offset:
        synthetic = synthetic.iloc[offset:].reset_index(drop=True)
    synthetic = synthetic.iloc[:len(real)].reset_index(drop=True)
    if len(synthetic) != len(real):
        raise SystemExit("synthetic/real row mismatch")

    category_counts = {}
    sym_counts = {str(k): 0 for k in range(5, 11)}
    numerone_hits = 0
    wins = 0
    rank_sum = 0
    max_rank = 0

    yearly = {}
    for i in range(len(real)):
        actual = {int(real.iloc[i][c]) for c in MAIN_COLS}
        predicted = {int(synthetic.iloc[i][c]) for c in MAIN_COLS}
        k = len(actual & predicted)
        sym = max(k, 10-k)
        num_hit = int(real.iloc[i]["numerone"]) == int(synthetic.iloc[i]["numerone"])
        code, rank = classify(k, num_hit)

        category_counts[code] = category_counts.get(code, 0) + 1
        sym_counts[str(sym)] += 1
        numerone_hits += int(num_hit)
        wins += int(rank > 0)
        rank_sum += rank
        max_rank = max(max_rank, rank)

        year = str(real.iloc[i]["date"])[:4]
        rec = yearly.setdefault(year, {
            "contests": 0,
            "wins": 0,
            "category_counts": {},
            "numerone_hits": 0,
        })
        rec["contests"] += 1
        rec["wins"] += int(rank > 0)
        rec["numerone_hits"] += int(num_hit)
        rec["category_counts"][code] = rec["category_counts"].get(code, 0) + 1

    n = len(real)
    ref = random_reference()
    for year, rec in yearly.items():
        rec["win_rate"] = rec["wins"] / rec["contests"]

    payload = {
        "schema": "wfl-hourly-twin-train-prize-backtest-1",
        "evidence_grade": "RETRODICTIVE_TRAIN_IN_SAMPLE_NOT_PROSPECTIVE",
        "candidate_index": 16,
        "candidate_id": candidate["candidate_id"],
        "candidate_identity": candidate["candidate_identity"],
        "candidate_config": candidate["config"],
        "train_period": {
            "start": str(real["date"].min()),
            "end": str(real["date"].max()),
            "contests": n,
        },
        "classification_policy": {
            "10_or_0_plus_numerone_equal_weight": True,
            "10_or_0_equal_weight": True,
            "9_or_1_equal_weight": True,
            "8_or_2_equal_weight": True,
            "7_or_3_equal_weight": True,
        },
        "observed": {
            "winning_contests": wins,
            "winning_rate": wins / n,
            "category_counts": category_counts,
            "symmetric_overlap_counts": sym_counts,
            "numerone_hits": numerone_hits,
            "numerone_hit_rate": numerone_hits / n,
            "mean_prize_rank": rank_sum / n,
            "max_prize_rank": max_rank,
            "yearly": yearly,
        },
        "random_reference": {
            **ref,
            "expected_winning_contests": n * ref["any_prize_probability"],
            "expected_category_counts": {
                code: n * prob
                for code, prob in ref["category_probability"].items()
            },
            "expected_numerone_hits": n / 20,
        },
        "guardrails": {
            "train_only": True,
            "validation_accessed": False,
            "holdout_accessed": False,
            "historical_results_used_for_scoring": True,
            "historical_results_not_used_to_change_candidate": True,
            "retrodiction_is_not_prediction": True,
            "no_edge_claim_from_backtest_alone": True,
        },
        "status": "RETRODICTIVE_BACKTEST_ONLY",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "contests": n,
        "wins": wins,
        "win_rate": wins/n,
        "random_win_rate": ref["any_prize_probability"],
        "category_counts": category_counts,
        "max_prize_rank": max_rank,
        "out": str(OUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
