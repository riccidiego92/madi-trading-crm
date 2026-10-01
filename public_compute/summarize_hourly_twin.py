from __future__ import annotations

from math import comb
import json
from pathlib import Path

SCORE_ROOT = Path("public_compute/prospective/hourly_scores")
OUT = Path("public_compute/prospective/hourly_evidence.json")


def classify_prize(direct_overlap: int, numerone_exact: bool) -> dict:
    k = int(direct_overlap)
    if not 0 <= k <= 10:
        raise ValueError("direct overlap must be 0..10")
    sym = max(k, 10 - k)
    num = bool(numerone_exact)

    code, rank, label = "NO_PRIZE", 0, "Nessun premio"
    if sym == 10 and num:
        code, rank, label = "WFL_RENDITA", 8, "Rendita / Win for Life"
    elif sym == 10:
        code, rank, label = "CAT_1", 7, "1a categoria"
    elif sym == 9 and num:
        code, rank, label = "CAT_2", 6, "2a categoria"
    elif sym == 9:
        code, rank, label = "CAT_3", 5, "3a categoria"
    elif sym == 8 and num:
        code, rank, label = "CAT_4", 4, "4a categoria"
    elif sym == 7 and num:
        code, rank, label = "CAT_5", 3, "5a categoria"
    elif sym == 8:
        code, rank, label = "CAT_6", 2, "6a categoria"
    elif sym == 7:
        code, rank, label = "CAT_7", 1, "7a categoria"

    return {
        "direct_overlap": k,
        "complement_overlap": 10 - k,
        "symmetric_overlap": sym,
        "numerone_exact": num,
        "prize_code": code,
        "prize_rank": rank,
        "prize_label": label,
        "winning_category": rank > 0,
    }


def random_reference() -> dict:
    total = comb(20, 10)

    def p_direct(k: int) -> float:
        return comb(10, k) * comb(10, 10 - k) / total

    sym = {
        s: (p_direct(s) + p_direct(10 - s) if s != 5 else p_direct(5))
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
        "symmetric_overlap_probability": {str(k): float(v) for k, v in sym.items()},
        "category_probability": cats,
        "any_prize_probability": float(sum(cats.values())),
        "numerone_probability": hit,
    }


def main() -> None:
    rows = []
    if SCORE_ROOT.exists():
        for p in sorted(SCORE_ROOT.glob("*/*.json")):
            row = json.loads(p.read_text())
            if row.get("schema") != "wfl-hourly-twin-development-score-1":
                raise SystemExit(f"unexpected score schema: {p}")
            rows.append((p, row))

    n = len(rows)
    sym = []
    direct = []
    exact = 0
    nums = 0
    joint = 0
    prize_rows = []
    category_counts = {}
    winning = 0

    for path, row in rows:
        score = row["score"]
        k = int(score["direct_main_overlap"])
        s = int(score["symmetric_main_overlap"])
        direct.append(k)
        sym.append(s)
        exact += int(bool(score["exact_main"] or score["exact_complement"]))
        nums += int(bool(score["numerone_exact"]))
        joint += int(bool(
            score["exact_main_plus_numerone"]
            or score["exact_complement_plus_numerone"]
        ))
        prize = classify_prize(k, bool(score["numerone_exact"]))
        category_counts[prize["prize_code"]] = (
            category_counts.get(prize["prize_code"], 0) + 1
        )
        winning += int(prize["winning_category"])
        prize_rows.append({
            "target": row["target"]["datetime"],
            "score_file": str(path),
            "freeze_sha256": row["freeze_sha256"],
            "candidate_id": row["candidate_id"],
            "prize": prize,
        })

    histogram = {str(k): 0 for k in range(5, 11)}
    for x in sym:
        histogram[str(x)] = histogram.get(str(x), 0) + 1

    ref = random_reference()
    payload = {
        "schema": "wfl-hourly-twin-development-evidence-2",
        "evidence_grade": "DEVELOPMENT_PROSPECTIVE_ONLY",
        "classification_policy": {
            "symmetric_ticket": True,
            "10_or_0_equal_weight": True,
            "10_or_0_plus_numerone_equal_weight": True,
            "9_or_1_equal_weight": True,
            "8_or_2_equal_weight": True,
            "7_or_3_equal_weight": True,
            "ranking_high_to_low": [
                "WFL_RENDITA", "CAT_1", "CAT_2", "CAT_3",
                "CAT_4", "CAT_5", "CAT_6", "CAT_7", "NO_PRIZE"
            ],
        },
        "scored_prediction_count": n,
        "observed": {
            "mean_direct_overlap": (sum(direct) / n) if n else None,
            "mean_symmetric_overlap": (sum(sym) / n) if n else None,
            "symmetric_overlap_histogram": histogram,
            "exact_main_or_complement_hits": int(exact),
            "numerone_hits": int(nums),
            "exact_pair_plus_numerone_hits": int(joint),
            "winning_prediction_count": int(winning),
            "winning_prediction_rate": (winning / n) if n else None,
            "category_counts": category_counts,
        },
        "random_reference": {
            **ref,
            "expected_winning_predictions": n * ref["any_prize_probability"],
            "expected_category_counts": {
                key: n * value
                for key, value in ref["category_probability"].items()
            },
        },
        "prize_classification_rows": prize_rows,
        "guardrails": {
            "no_edge_claim_from_small_n": True,
            "failures_retained": True,
            "all_scores_require_pre_target_freeze": True,
            "original_scores_are_immutable": True,
            "retroactive_reclassification_only": True,
            "post_result_retuning": False,
        },
        "status": (
            "NO_PROSPECTIVE_SCORES_YET"
            if n == 0
            else "PROSPECTIVE_PRIZE_EVIDENCE_ACCUMULATING_NO_EDGE_CLAIM"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
