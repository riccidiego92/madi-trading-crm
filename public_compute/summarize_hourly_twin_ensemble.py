from __future__ import annotations

from math import comb
import json
from pathlib import Path


SCORE_ROOT = Path("public_compute/prospective/hourly_ensemble_scores")
OUT = Path("public_compute/prospective/hourly_ensemble_evidence.json")


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
        for path in sorted(SCORE_ROOT.glob("*/*.json")):
            row = json.loads(path.read_text())
            if row.get("schema") != "wfl-hourly-twin-ensemble-development-score-1":
                raise SystemExit(f"unexpected ensemble score schema: {path}")
            rows.append((path, row))

    n = len(rows)
    direct = []
    sym = []
    numerone_hits = 0
    exact = 0
    exact_joint = 0
    wins = 0
    categories = {}
    agreement = []
    members = {}

    for path, row in rows:
        score = row["consensus_score"]
        direct.append(int(score["direct_main_overlap"]))
        sym.append(int(score["symmetric_main_overlap"]))
        numerone_hits += int(bool(score["numerone_exact"]))
        exact += int(bool(score["exact_main"] or score["exact_complement"]))
        exact_joint += int(bool(
            score["exact_main_plus_numerone"]
            or score["exact_complement_plus_numerone"]
        ))
        prize = score["prize_category"]
        code = prize["prize_code"]
        categories[code] = categories.get(code, 0) + 1
        wins += int(bool(prize["winning_category"]))
        agreement.append(float(
            row["consensus_agreement"]["pairwise_symmetric_overlap_mean"]
        ))

        for member in row["member_scores"]:
            idx = str(int(member["candidate_index"]))
            rec = members.setdefault(idx, {
                "candidate_id": member["candidate_id"],
                "candidate_identity": member["candidate_identity"],
                "scored": 0,
                "wins": 0,
                "numerone_hits": 0,
                "exact_main_or_complement_hits": 0,
                "symmetric_overlap_sum": 0,
            })
            ms = member["score"]
            rec["scored"] += 1
            rec["wins"] += int(bool(ms["prize_category"]["winning_category"]))
            rec["numerone_hits"] += int(bool(ms["numerone_exact"]))
            rec["exact_main_or_complement_hits"] += int(bool(
                ms["exact_main"] or ms["exact_complement"]
            ))
            rec["symmetric_overlap_sum"] += int(ms["symmetric_main_overlap"])

    for rec in members.values():
        d = rec["scored"]
        rec["winning_rate"] = rec["wins"] / d if d else None
        rec["mean_symmetric_overlap"] = rec["symmetric_overlap_sum"] / d if d else None
        del rec["symmetric_overlap_sum"]

    histogram = {str(k): 0 for k in range(5, 11)}
    for x in sym:
        histogram[str(x)] += 1

    ref = random_reference()
    payload = {
        "schema": "wfl-hourly-twin-ensemble-development-evidence-1",
        "evidence_grade": "DEVELOPMENT_ENSEMBLE_PROSPECTIVE_ONLY",
        "scored_prediction_count": n,
        "observed": {
            "mean_direct_overlap": sum(direct) / n if n else None,
            "mean_symmetric_overlap": sum(sym) / n if n else None,
            "symmetric_overlap_histogram": histogram,
            "numerone_hits": int(numerone_hits),
            "exact_main_or_complement_hits": int(exact),
            "exact_pair_plus_numerone_hits": int(exact_joint),
            "winning_prediction_count": int(wins),
            "winning_prediction_rate": wins / n if n else None,
            "category_counts": categories,
            "mean_member_pairwise_symmetric_overlap": (
                sum(agreement) / n if n else None
            ),
        },
        "member_descriptive_scores": members,
        "random_reference": {
            **ref,
            "expected_winning_predictions": n * ref["any_prize_probability"],
            "expected_numerone_hits": n * ref["numerone_probability"],
            "expected_category_counts": {
                key: n * value
                for key, value in ref["category_probability"].items()
            },
        },
        "guardrails": {
            "all_scores_require_pre_target_freeze": True,
            "member_set_frozen_before_first_ensemble_output": True,
            "member_descriptive_scores_not_used_for_reselection": True,
            "post_result_retuning": False,
            "failures_retained": True,
            "p48_validated": False,
            "no_edge_claim_from_small_n": True,
        },
        "status": (
            "NO_PROSPECTIVE_ENSEMBLE_SCORES_YET"
            if n == 0
            else "PROSPECTIVE_ENSEMBLE_EVIDENCE_ACCUMULATING_NO_EDGE_CLAIM"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
