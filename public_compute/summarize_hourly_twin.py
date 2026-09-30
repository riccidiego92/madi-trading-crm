from __future__ import annotations

import json
from pathlib import Path

SCORE_ROOT = Path("public_compute/prospective/hourly_scores")
OUT = Path("public_compute/prospective/hourly_evidence.json")


def main() -> None:
    rows = []
    if SCORE_ROOT.exists():
        for p in sorted(SCORE_ROOT.glob("*/*.json")):
            row = json.loads(p.read_text())
            if row.get("schema") != "wfl-hourly-twin-development-score-1":
                raise SystemExit(f"unexpected score schema: {p}")
            rows.append(row)

    n = len(rows)
    sym = [int(r["score"]["symmetric_main_overlap"]) for r in rows]
    direct = [int(r["score"]["direct_main_overlap"]) for r in rows]
    exact = sum(bool(r["score"]["exact_main"] or r["score"]["exact_complement"]) for r in rows)
    nums = sum(bool(r["score"]["numerone_exact"]) for r in rows)
    joint = sum(
        bool(
            r["score"]["exact_main_plus_numerone"]
            or r["score"]["exact_complement_plus_numerone"]
        )
        for r in rows
    )

    histogram = {str(k): 0 for k in range(5, 11)}
    for x in sym:
        histogram[str(x)] = histogram.get(str(x), 0) + 1

    payload = {
        "schema": "wfl-hourly-twin-development-evidence-1",
        "evidence_grade": "DEVELOPMENT_PROSPECTIVE_ONLY",
        "scored_prediction_count": n,
        "observed": {
            "mean_direct_overlap": (sum(direct) / n) if n else None,
            "mean_symmetric_overlap": (sum(sym) / n) if n else None,
            "symmetric_overlap_histogram": histogram,
            "exact_main_or_complement_hits": int(exact),
            "numerone_hits": int(nums),
            "exact_pair_plus_numerone_hits": int(joint),
        },
        "random_reference": {
            "exact_main_or_complement_probability_per_prediction": 1 / 92378,
            "numerone_probability_per_prediction": 1 / 20,
            "exact_pair_plus_numerone_probability_per_prediction": 1 / 1847560,
            "expected_exact_main_or_complement_hits": n / 92378,
            "expected_numerone_hits": n / 20,
            "expected_exact_pair_plus_numerone_hits": n / 1847560,
        },
        "guardrails": {
            "no_edge_claim_from_small_n": True,
            "failures_retained": True,
            "all_scores_require_pre_target_freeze": True,
            "post_result_retuning": False,
        },
        "status": (
            "NO_PROSPECTIVE_SCORES_YET"
            if n == 0
            else "PROSPECTIVE_EVIDENCE_ACCUMULATING_NO_EDGE_CLAIM"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
