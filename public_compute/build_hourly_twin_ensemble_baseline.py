from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from hourly_twin import ROME, _load_candidate, _predict
from hourly_twin_ensemble import _consensus, _load_manifest


MANIFEST = Path("public_compute/prospective/hourly_ensemble_manifest.json")
CANDIDATE_ROOT = Path("public_compute/results/member_confirmation")
DATA_ROOT = Path("public_compute/data")
OUT = Path("public_compute/results/hourly_twin_ensemble_structural_baseline.json")


def _canonical_hash(payload: dict) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _target_from_row(row: pd.Series) -> datetime:
    d = str(row["date"])
    t = str(row["time"])
    hhmm = t[:5]
    return datetime.fromisoformat(f"{d}T{hhmm}:00").replace(tzinfo=ROME)


def main() -> None:
    manifest = _load_manifest(MANIFEST)
    member_rows = []
    for entry in manifest["candidates"]:
        idx = int(entry["candidate_index"])
        row = _load_candidate(idx, CANDIDATE_ROOT)
        if row["candidate_id"] != entry["candidate_id"]:
            raise SystemExit(f"candidate id mismatch: {idx}")
        member_rows.append((idx, row))

    parts = [pd.read_csv(p) for p in sorted(DATA_ROOT.glob("wfl_train_*.csv"))]
    if not parts:
        raise SystemExit("no TRAIN data")
    train = pd.concat(parts, ignore_index=True)
    train = train.sort_values(["date", "time", "contest"]).reset_index(drop=True)

    pair_mean = []
    pair_min = []
    pair_max = []
    exact_partition = []
    numerone_mode = []
    support_margin = []
    max_number_support = []
    tenth_support = []

    for _, record in train.iterrows():
        target = _target_from_row(record)
        members = []
        for idx, row in member_rows:
            pred, _ = _predict(row, target)
            members.append({"candidate_index": idx, "prediction": pred})
        c = _consensus(members)
        a = c["agreement"]
        b = c["consensus_boundary"]
        pair_mean.append(float(a["pairwise_symmetric_overlap_mean"]))
        pair_min.append(float(a["pairwise_symmetric_overlap_min"]))
        pair_max.append(float(a["pairwise_symmetric_overlap_max"]))
        exact_partition.append(int(a["exact_partition_member_count"]))
        numerone_mode.append(int(a["numerone_mode_support"]))
        support_margin.append(int(b["support_margin"]))
        tenth_support.append(int(b["tenth_number_support"]))
        max_number_support.append(max(int(v) for v in c["number_support"].values()))

    metrics = {
        "pairwise_symmetric_overlap_mean": pair_mean,
        "pairwise_symmetric_overlap_min": pair_min,
        "pairwise_symmetric_overlap_max": pair_max,
        "exact_partition_member_count": exact_partition,
        "numerone_mode_support": numerone_mode,
        "support_margin": support_margin,
        "tenth_number_support": tenth_support,
        "max_number_support": max_number_support,
    }

    summary = {}
    for name, values in metrics.items():
        summary[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "q50": _quantile(values, 0.50),
            "q90": _quantile(values, 0.90),
            "q95": _quantile(values, 0.95),
            "q99": _quantile(values, 0.99),
        }

    payload = {
        "schema": "wfl-hourly-twin-ensemble-structural-baseline-1",
        "evidence_grade": "TRAIN_OUTPUT_STRUCTURE_ONLY_NO_OFFICIAL_OUTCOME_USED",
        "manifest_sha256": _canonical_hash(manifest),
        "candidate_indices": [int(x) for x in manifest["candidate_indices"]],
        "train_period": {
            "start": str(train["date"].min()),
            "end": str(train["date"].max()),
            "contest_count": int(len(train)),
        },
        "metrics": summary,
        "guardrails": {
            "official_main_results_used_for_scoring": False,
            "official_numerone_used_for_scoring": False,
            "validation_accessed": False,
            "holdout_accessed": False,
            "prospective_scores_used_for_thresholds": False,
            "purpose": "pre-draw structural convergence calibration only",
            "structural_convergence_does_not_imply_predictive_edge": True,
        },
    }
    payload["baseline_sha256"] = _canonical_hash(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "contest_count": len(train),
        "baseline_sha256": payload["baseline_sha256"],
        "metrics": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
