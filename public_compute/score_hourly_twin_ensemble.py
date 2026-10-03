from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from hourly_twin import ROME
from score_hourly_twin import classify_prize, fetch_official


FREEZE_ROOT = Path("public_compute/prospective/hourly_ensemble")
SCORE_ROOT = Path("public_compute/prospective/hourly_ensemble_scores")


def _canonical_hash(payload: dict) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_freeze(freeze: dict) -> datetime:
    if freeze.get("schema") != "wfl-hourly-twin-ensemble-development-freeze-1":
        raise ValueError("unexpected ensemble freeze schema")
    saved = freeze.get("freeze_sha256")
    unsigned = dict(freeze)
    unsigned.pop("freeze_sha256", None)
    if _canonical_hash(unsigned) != saved:
        raise ValueError("ensemble freeze hash mismatch")
    target = datetime.fromisoformat(freeze["target"]["datetime"]).astimezone(ROME)
    frozen = datetime.fromisoformat(freeze["frozen_at"]).astimezone(ROME)
    if frozen >= target:
        raise ValueError("ensemble freeze does not precede target")
    if freeze["guardrails"].get("selection_frozen_before_output") is not True:
        raise ValueError("ensemble selection was not frozen before output")
    if freeze["guardrails"].get("future_result_used_for_generation") is not False:
        raise ValueError("future-result guardrail breach")
    return target


def _score_prediction(prediction: dict, official: dict) -> dict:
    pred = tuple(sorted(int(x) for x in prediction["main"]))
    comp = tuple(sorted(int(x) for x in prediction["complement"]))
    actual = tuple(sorted(int(x) for x in official["main"]))
    k = len(set(pred) & set(actual))
    num_hit = int(prediction["numerone"]) == int(official["numerone"])
    return {
        "direct_main_overlap": int(k),
        "symmetric_main_overlap": int(max(k, 10 - k)),
        "exact_main": bool(pred == actual),
        "exact_complement": bool(comp == actual),
        "numerone_exact": bool(num_hit),
        "exact_main_plus_numerone": bool(pred == actual and num_hit),
        "exact_complement_plus_numerone": bool(comp == actual and num_hit),
        "prize_category": classify_prize(int(k), bool(num_hit)),
    }


def score_one(path: Path, score_root: Path, now: datetime) -> dict | None:
    freeze = json.loads(path.read_text())
    target = verify_freeze(freeze)
    if now.astimezone(ROME) <= target:
        return None

    rel = path.relative_to(FREEZE_ROOT)
    dest = score_root / rel
    if dest.exists():
        return {"status": "EXISTS_IMMUTABLE", "path": str(dest)}

    official = fetch_official(target)
    if official is None:
        return None

    consensus_score = _score_prediction(
        freeze["consensus"]["prediction"], official
    )
    member_scores = []
    for member in freeze["members"]:
        score = _score_prediction(member["prediction"], official)
        member_scores.append({
            "candidate_index": int(member["candidate_index"]),
            "candidate_id": member["candidate_id"],
            "candidate_identity": member["candidate_identity"],
            "partition_sha256": member["partition_sha256"],
            "score": score,
        })

    payload = {
        "schema": "wfl-hourly-twin-ensemble-development-score-1",
        "evidence_grade": "DEVELOPMENT_ENSEMBLE_PROSPECTIVE_SCORE_NO_EDGE_CLAIM",
        "freeze_sha256": freeze["freeze_sha256"],
        "scored_at": now.astimezone(ROME).isoformat(),
        "target": freeze["target"],
        "ensemble": freeze["ensemble"],
        "consensus_prediction": freeze["consensus"]["prediction"],
        "consensus_agreement": freeze["consensus"]["agreement"],
        "consensus_score": consensus_score,
        "member_scores": member_scores,
        "official": official,
        "guardrails": {
            "freeze_hash_verified": True,
            "freeze_precedes_target": True,
            "official_source_retained": True,
            "post_result_retuning": False,
            "member_selection_changed_after_result": False,
            "failures_retained": True,
            "development_only": True,
            "no_edge_claim": True,
        },
    }
    payload["score_sha256"] = _canonical_hash(payload)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "status": "SCORED",
        "path": str(dest),
        "consensus_score": consensus_score,
    }


def main() -> None:
    now = datetime.now(ROME)
    results = []
    if FREEZE_ROOT.exists():
        for path in sorted(FREEZE_ROOT.glob("*/*.json")):
            result = score_one(path, SCORE_ROOT, now)
            if result is not None:
                results.append(result)
    print(json.dumps({"processed": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
