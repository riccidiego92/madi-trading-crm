from __future__ import annotations

import argparse
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path

from hourly_twin import (
    ROME,
    _load_candidate,
    _predict,
    next_target,
    public_ordinal_for_datetime,
)

DEFAULT_MANIFEST = Path("public_compute/prospective/hourly_ensemble_manifest.json")
DEFAULT_CANDIDATE_ROOT = Path("public_compute/results/member_confirmation")
DEFAULT_OUT_ROOT = Path("public_compute/prospective/hourly_ensemble")
HOURS = tuple(range(7, 24))


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _partition_key(main: list[int], complement: list[int]) -> str:
    a = tuple(sorted(int(x) for x in main))
    b = tuple(sorted(int(x) for x in complement))
    lo, hi = (a, b) if a <= b else (b, a)
    return sha256(json.dumps([lo, hi], separators=(",", ":")).encode("utf-8")).hexdigest()


def _orient(main: list[int], complement: list[int], anchor: set[int]) -> list[int]:
    a = tuple(sorted(int(x) for x in main))
    b = tuple(sorted(int(x) for x in complement))
    ka = len(set(a) & anchor)
    kb = len(set(b) & anchor)
    if ka > kb:
        return list(a)
    if kb > ka:
        return list(b)
    return list(min(a, b))


def _consensus(member_predictions: list[dict]) -> dict:
    if not member_predictions:
        raise ValueError("ensemble has no member predictions")

    anchor = set(int(x) for x in member_predictions[0]["prediction"]["main"])
    votes = {n: 0 for n in range(1, 21)}
    numerone_votes = {n: 0 for n in range(1, 21)}
    oriented = []

    for row in member_predictions:
        pred = row["prediction"]
        side = _orient(pred["main"], pred["complement"], anchor)
        oriented.append(side)
        for n in side:
            votes[int(n)] += 1
        numerone_votes[int(pred["numerone"])] += 1

    ranked = sorted(votes, key=lambda n: (-votes[n], n))
    main = sorted(ranked[:10])
    complement = sorted(set(range(1, 21)) - set(main))
    num_ranked = sorted(numerone_votes, key=lambda n: (-numerone_votes[n], n))
    numerone = int(num_ranked[0])

    pairwise = []
    for i in range(len(member_predictions)):
        a = set(int(x) for x in member_predictions[i]["prediction"]["main"])
        for j in range(i + 1, len(member_predictions)):
            b = set(int(x) for x in member_predictions[j]["prediction"]["main"])
            k = len(a & b)
            pairwise.append(max(k, 10 - k))

    exact_partition_members = 0
    cm = set(main)
    cc = set(complement)
    for row in member_predictions:
        ms = set(int(x) for x in row["prediction"]["main"])
        if ms == cm or ms == cc:
            exact_partition_members += 1

    tenth_vote = votes[ranked[9]]
    eleventh_vote = votes[ranked[10]]
    prediction = {
        "main": main,
        "complement": complement,
        "numerone": numerone,
    }
    prediction["prediction_sha256"] = _canonical_hash(prediction)

    return {
        "prediction": prediction,
        "alignment_anchor_candidate_index": int(member_predictions[0]["candidate_index"]),
        "oriented_member_sides": oriented,
        "number_support": {str(n): int(votes[n]) for n in range(1, 21)},
        "numerone_support": {str(n): int(numerone_votes[n]) for n in range(1, 21)},
        "consensus_boundary": {
            "tenth_number_support": int(tenth_vote),
            "eleventh_number_support": int(eleventh_vote),
            "support_margin": int(tenth_vote - eleventh_vote),
        },
        "agreement": {
            "member_count": len(member_predictions),
            "pairwise_symmetric_overlap_mean": (
                sum(pairwise) / len(pairwise) if pairwise else None
            ),
            "pairwise_symmetric_overlap_min": min(pairwise) if pairwise else None,
            "pairwise_symmetric_overlap_max": max(pairwise) if pairwise else None,
            "exact_partition_member_count": int(exact_partition_members),
            "numerone_mode_support": int(max(numerone_votes.values())),
        },
    }


def _load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text())
    if m.get("schema") != "wfl-hourly-twin-ensemble-manifest-1":
        raise ValueError("unexpected ensemble manifest schema")
    if m.get("status") != "DEVELOPMENT_ENSEMBLE_FROZEN":
        raise ValueError("ensemble manifest is not frozen")
    g = m.get("guardrails", {})
    if g.get("validation_accessed") is not False or g.get("holdout_accessed") is not False:
        raise ValueError("ensemble manifest guardrail breach")
    if g.get("prospective_result_feedback_into_selection") is not False:
        raise ValueError("prospective feedback may not select ensemble members")
    return m


def freeze_ensemble(
    *,
    now: datetime,
    target: datetime | None,
    manifest_path: Path,
    candidate_root: Path,
    out_root: Path,
    min_lead_seconds: int,
) -> dict:
    local_now = now.astimezone(ROME)
    target = (target or next_target(local_now, min_lead_seconds)).astimezone(ROME)
    if target <= local_now:
        raise ValueError("target must be in the future")
    lead = int((target - local_now).total_seconds())
    if lead < int(min_lead_seconds):
        raise ValueError("insufficient prospective lead time")

    manifest = _load_manifest(manifest_path)
    manifest_hash = _canonical_hash(manifest)
    members = []

    for entry in manifest["candidates"]:
        idx = int(entry["candidate_index"])
        row = _load_candidate(idx, candidate_root)
        if row["candidate_id"] != entry["candidate_id"]:
            raise ValueError(f"candidate id mismatch: {idx}")
        if row["candidate_identity"] != entry["candidate_identity"]:
            raise ValueError(f"candidate identity mismatch: {idx}")
        pred, machine_ordinal = _predict(row, target)
        members.append({
            "candidate_index": idx,
            "candidate_id": row["candidate_id"],
            "candidate_identity": row["candidate_identity"],
            "candidate_signature_sha256": row["candidate_signature_sha256"],
            "machine_ordinal": int(machine_ordinal),
            "member_confirmation": {
                "simulation_reps": int(row["simulation_reps"]),
                "p_max_stat": float(row["test"]["p_max_stat"]),
                "rejected_at_triage_alpha_0_01": bool(
                    row["test"]["rejected_at_triage_alpha"]
                ),
            },
            "prediction": pred,
            "partition_sha256": _partition_key(
                pred["main"], pred["complement"]
            ),
        })

    consensus = _consensus(members)
    dest = (
        out_root
        / target.date().isoformat()
        / f"{target.hour:02d}00_ensemble_v1.json"
    )
    if dest.exists():
        existing = json.loads(dest.read_text())
        return {
            "status": "EXISTS_IMMUTABLE",
            "path": str(dest),
            "freeze_sha256": existing["freeze_sha256"],
            "target": existing["target"],
            "consensus_prediction": existing["consensus"]["prediction"],
        }

    payload = {
        "schema": "wfl-hourly-twin-ensemble-development-freeze-1",
        "evidence_grade": "DEVELOPMENT_ENSEMBLE_UNVALIDATED_NOT_FOR_LIVE_INFERENCE",
        "frozen_at": local_now.isoformat(),
        "lead_seconds": lead,
        "target": {
            "datetime": target.isoformat(),
            "timezone": "Europe/Rome",
            "public_ordinal": public_ordinal_for_datetime(target),
        },
        "ensemble": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "selection_policy": manifest["selection_policy"],
            "candidate_indices": [int(x) for x in manifest["candidate_indices"]],
            "member_count": len(members),
        },
        "members": members,
        "consensus": consensus,
        "guardrails": {
            "generated_before_target": True,
            "future_result_used_for_generation": False,
            "selection_frozen_before_output": True,
            "prospective_score_used_for_member_selection": False,
            "operational_seed_or_state_search": False,
            "post_result_retuning_allowed": False,
            "counts_as_p48_validated_prediction": False,
            "failures_retained": True,
            "immutable_once_written": True,
            "development_only": True,
        },
    }
    payload["freeze_sha256"] = _canonical_hash(payload)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "status": "FROZEN",
        "path": str(dest),
        "freeze_sha256": payload["freeze_sha256"],
        "target": payload["target"],
        "consensus_prediction": consensus["prediction"],
        "agreement": consensus["agreement"],
    }


def prefreeze_day(
    *,
    target_day: date,
    now: datetime,
    manifest_path: Path,
    candidate_root: Path,
    out_root: Path,
    min_lead_seconds: int,
) -> dict:
    results = []
    for hour in HOURS:
        target = datetime(
            target_day.year,
            target_day.month,
            target_day.day,
            hour,
            0,
            tzinfo=ROME,
        )
        if target <= now:
            continue
        if (target - now).total_seconds() < min_lead_seconds:
            continue
        results.append(
            freeze_ensemble(
                now=now,
                target=target,
                manifest_path=manifest_path,
                candidate_root=candidate_root,
                out_root=out_root,
                min_lead_seconds=min_lead_seconds,
            )
        )
    return {
        "schema": "wfl-hourly-twin-ensemble-prefreeze-day-run-1",
        "generated_at": now.astimezone(ROME).isoformat(),
        "target_date": target_day.isoformat(),
        "future_targets_processed": len(results),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze", "prefreeze-day"])
    ap.add_argument("--target", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--min-lead-seconds", type=int, default=300)
    args = ap.parse_args()

    now = datetime.now(ROME)
    if args.command == "freeze":
        target = datetime.fromisoformat(args.target) if args.target else None
        if target is not None and (target.tzinfo is None or target.utcoffset() is None):
            raise SystemExit("--target must be offset-aware")
        result = freeze_ensemble(
            now=now,
            target=target,
            manifest_path=Path(args.manifest),
            candidate_root=Path(args.candidate_root),
            out_root=Path(args.out_root),
            min_lead_seconds=args.min_lead_seconds,
        )
    else:
        target_day = date.fromisoformat(args.date) if args.date else now.date()
        result = prefreeze_day(
            target_day=target_day,
            now=now,
            manifest_path=Path(args.manifest),
            candidate_root=Path(args.candidate_root),
            out_root=Path(args.out_root),
            min_lead_seconds=args.min_lead_seconds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
