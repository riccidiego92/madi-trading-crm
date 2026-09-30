from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from wfl_public.combinatorics import complement_numbers
from wfl_public.p26_public_drbg import draw_main as draw_drbg_main, draw_numerone as draw_drbg_numerone
from wfl_public.p42_fast import simulate_phased_fast
from wfl_public.p42_phased_draw_machine import PhasedMachineConfig, _background, _pair
from wfl_public.p46_calendar_alignment import aligned_machine_ordinal, calendar_alignment_offset

ROME = ZoneInfo("Europe/Rome")
LAUNCH_DATE = datetime(2013, 1, 21, tzinfo=ROME).date()
LAUNCH_HOURS = tuple(range(12, 24))
NORMAL_HOURS = tuple(range(7, 24))
DEFAULT_CANDIDATE_INDEX = 16


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def public_ordinal_for_datetime(when: datetime) -> int:
    local = when.astimezone(ROME)
    if any((local.minute, local.second, local.microsecond)):
        raise ValueError("target contest must be exactly on the hour")
    if local.date() < LAUNCH_DATE:
        raise ValueError("target predates Classico launch")
    if local.date() == LAUNCH_DATE:
        if local.hour not in LAUNCH_HOURS:
            raise ValueError("launch day contests run 12:00..23:00")
        return local.hour - 12
    if local.hour not in NORMAL_HOURS:
        raise ValueError("normal Classico contest hour must be 07:00..23:00")
    day_delta = (local.date() - LAUNCH_DATE).days
    return 12 + (day_delta - 1) * 17 + (local.hour - 7)


def next_target(now: datetime, min_lead_seconds: int = 300) -> datetime:
    local = now.astimezone(ROME)
    floor = local.replace(minute=0, second=0, microsecond=0)
    candidates = []
    for day_add in (0, 1, 2):
        day = (local + timedelta(days=day_add)).date()
        hours = LAUNCH_HOURS if day == LAUNCH_DATE else NORMAL_HOURS
        for hour in hours:
            dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=ROME)
            if dt > local and (dt - local).total_seconds() >= int(min_lead_seconds):
                candidates.append(dt)
    if not candidates:
        raise RuntimeError("could not resolve a future Classico target")
    return min(candidates)


def _load_candidate(index: int, root: Path) -> dict:
    p = root / f"candidate_{int(index)}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    row = json.loads(p.read_text())
    if row.get("schema") != "wfl-p43-p42-train-candidate-evaluation-1":
        raise ValueError("unexpected candidate result schema")
    if row.get("stage") != "train" or int(row.get("simulation_reps", 0)) != 2000:
        raise ValueError("candidate is not a frozen P43 member-confirmation result")
    g = row.get("guardrails", {})
    if g.get("validation_accessed") is not False or g.get("holdout_accessed") is not False:
        raise ValueError("candidate guardrail breach")
    if bool(row["test"]["rejected_at_triage_alpha"]):
        raise ValueError("fixed development candidate was rejected by P43")
    return row


def _predict(row: dict, target: datetime) -> tuple[dict, int]:
    cfg = PhasedMachineConfig(**dict(row["config"]))
    public_ordinal = public_ordinal_for_datetime(target)
    machine_ordinal = aligned_machine_ordinal(public_ordinal, cfg, research_phase_offset=0)

    # Fast direct route for per-contest lifecycle; exact same frozen machinery.
    if cfg.reseed_mode == "per_contest":
        main_rng, num_rng = _pair(cfg, f"contest:{machine_ordinal}")
        _background(main_rng, cfg.background_cycles_main)
        main = tuple(sorted(int(x) for x in draw_drbg_main(main_rng, cfg.mapper)))
        _background(num_rng, cfg.background_cycles_numerone)
        numerone = int(draw_drbg_numerone(num_rng))
    else:
        frame = simulate_phased_fast(machine_ordinal + 1, cfg)
        last = frame.iloc[-1]
        main = tuple(sorted(int(last[f"n{i}"]) for i in range(1, 11)))
        numerone = int(last["numerone"])

    pred = {
        "main": list(main),
        "complement": list(complement_numbers(main)),
        "numerone": numerone,
    }
    pred["prediction_sha256"] = _canonical_hash(pred)
    return pred, machine_ordinal


def freeze(
    *,
    candidate_index: int,
    now: datetime,
    target: datetime | None,
    result_root: Path,
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

    row = _load_candidate(candidate_index, result_root)
    pred, machine_ordinal = _predict(row, target)
    cfg = PhasedMachineConfig(**dict(row["config"]))

    dest = (
        out_root
        / target.date().isoformat()
        / f"{target.hour:02d}00_candidate_{int(candidate_index)}.json"
    )
    if dest.exists():
        existing = json.loads(dest.read_text())
        return {
            "status": "EXISTS_IMMUTABLE",
            "path": str(dest),
            "freeze_sha256": existing["freeze_sha256"],
            "target": existing["target"],
            "prediction": existing["prediction"],
        }

    payload = {
        "schema": "wfl-hourly-twin-development-freeze-1",
        "evidence_grade": "DEVELOPMENT_UNVALIDATED_NOT_FOR_LIVE_INFERENCE",
        "frozen_at": local_now.isoformat(),
        "lead_seconds": lead,
        "target": {
            "datetime": target.isoformat(),
            "timezone": "Europe/Rome",
            "public_ordinal": public_ordinal_for_datetime(target),
        },
        "selection": {
            "rule": "FIXED_DEVELOPMENT_CANDIDATE_INDEX",
            "candidate_index": int(candidate_index),
            "selection_frozen_before_output": True,
            "generated_numbers_used_for_selection": False,
        },
        "machine": {
            "candidate_id": row["candidate_id"],
            "candidate_identity": row["candidate_identity"],
            "candidate_signature_sha256": row["candidate_signature_sha256"],
            "config": row["config"],
            "calendar_alignment_offset": int(calendar_alignment_offset(cfg)),
            "research_phase_offset": 0,
            "machine_ordinal": int(machine_ordinal),
            "member_confirmation": {
                "simulation_reps": int(row["simulation_reps"]),
                "p_max_stat": float(row["test"]["p_max_stat"]),
                "rejected_at_triage_alpha_0_01": bool(
                    row["test"]["rejected_at_triage_alpha"]
                ),
                "final_train_complete": False,
            },
        },
        "prediction": pred,
        "guardrails": {
            "generated_before_target": True,
            "future_result_used_for_generation": False,
            "operational_seed_or_state_search": False,
            "post_result_retuning_allowed": False,
            "counts_as_validated_prediction": False,
            "failures_retained": True,
            "immutable_once_written": True,
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
        "prediction": payload["prediction"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze"])
    ap.add_argument("--candidate-index", type=int, default=DEFAULT_CANDIDATE_INDEX)
    ap.add_argument(
        "--candidate-root",
        default="public_compute/results/member_confirmation",
    )
    ap.add_argument(
        "--out-root",
        default="public_compute/prospective/hourly",
    )
    ap.add_argument("--min-lead-seconds", type=int, default=300)
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    now = datetime.now(ROME)
    target = datetime.fromisoformat(args.target) if args.target else None
    if target is not None and (target.tzinfo is None or target.utcoffset() is None):
        raise SystemExit("--target must be offset-aware")

    result = freeze(
        candidate_index=args.candidate_index,
        now=now,
        target=target,
        result_root=Path(args.candidate_root),
        out_root=Path(args.out_root),
        min_lead_seconds=args.min_lead_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
