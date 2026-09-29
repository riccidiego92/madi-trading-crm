from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from wfl_public.p43_p42_train_screen import classify_family, registry_rows

ROOT = Path("public_compute/results/member_confirmation")
OUT = Path("public_compute/results/final_train_plan.json")


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    files = sorted(ROOT.glob("candidate_*.json"))
    by_index: dict[int, dict] = {}
    file_hashes: dict[int, str] = {}

    for path in files:
        m = re.fullmatch(r"candidate_(\d+)\.json", path.name)
        if not m:
            continue
        idx = int(m.group(1))
        raw = path.read_bytes()
        row = json.loads(raw)
        if row.get("schema") != "wfl-p43-p42-train-candidate-evaluation-1":
            raise SystemExit(f"bad schema: {path}")
        if row.get("stage") != "train":
            raise SystemExit(f"non-TRAIN result: {path}")
        if int(row.get("simulation_reps", 0)) != 2000:
            raise SystemExit(f"wrong member-confirmation reps: {path}")
        g = row.get("guardrails", {})
        if g.get("validation_accessed") is not False or g.get("holdout_accessed") is not False:
            raise SystemExit(f"guardrail breach: {path}")
        p = float(row["test"]["p_max_stat"])
        if row["test"].get("rejected_at_triage_alpha") is not (p <= 0.01):
            raise SystemExit(f"triage decision mismatch: {path}")
        if idx in by_index:
            raise SystemExit(f"duplicate candidate index {idx}")
        by_index[idx] = row
        file_hashes[idx] = hashlib.sha256(raw).hexdigest()

    expected_indices = set(range(384))
    got = set(by_index)
    if got != expected_indices:
        missing = sorted(expected_indices - got)
        print(json.dumps({
            "schema": "wfl-public-p43-final-plan-status-1",
            "status": "BLOCKED_INCOMPLETE_MEMBER_CONFIRMATION",
            "candidate_count": len(got),
            "required": 384,
            "missing_count": len(missing),
            "missing_indices": missing,
        }, indent=2))
        return

    registry = registry_rows()
    if len(registry) != 384:
        raise SystemExit("registry size changed")
    for idx, reg in enumerate(registry):
        row = by_index[idx]
        if row["candidate_id"] != reg["candidate_id"]:
            raise SystemExit(f"candidate index/id mismatch at {idx}")
        if row["candidate_signature_sha256"] != __import__(
            "wfl_public.p43_p42_train_screen", fromlist=["candidate_signature"]
        ).candidate_signature(reg):
            raise SystemExit(f"candidate signature mismatch at {idx}")

    families: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in by_index.items():
        families[row["core_family_key"]].append((idx, row))

    family_rows = []
    final_indices = []
    for key in sorted(families):
        members = sorted(families[key], key=lambda x: x[0])
        if len(members) != 16:
            raise SystemExit(f"family {key} has {len(members)} members, expected 16")
        classified = classify_family([r for _, r in members], "member_confirmation")
        retained = classified["classification"] == "FAMILY_RETAINED_COMPATIBLE_CLASS"
        family_rows.append({
            "core_family_key": key,
            "member_indices": [i for i, _ in members],
            "all_members_rejected": bool(classified.get("all_members_rejected", False)),
            "classification": classified["classification"],
            "retained_for_final_train": retained,
        })
        if retained:
            final_indices.extend(i for i, _ in members)

    plan = {
        "schema": "wfl-public-p43-final-train-plan-1",
        "status": (
            "FINAL_TRAIN_PLAN_FROZEN"
            if final_indices
            else "TERMINAL_ALL_FAMILIES_REJECTED"
        ),
        "member_confirmation": {
            "candidate_count": 384,
            "reps": 2000,
            "triage_alpha": 0.01,
            "validation_accessed": False,
            "holdout_accessed": False,
            "candidate_file_sha256": {
                str(i): file_hashes[i] for i in sorted(file_hashes)
            },
        },
        "family_count": len(family_rows),
        "retained_family_count": sum(
            1 for x in family_rows if x["retained_for_final_train"]
        ),
        "families": family_rows,
        "final_train": {
            "required_reps_per_candidate": 100000,
            "alpha": 0.05,
            "candidate_count": len(final_indices),
            "candidate_indices": sorted(final_indices),
            "rule": "all 16 members of every retained family",
        },
        "guardrails": {
            "complete_384_required": True,
            "family_rejection_requires_all_16_rejected_at_0_01": True,
            "no_validation_access": True,
            "no_holdout_access": True,
            "no_post_result_retuning": True,
        },
    }
    plan["plan_sha256"] = _hash(plan)

    if OUT.exists():
        existing = json.loads(OUT.read_text())
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise SystemExit("refusing to overwrite a different frozen FINAL_TRAIN plan")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
