from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

from hourly_twin import ROME, freeze

HOURS = tuple(range(7, 24))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--candidate-index", type=int, default=16)
    ap.add_argument("--min-lead-seconds", type=int, default=300)
    args = ap.parse_args()

    now = datetime.now(ROME)
    target_day = date.fromisoformat(args.date) if args.date else now.date()

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
        if (target - now).total_seconds() < args.min_lead_seconds:
            continue
        result = freeze(
            candidate_index=args.candidate_index,
            now=now,
            target=target,
            result_root=Path("public_compute/results/member_confirmation"),
            out_root=Path("public_compute/prospective/hourly"),
            min_lead_seconds=args.min_lead_seconds,
        )
        results.append(result)

    print(json.dumps({
        "schema": "wfl-hourly-twin-prefreeze-day-run-1",
        "generated_at": now.isoformat(),
        "target_date": target_day.isoformat(),
        "candidate_index": args.candidate_index,
        "future_targets_processed": len(results),
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
