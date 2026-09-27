from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wfl_public.p43_p42_train_screen import (
    observed_complete_vector,
    registry_rows,
    streaming_train_evaluation,
    train_frame_from_full_history,
)

def load_train_data(root: Path) -> pd.DataFrame:
    parts = [
        pd.read_csv(p)
        for p in sorted(root.glob("wfl_train_*.csv"))
    ]
    if not parts:
        raise SystemExit("no TRAIN data files found")
    df = pd.concat(parts, ignore_index=True)
    return train_frame_from_full_history(df)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-id")
    ap.add_argument("--candidate-index", type=int)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--data-dir", default="public_compute/data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = registry_rows()
    if args.candidate_id is not None:
        matches = [r for r in rows if r["candidate_id"] == args.candidate_id]
        if len(matches) != 1:
            raise SystemExit("candidate_id not found uniquely")
        cid = matches[0]["candidate_id"]
    elif args.candidate_index is not None:
        if not 0 <= args.candidate_index < len(rows):
            raise SystemExit("candidate-index out of range")
        cid = rows[args.candidate_index]["candidate_id"]
    else:
        raise SystemExit("candidate-id or candidate-index is required")

    train = load_train_data(Path(args.data_dir))
    observed = observed_complete_vector(train)
    result = streaming_train_evaluation(
        train,
        cid,
        args.reps,
        observed=observed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate_id": cid,
        "reps": args.reps,
        "p_max_stat": result["test"]["p_max_stat"],
        "out": str(out),
    }))

if __name__ == "__main__":
    main()
