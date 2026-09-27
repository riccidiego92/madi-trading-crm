from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from wfl_public.p43_p42_train_screen import (
    observed_complete_vector,
    registry_rows,
    streaming_train_evaluation,
    train_frame_from_full_history,
)
from wfl_public.p43_parallel import parallel_train_evaluation


def load_train():
    parts=[pd.read_csv(p) for p in sorted(Path("public_compute/data").glob("wfl_train_*.csv"))]
    return train_frame_from_full_history(pd.concat(parts, ignore_index=True))


def scientific_view(x):
    y=json.loads(json.dumps(x))
    y.pop("execution", None)
    y.get("guardrails", {}).pop("parallel_execution_only", None)
    return y


train=load_train()
observed=observed_complete_vector(train)
rows=registry_rows()

# Cover materially different P42 lifecycle/stream modes without retuning.
chosen=[]
seen=set()
for row in rows:
    cfg=row["config"]
    key=(cfg["stream_mode"], cfg["reseed_mode"])
    if key not in seen:
        seen.add(key)
        chosen.append(row["candidate_id"])
assert len(chosen)==6

for cid in chosen:
    seq=streaming_train_evaluation(train,cid,4,observed=observed)
    par=parallel_train_evaluation(train,cid,4,observed=observed,workers=4)
    if scientific_view(seq) != scientific_view(par):
        raise SystemExit(
            "P43_PARALLEL_MISMATCH "+cid+"\n"+
            json.dumps({"sequential":scientific_view(seq),"parallel":scientific_view(par)},sort_keys=True)
        )

print("P43_PARALLEL_EQUIVALENCE_OK", len(chosen), "representative candidates")
