from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from wfl_public.p43_p42_train_screen import (
    _feature_vector,
    registry_rows,
    simulate_candidate,
    train_frame_from_full_history,
)

def load_train() -> pd.DataFrame:
    parts=[pd.read_csv(p) for p in sorted(Path("public_compute/data").glob("wfl_train_*.csv"))]
    return train_frame_from_full_history(pd.concat(parts, ignore_index=True))

train=load_train()
row=registry_rows()[0]

t0=perf_counter()
obs_names, obs_vec=_feature_vector(train)
t1=perf_counter()

frame=simulate_candidate(row, len(train), 202609264301)
t2=perf_counter()

names, vec=_feature_vector(frame)
t3=perf_counter()

print({
    "train_rows": len(train),
    "observed_fingerprint_seconds": t1-t0,
    "simulate_one_history_seconds": t2-t1,
    "fingerprint_one_history_seconds": t3-t2,
    "one_rep_total_seconds": t3-t1,
    "feature_count": len(names),
    "schema_match": names == obs_names,
})
