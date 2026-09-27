from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from wfl_public.p43_p42_train_screen import (
    observed_complete_vector,
    registry_rows,
    train_frame_from_full_history,
)
from wfl_public.p43_parallel import parallel_train_evaluation

parts=[pd.read_csv(p) for p in sorted(Path("public_compute/data").glob("wfl_train_*.csv"))]
train=train_frame_from_full_history(pd.concat(parts,ignore_index=True))
observed=observed_complete_vector(train)
cid=registry_rows()[0]["candidate_id"]

t0=perf_counter()
r=parallel_train_evaluation(train,cid,16,observed=observed,workers=4)
t1=perf_counter()
print({
    "candidate_id":cid,
    "reps":16,
    "workers":r["execution"]["workers"],
    "wall_seconds":t1-t0,
    "seconds_per_rep_wall":(t1-t0)/16.0,
    "projected_2000_rep_hours":((t1-t0)/16.0)*2000/3600.0,
    "p_max_stat":r["test"]["p_max_stat"],
})
