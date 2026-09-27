from __future__ import annotations

from dataclasses import replace
import numpy as np

from wfl_public.p42_fast import simulate_phased_fast
from wfl_public.p42_phased_draw_machine import phase_grid, simulate_phased

COLS=[f"n{i}" for i in range(1,11)]+["numerone"]

configs=phase_grid()
checked=0
for seed in (202609264301, 202609264302):
    for base in configs:
        cfg=replace(base, experiment_seed=seed)
        ref=simulate_phased(25, cfg)[COLS].to_numpy(dtype=np.int64)
        fast=simulate_phased_fast(25, cfg)[COLS].to_numpy(dtype=np.int64)
        if not np.array_equal(ref, fast):
            where=np.argwhere(ref != fast)
            i,j=map(int, where[0])
            raise SystemExit(
                "FAST_PATH_VALUE_MISMATCH "
                f"{cfg.identity()} seed={seed} row={i} col={COLS[j]} "
                f"reference={int(ref[i,j])} fast={int(fast[i,j])}"
            )
        checked += 1

print("P42_FAST_EQUIVALENCE_OK", checked, "config-seed cases")
