from __future__ import annotations

from dataclasses import replace

from wfl_public.p42_fast import simulate_phased_fast
from wfl_public.p42_phased_draw_machine import phase_grid, simulate_phased

COLS=[f"n{i}" for i in range(1,11)]+["numerone"]

configs=phase_grid()
checked=0
for seed in (202609264301, 202609264302):
    for base in configs:
        cfg=replace(base, experiment_seed=seed)
        ref=simulate_phased(25, cfg)[COLS].reset_index(drop=True)
        fast=simulate_phased_fast(25, cfg)[COLS].reset_index(drop=True)
        if not ref.equals(fast):
            raise SystemExit(f"FAST_PATH_MISMATCH {cfg.identity()} seed={seed}")
        checked += 1

print("P42_FAST_EQUIVALENCE_OK", checked, "config-seed cases")
