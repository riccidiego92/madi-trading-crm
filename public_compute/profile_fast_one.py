from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from wfl_public.p42_fast import simulate_phased_fast
from wfl_public.p42_phased_draw_machine import phase_grid
from wfl_public.p29_metrics import complete_fingerprint, complete_fingerprint_vector
from wfl_public.p28_discriminability_v2 import attach_public_schedule

N=36902
cfg=replace(phase_grid()[0], experiment_seed=202609264301)

t0=perf_counter()
frame=simulate_phased_fast(N, cfg)
t1=perf_counter()
frame=attach_public_schedule(frame, start_date="2013-01-21", launch_partial_first_day=True)
names, vec=complete_fingerprint_vector(complete_fingerprint(frame))
t2=perf_counter()

print({
  "n_contests":N,
  "fast_simulate_seconds":t1-t0,
  "fingerprint_seconds":t2-t1,
  "total_seconds":t2-t0,
  "feature_count":len(names),
})
