from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import MAIN_COLS
from .p26_public_drbg import draw_main as draw_drbg_main, draw_numerone as draw_drbg_numerone
from .p42_phased_draw_machine import PhasedMachineConfig, _background, _pair, _validate_result


def simulate_phased_fast(n_contests: int, config: PhasedMachineConfig) -> pd.DataFrame:
    """P42 output-equivalent fast path for statistical simulation.

    It deliberately omits P42 audit-event construction, state fingerprints,
    public hashes, complement/rank metadata, and datetime serialization.
    The RNG instantiation, reseed scopes, background cycles, Main mapper,
    Numerone draw, and draw ordering are identical to run_phased().
    """
    config.validate()
    n_contests = int(n_contests)
    if n_contests < 0:
        raise ValueError("n_contests must be non-negative")

    main_out = np.empty((n_contests, 10), dtype=np.int16)
    num_out = np.empty(n_contests, dtype=np.int16)

    pair = None
    active_scope = None

    for i in range(n_contests):
        day_index = i // config.contests_per_day

        if config.reseed_mode == "continuous":
            wanted_scope = "continuous"
        elif config.reseed_mode == "per_day":
            wanted_scope = f"day:{day_index}"
        elif config.reseed_mode == "per_contest":
            wanted_scope = f"contest:{i}"
        else:
            raise ValueError(config.reseed_mode)

        if pair is None or wanted_scope != active_scope:
            active_scope = wanted_scope
            pair = _pair(config, wanted_scope)
        main_rng, num_rng = pair

        _background(main_rng, config.background_cycles_main)
        main = draw_drbg_main(main_rng, config.mapper)

        _background(num_rng, config.background_cycles_numerone)
        numerone = draw_drbg_numerone(num_rng)

        vals = _validate_result(main, numerone)
        main_out[i, :] = vals
        num_out[i] = int(numerone)

    data = {
        "synthetic_index": np.arange(n_contests, dtype=np.int64),
        "synthetic_day": np.arange(n_contests, dtype=np.int64) // int(config.contests_per_day),
        "synthetic_contest_in_day": np.arange(n_contests, dtype=np.int64) % int(config.contests_per_day),
    }
    for j, col in enumerate(MAIN_COLS):
        data[col] = main_out[:, j]
    data["numerone"] = num_out
    return pd.DataFrame(data)
