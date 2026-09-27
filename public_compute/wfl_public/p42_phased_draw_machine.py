from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable

import pandas as pd

from .analysis import MAIN_COLS
from .combinatorics import (
    canonical_pair_rank,
    combination_rank,
    complement_numbers,
    joint_jackpot_state,
)
from .p26_public_drbg import (
    HmacDrbgSha256,
    draw_main as draw_drbg_main,
    draw_numerone as draw_drbg_numerone,
)
from .p40_machine_emulator import schedule_for_ordinal

MAPPERS = ("fisher_yates", "sequential", "rejection", "rank_unrank")
STREAM_MODES = ("single", "split_main_numerone")
RESEED_MODES = ("continuous", "per_day", "per_contest")
BACKGROUND_CYCLES = (0, 1, 4, 16)
CONTESTS_PER_DAY = 17


@dataclass(frozen=True)
class PhasedMachineConfig:
    mapper: str = "fisher_yates"
    stream_mode: str = "single"
    reseed_mode: str = "per_day"
    background_cycles_main: int = 0
    background_cycles_numerone: int = 0
    experiment_seed: int = 2026092642
    contests_per_day: int = CONTESTS_PER_DAY

    def validate(self) -> None:
        if self.mapper not in MAPPERS:
            raise ValueError(f"unknown mapper {self.mapper}")
        if self.stream_mode not in STREAM_MODES:
            raise ValueError(f"unknown stream mode {self.stream_mode}")
        if self.reseed_mode not in RESEED_MODES:
            raise ValueError(f"unknown reseed mode {self.reseed_mode}")
        if int(self.background_cycles_main) not in BACKGROUND_CYCLES:
            raise ValueError("unsupported background_cycles_main")
        if int(self.background_cycles_numerone) not in BACKGROUND_CYCLES:
            raise ValueError("unsupported background_cycles_numerone")
        if int(self.contests_per_day) != CONTESTS_PER_DAY:
            raise ValueError("P42 is frozen to 17 contests/day")

    def identity(self) -> str:
        self.validate()
        return (
            "P42_HMAC_DRBG_SHA256|"
            f"{self.mapper}|{self.stream_mode}|{self.reseed_mode}|"
            f"bgm{int(self.background_cycles_main)}|"
            f"bgn{int(self.background_cycles_numerone)}"
        )

    def candidate_id(self) -> str:
        digest = sha256(self.identity().encode("utf-8")).hexdigest()[:16]
        return f"P42_PHASED:{digest}"

    def to_dict(self) -> dict:
        return asdict(self)


def _seed_material(config: PhasedMachineConfig, scope: object, label: str) -> bytes:
    payload = (
        f"P42|{int(config.experiment_seed)}|{config.identity()}|{scope}|{label}"
    ).encode("utf-8")
    return sha256(b"A|" + payload).digest() + sha256(b"B|" + payload).digest()[:16]


def _pair(
    config: PhasedMachineConfig,
    scope: object,
) -> tuple[HmacDrbgSha256, HmacDrbgSha256]:
    main = HmacDrbgSha256(_seed_material(config, scope, "main"))
    if config.stream_mode == "single":
        return main, main
    num = HmacDrbgSha256(_seed_material(config, scope, "numerone"))
    return main, num


def _state_fingerprint(rng: HmacDrbgSha256) -> str:
    counter = int(rng.reseed_counter).to_bytes(8, "big", signed=False)
    return sha256(
        b"P42STATE|" + bytes(rng.key) + bytes(rng.value) + counter
    ).hexdigest()


def _pair_fingerprint(
    main_rng: HmacDrbgSha256,
    num_rng: HmacDrbgSha256,
) -> dict:
    return {
        "main": _state_fingerprint(main_rng),
        "numerone": _state_fingerprint(num_rng),
        "shared_object": main_rng is num_rng,
    }


def _background(rng: HmacDrbgSha256, cycles: int) -> None:
    for _ in range(int(cycles)):
        rng.generate(8)


def _hash_payload(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def _append_event(
    events: list[dict],
    *,
    contest_index: int,
    phase: str,
    payload: dict,
) -> str:
    prev = events[-1]["event_sha256"] if events else None
    row = {
        "schema": "wfl-p42-phase-event-1",
        "contest_index": int(contest_index),
        "phase": str(phase),
        "previous_event_sha256": prev,
        **payload,
    }
    row["event_sha256"] = _hash_payload(row)
    events.append(row)
    return row["event_sha256"]


def _validate_result(main: Iterable[int], numerone: int) -> tuple[int, ...]:
    vals = tuple(sorted(int(x) for x in main))
    if len(vals) != 10 or len(set(vals)) != 10 or vals[0] < 1 or vals[-1] > 20:
        raise AssertionError(f"invalid Main output {vals}")
    if not 1 <= int(numerone) <= 20:
        raise AssertionError(f"invalid Numerone output {numerone}")
    return vals


def run_phased(
    n_contests: int,
    config: PhasedMachineConfig,
    *,
    epoch_date: str = "2013-01-21",
) -> tuple[pd.DataFrame, list[dict]]:
    config.validate()
    n_contests = int(n_contests)
    if n_contests < 0:
        raise ValueError("n_contests must be non-negative")

    events: list[dict] = []
    rows: list[dict] = []

    pair = None
    active_scope = None

    for i in range(n_contests):
        day_index = i // config.contests_per_day
        slot = i % config.contests_per_day
        target = schedule_for_ordinal(i, epoch_date=epoch_date)

        _append_event(
            events,
            contest_index=i,
            phase="SCHEDULE",
            payload={
                "target_datetime": target.isoformat(),
                "synthetic_day": day_index,
                "synthetic_slot": slot,
            },
        )

        if config.reseed_mode == "continuous":
            wanted_scope = "continuous"
        elif config.reseed_mode == "per_day":
            wanted_scope = f"day:{day_index}"
        elif config.reseed_mode == "per_contest":
            wanted_scope = f"contest:{i}"
        else:
            raise ValueError(config.reseed_mode)

        action = "reuse"
        if pair is None or wanted_scope != active_scope:
            active_scope = wanted_scope
            pair = _pair(config, wanted_scope)
            action = "instantiate"
        main_rng, num_rng = pair

        _append_event(
            events,
            contest_index=i,
            phase="INIT_OR_RESEED",
            payload={
                "scope": wanted_scope,
                "action": action,
                "state_after": _pair_fingerprint(main_rng, num_rng),
            },
        )

        state_before = _pair_fingerprint(main_rng, num_rng)
        _background(main_rng, config.background_cycles_main)
        state_after = _pair_fingerprint(main_rng, num_rng)
        _append_event(
            events,
            contest_index=i,
            phase="BACKGROUND_MAIN",
            payload={
                "cycles": int(config.background_cycles_main),
                "state_before": state_before,
                "state_after": state_after,
            },
        )

        state_before = _pair_fingerprint(main_rng, num_rng)
        main = draw_drbg_main(main_rng, config.mapper)
        state_after = _pair_fingerprint(main_rng, num_rng)
        _append_event(
            events,
            contest_index=i,
            phase="MAIN_DRAW",
            payload={
                "mapper": config.mapper,
                "main": list(main),
                "state_before": state_before,
                "state_after": state_after,
            },
        )

        state_before = _pair_fingerprint(main_rng, num_rng)
        _background(num_rng, config.background_cycles_numerone)
        state_after = _pair_fingerprint(main_rng, num_rng)
        _append_event(
            events,
            contest_index=i,
            phase="BACKGROUND_NUMERONE",
            payload={
                "cycles": int(config.background_cycles_numerone),
                "state_before": state_before,
                "state_after": state_after,
            },
        )

        state_before = _pair_fingerprint(main_rng, num_rng)
        numerone = draw_drbg_numerone(num_rng)
        state_after = _pair_fingerprint(main_rng, num_rng)
        _append_event(
            events,
            contest_index=i,
            phase="NUMERONE_DRAW",
            payload={
                "numerone": int(numerone),
                "state_before": state_before,
                "state_after": state_after,
            },
        )

        vals = _validate_result(main, numerone)
        complement = complement_numbers(vals)
        public = {
            "candidate_id": config.candidate_id(),
            "candidate_identity": config.identity(),
            "synthetic_index": i,
            "synthetic_day": day_index,
            "synthetic_contest_in_day": slot,
            "target_datetime": target.isoformat(),
            "main": list(vals),
            "numerone": int(numerone),
            "complement": list(complement),
            "main_combination_rank": int(combination_rank(vals)),
            "main_complement_pair_rank": int(canonical_pair_rank(vals)),
            "exact_main_plus_numerone_state": int(
                combination_rank(vals) * 20 + (int(numerone) - 1)
            ),
            "complement_pair_plus_numerone_state": int(
                joint_jackpot_state(vals, numerone)
            ),
        }
        public_hash = _hash_payload(public)

        final_event_hash = _append_event(
            events,
            contest_index=i,
            phase="COMMIT",
            payload={
                "public_output_sha256": public_hash,
                "state_after": _pair_fingerprint(main_rng, num_rng),
            },
        )

        row = {
            "synthetic_index": i,
            "synthetic_day": day_index,
            "synthetic_contest_in_day": slot,
            "target_datetime": target.isoformat(),
            "candidate_id": config.candidate_id(),
            "candidate_identity": config.identity(),
            "numerone": int(numerone),
            "public_output_sha256": public_hash,
            "audit_tail_sha256": final_event_hash,
        }
        for j, value in enumerate(vals, start=1):
            row[f"n{j}"] = int(value)
        rows.append(row)

    frame = pd.DataFrame(
        rows,
        columns=[
            "synthetic_index",
            "synthetic_day",
            "synthetic_contest_in_day",
            "target_datetime",
            "candidate_id",
            "candidate_identity",
            *MAIN_COLS,
            "numerone",
            "public_output_sha256",
            "audit_tail_sha256",
        ],
    )
    return frame, events


def simulate_phased(
    n_contests: int,
    config: PhasedMachineConfig,
    *,
    epoch_date: str = "2013-01-21",
) -> pd.DataFrame:
    frame, _ = run_phased(n_contests, config, epoch_date=epoch_date)
    return frame


def phase_grid(seed_root: int = 2026092642) -> list[PhasedMachineConfig]:
    out: list[PhasedMachineConfig] = []
    for mapper in MAPPERS:
        for stream_mode in STREAM_MODES:
            for reseed_mode in RESEED_MODES:
                for bg_main in BACKGROUND_CYCLES:
                    for bg_num in BACKGROUND_CYCLES:
                        out.append(
                            PhasedMachineConfig(
                                mapper=mapper,
                                stream_mode=stream_mode,
                                reseed_mode=reseed_mode,
                                background_cycles_main=bg_main,
                                background_cycles_numerone=bg_num,
                                experiment_seed=int(seed_root),
                            )
                        )
    return out


def registry_report(seed_root: int = 2026092642) -> dict:
    rows = phase_grid(seed_root)
    return {
        "schema": "wfl-p42-phased-machine-registry-1",
        "candidate_count": len(rows),
        "candidates": [
            {
                "candidate_id": cfg.candidate_id(),
                "identity": cfg.identity(),
                "config": cfg.to_dict(),
                "canonical_public_analogue": {
                    "rng_source": "HMAC_DRBG_SHA256 synthetic comparator",
                    "draw_routine": cfg.mapper,
                    "sampling_without_replacement": True,
                    "stream_mode": cfg.stream_mode,
                    "lifecycle": cfg.reseed_mode,
                    "background_cycles_main": cfg.background_cycles_main,
                    "background_cycles_numerone": cfg.background_cycles_numerone,
                },
            }
            for cfg in rows
        ],
        "guardrails": {
            "public_experiment_seed_only": True,
            "operational_seed_search": False,
            "private_state_or_credentials": False,
            "claims_actual_classico_rng": False,
            "train_only_phase_screening": True,
            "prospective_retuning": False,
        },
    }
