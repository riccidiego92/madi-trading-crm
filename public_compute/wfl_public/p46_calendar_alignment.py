from __future__ import annotations

from .p42_phased_draw_machine import PhasedMachineConfig

LAUNCH_PARTIAL_MISSING_SLOTS = 5


def calendar_alignment_offset(config: PhasedMachineConfig) -> int:
    """Documentary offset induced by Classico's 12-draw launch day.

    P42 internal daily lifecycle is 17 slots/day from machine index zero.
    The public launch day exposed only 12 slots (12:00..23:00), so a per-day
    lifecycle must start at internal slot 5 to make the next public 07:00 draw
    coincide with the next internal day boundary.

    This is public schedule structure, not outcome-fitted phase tuning.
    """
    config.validate()
    return (
        LAUNCH_PARTIAL_MISSING_SLOTS
        if config.reseed_mode == "per_day"
        else 0
    )


def aligned_machine_ordinal(
    public_ordinal: int,
    config: PhasedMachineConfig,
    research_phase_offset: int = 0,
) -> int:
    public_ordinal = int(public_ordinal)
    research_phase_offset = int(research_phase_offset)
    if public_ordinal < 0:
        raise ValueError("public_ordinal must be non-negative")
    out = (
        public_ordinal
        + calendar_alignment_offset(config)
        + research_phase_offset
    )
    if out < 0:
        raise ValueError("alignment yields negative machine ordinal")
    return out
