from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Rome")
DAILY_HOURS = tuple(range(7, 24))
CONTESTS_PER_DAY = len(DAILY_HOURS)
DEFAULT_SYNTHETIC_EPOCH = "2013-01-21"

def _parse_epoch(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))

def schedule_for_ordinal(
    ordinal: int,
    epoch_date: str | date = DEFAULT_SYNTHETIC_EPOCH,
) -> datetime:
    ordinal = int(ordinal)
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    epoch = _parse_epoch(epoch_date)
    day_index, slot = divmod(ordinal, CONTESTS_PER_DAY)
    d = epoch + timedelta(days=day_index)
    return datetime.combine(d, time(DAILY_HOURS[slot], 0), tzinfo=TZ)
