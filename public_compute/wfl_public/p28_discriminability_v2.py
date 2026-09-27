from __future__ import annotations

import pandas as pd

def attach_public_schedule(
    frame: pd.DataFrame,
    start_date: str = "2013-01-21",
    launch_partial_first_day: bool = True,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    start = pd.Timestamp(start_date)
    dates: list[str] = []
    times: list[str] = []
    for i in range(len(out)):
        if launch_partial_first_day and i < 12:
            day_offset = 0
            hour = 12 + i
        else:
            j = i - 12 if launch_partial_first_day else i
            day_offset = (1 + j // 17) if launch_partial_first_day else (j // 17)
            hour = 7 + (j % 17)
        dates.append((start + pd.Timedelta(days=day_offset)).date().isoformat())
        times.append(f"{hour:02d}:00")
    out["date"] = dates
    out["time"] = times
    return out
