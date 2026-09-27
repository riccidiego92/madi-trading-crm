from __future__ import annotations

import pandas as pd

TRAIN_START = "2013-01-21"
TRAIN_END = "2018-12-31"

def stage_frame(df: pd.DataFrame, stage: str, unlock_holdout: bool = False) -> pd.DataFrame:
    if stage != "train":
        raise RuntimeError("public compute lane is TRAIN-only")
    z = df.copy()
    dates = pd.to_datetime(z["date"], errors="raise")
    mask = (dates >= pd.Timestamp(TRAIN_START)) & (dates <= pd.Timestamp(TRAIN_END))
    return z.loc[mask].reset_index(drop=True)
