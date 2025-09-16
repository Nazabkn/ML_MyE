"""Comprension!"""
from __future__ import annotations
import pandas as pd

def summarize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    
    try:
       
        desc = df.describe(include="all", datetime_is_numeric=True).T
    except TypeError:
     
        desc = df.describe(include="all").T

    desc["dtype"] = df.dtypes.astype(str)
    desc.insert(0, "dataset", name)
    return desc.reset_index(names="column")

