"""Limpieza de neo y meteorites :3"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

def _extract_year_from_name(s: object) -> pd.Series:
    if pd.isna(s):
        return pd.NA
    m = re.search(r"((19|20)\d{2})", str(s))
    return int(m.group(1)) if m else pd.NA

def _to_numeric_if_exists(df: pd.DataFrame, cols: list[str]) -> None:

    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- NEO ----------
def clean_neo(n: pd.DataFrame) -> pd.DataFrame:

    n = n.copy()

    if "year" not in n.columns or n["year"].isna().all():
        if "name" in n.columns:
            n["year"] = n["name"].apply(_extract_year_from_name).astype("Int64")
        else:
            n["year"] = pd.Series(pd.NA, index=n.index, dtype="Int64")

   
    if "diameter_mean" not in n.columns:
        _to_numeric_if_exists(n, ["est_diameter_min", "est_diameter_max"])
        if {"est_diameter_min", "est_diameter_max"}.issubset(n.columns):
            n["diameter_mean"] = (n["est_diameter_min"] + n["est_diameter_max"]) / 2

    _to_numeric_if_exists(
        n,
        [
            "relative_velocity",
            "miss_distance",
            "absolute_magnitude",
            "diameter_mean",
        ],
    )


    if "relative_velocity" in n.columns and "log_velocity" not in n.columns:
        n["log_velocity"] = np.log1p(n["relative_velocity"])
    if "miss_distance" in n.columns and "log_miss_distance" not in n.columns:
        n["log_miss_distance"] = np.log1p(n["miss_distance"])


    if "sentry_object" in n.columns:
        n["sentry_object"] = n["sentry_object"].astype("Int64")
    if "hazardous" in n.columns:
        n["hazardous"] = n["hazardous"].astype("Int64")

    return n

def clean_meteorites(m: pd.DataFrame) -> pd.DataFrame:

    m = m.copy()

    if "GeoLocation" in m.columns:
        m["GeoLocation"] = m["GeoLocation"].fillna("Unknown")

    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    m.loc[(m["year"] < 860) | (m["year"] > 2025), "year"] = pd.NA
    m = m.dropna(subset=["year"])
    m["year"] = m["year"].astype("Int64")

    if "mass" in m.columns and "log_mass" not in m.columns:
        m["mass"] = pd.to_numeric(m["mass"], errors="coerce")
        m["log_mass"] = np.log1p(m["mass"])

    return m
