"""Unificación NEO ⟵ Meteorites por año (model_input_table)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def aggregate_meteorites_by_year(m: pd.DataFrame) -> pd.DataFrame:
 
    m = m.copy()
    m["year"] = m["year"].astype("Int64")

    if "id" in m.columns:
        met_count = m.groupby("year", dropna=False)["id"].count().rename("met_count")
    else:
        met_count = m.groupby("year", dropna=False).size().rename("met_count")


    if "fall" in m.columns:
        fell = (m["fall"].astype(str).str.lower() == "fell").astype(int)
        met_fell = fell.groupby(m["year"]).sum().rename("met_fell_count")
    else:
        met_fell = met_count.mul(0).rename("met_fell_count")

    if "mass" in m.columns:
        mass = pd.to_numeric(m["mass"], errors="coerce")
        met_mass_mean = mass.groupby(m["year"]).mean().rename("met_mass_mean")
        met_mass_median = mass.groupby(m["year"]).median().rename("met_mass_median")
    else:
        idx = met_count.index
        met_mass_mean = pd.Series(np.nan, index=idx, name="met_mass_mean")
        met_mass_median = pd.Series(np.nan, index=idx, name="met_mass_median")

    agg = pd.concat([met_count, met_fell, met_mass_mean, met_mass_median], axis=1).reset_index()
    return agg

def unify_by_year(neo_clean: pd.DataFrame, met_agg: pd.DataFrame) -> pd.DataFrame:

    neo = neo_clean.copy()
    agg = met_agg.copy()

    unified = neo.merge(agg, on="year", how="left")

    for c in ["met_count", "met_fell_count"]:
        unified[c] = unified[c].fillna(0).astype(int)

    if "met_fell_ratio" not in unified.columns:
        unified["met_fell_ratio"] = (
            unified["met_fell_count"] / unified["met_count"].replace({0: np.nan})
        ).fillna(0.0)

    for c in ["met_mass_mean", "met_mass_median", "met_fell_ratio"]:
        unified[c] = unified[c].astype("float64")

    return unified
