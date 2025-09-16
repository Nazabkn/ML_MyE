from __future__ import annotations
from kedro.pipeline import Pipeline

from spaceflights.pipelines.f01_understanding.pipeline import create_pipeline as p_understanding
from spaceflights.pipelines.f02_preprocessing.pipeline import create_pipeline as p_preprocessing
from spaceflights.pipelines.f03_unification.pipeline import create_pipeline as p_unification


def register_pipelines() -> dict[str, Pipeline]:
    understanding = p_understanding()
    preprocessing = p_preprocessing()
    unification = p_unification()

    default = understanding + preprocessing + unification

    return {
        "__default__": default,
        "understanding": understanding,
        "preprocessing": preprocessing,
        "unification": unification,
    }
