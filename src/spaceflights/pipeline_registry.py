from __future__ import annotations
from kedro.pipeline import Pipeline

from spaceflights.pipelines.f01_understanding.pipeline import create_pipeline as p_understanding
from spaceflights.pipelines.f02_preprocessing.pipeline import create_pipeline as p_preprocessing
from spaceflights.pipelines.f03_unification.pipeline import create_pipeline as p_unification
from spaceflights.pipelines.f04_feature.pipeline import create_pipeline as p_feature
from spaceflights.pipelines.f05_classification.pipeline import create_pipeline as p_classif


def register_pipelines() -> dict[str, Pipeline]:
    understanding = p_understanding()
    preprocessing = p_preprocessing()
    unification = p_unification()
    feature = p_feature()
    classification = p_classif()

    default = understanding + preprocessing + unification + feature + classification

    return {
        "__default__": default,
        "understanding": understanding,
        "preprocessing": preprocessing,
        "unification": unification,
        "feature": feature,
        "classification": classification,
    }
