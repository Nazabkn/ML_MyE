from __future__ import annotations
from kedro.pipeline import Pipeline

from spaceflights.pipelines.f01_understanding.pipeline import create_pipeline as p_understanding
from spaceflights.pipelines.f02_preprocessing.pipeline import create_pipeline as p_preprocessing
from spaceflights.pipelines.f03_unification.pipeline import create_pipeline as p_unification
from spaceflights.pipelines.f04_feature.pipeline import create_pipeline as p_feature
from spaceflights.pipelines.f05_classification.pipeline import create_pipeline as p_classif
from spaceflights.pipelines.f06_evaluation.pipeline import create_pipeline as p_eval
from spaceflights.pipelines.f07_regression.pipeline import create_pipeline as p_regression

def register_pipelines() -> dict[str, Pipeline]:
    understanding = p_understanding()
    preprocessing = p_preprocessing()
    unification = p_unification()
    feature = p_feature()
    classification = p_classif()
    evaluation = p_eval()
    regression = p_regression()

    default = understanding + preprocessing + unification + feature + classification + evaluation + regression

    return {
        "__default__": default,
        "understanding": understanding,
        "preprocessing": preprocessing,
        "unification": unification,
        "feature": feature,
        "classification": classification,
        "evaluation": evaluation,
        "regression": regression,
    }
