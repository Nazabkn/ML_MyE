from __future__ import annotations

from kedro.pipeline import Pipeline

from .clustering.pipeline import create_pipeline as clustering_pipeline
from .dimensionality_reduction.pipeline import (
    create_pipeline as dimensionality_reduction_pipeline,
)
from .anomaly_detection.pipeline import create_pipeline as anomaly_detection_pipeline
from .association_rules.pipeline import create_pipeline as association_rules_pipeline


def create_pipeline(**kwargs) -> Pipeline:

    clustering = clustering_pipeline()
    dimensionality = dimensionality_reduction_pipeline()
    anomaly = anomaly_detection_pipeline()
    association = association_rules_pipeline()

    return clustering + dimensionality + anomaly + association