from __future__ import annotations
from kedro.pipeline import Pipeline

# Pipelines supervisados
from .f01_understanding.pipeline import create_pipeline as p_understanding
from .f02_preprocessing.pipeline import create_pipeline as p_preprocessing
from .f03_unification.pipeline import create_pipeline as p_unification
from .f04_feature.pipeline import create_pipeline as p_feature
from .f05_classification.pipeline import create_pipeline as p_classif
from .f06_evaluation.pipeline import create_pipeline as p_eval
from .f07_regression.pipeline import create_pipeline as p_regression

# Pipelines no supervisados (
from .unsupervised_learning.clustering.pipeline import create_pipeline as clustering_pipeline
from .unsupervised_learning.dimensionality_reduction.pipeline import (
    create_pipeline as dimensionality_pipeline,
)
from .unsupervised_learning.anomaly_detection.pipeline import (
    create_pipeline as anomaly_pipeline,
)
from .unsupervised_learning.association_rules.pipeline import (
    create_pipeline as association_pipeline,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([])
