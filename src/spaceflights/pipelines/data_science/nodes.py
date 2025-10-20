import logging
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)

def split_data(data: pd.DataFrame, parameters: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = parameters["target"]
    X = data.drop(columns=[target])
    y = data[target]
    return train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y
    )

def build_pipeline(parameters: dict, feature_names: Dict[str, Any]) -> Pipeline:
    num_features = feature_names.get("num_features", [])
    cat_features = feature_names.get("cat_features", [])

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=300,
        class_weight=parameters.get("class_weight", None)
    )


    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict, feature_names: dict):
    pipe = build_pipeline(parameters, feature_names)
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    out = {
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
    }
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        out["roc_auc"] = None
    logger.info("Metrics: %s", out)
    return out
