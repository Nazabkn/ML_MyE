from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Helperr
def _preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _estimator_by_name(name: str):
    
    name = (name or "").lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "knn":
        return KNeighborsClassifier()
    if name == "dt":
        return DecisionTreeClassifier()
    if name == "rf":
        return RandomForestClassifier(random_state=42)
    if name == "svc":
        return SVC(probability=True, random_state=42)
    raise ValueError(f"Estimator '{name}' no soportado.")



def build_splits(
    model_input_table: pd.DataFrame,
    params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    
    target = params["target"]
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    df = model_input_table.copy()
    y = df[[target]].astype(int)
    X = df.drop(columns=[target])

    
    user_cats = params.get("cat_features")
    cat_cols = [c for c in user_cats or [] if c in X.columns] or \
               X.select_dtypes(include=["object", "category"]).columns.tolist()

    user_nums = params.get("numeric_features")
    num_cols = [c for c in user_nums or [] if c in X.columns] or \
               X.select_dtypes(include=["number", "float", "int", "bool"]).columns.tolist()

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cols = {"cat": cat_cols, "num": num_cols}
    return X_train, X_test, y_train, y_test, cols


def model_selection(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    clf_cols: Dict[str, List[str]],
    params_processing: dict,
    models_cfg: dict,
):
    
    pre = _preprocessor(clf_cols["num"], clf_cols["cat"])

    scoring = models_cfg.get("scoring", "f1")
    cv = int(models_cfg.get("cv", 5))
    n_jobs = int(models_cfg.get("n_jobs", -1))

    best_model = None
    best_score = -np.inf

    all_cv_rows = []

    for model_name, spec in models_cfg.items():
        if model_name in {"scoring", "cv", "n_jobs", "num_features", "cat_features"}:
            continue

        est = _estimator_by_name(spec.get("estimator"))
        pipe = SkPipeline([("prep", pre), ("clf", est)])

        grid = spec.get("params", {})
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_train, y_train.values.ravel())

        
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df.insert(0, "model", model_name)
        all_cv_rows.append(cv_df)

        
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_

        

    clf_cv_results = pd.concat(all_cv_rows, ignore_index=True)
    return best_model, clf_cv_results


def predict(pipe: SkPipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return pd.DataFrame({"y_pred": y_pred, "y_proba": y_proba})


def export_leaderboard(clf_cv_results: pd.DataFrame) -> pd.DataFrame:

    cols_needed = {"model", "rank_test_score", "mean_test_score", "std_test_score", "params"}
    missing = cols_needed - set(clf_cv_results.columns)
    if missing:
        raise ValueError(f"Faltan columnas en cv_results: {sorted(missing)}")

    best_per_model = (
        clf_cv_results
        .sort_values(["model", "rank_test_score"])
        .groupby("model", as_index=False)
        .first()[["model", "mean_test_score", "std_test_score", "params"]]
        .rename(columns={
            "mean_test_score": "cv_mean_score",
            "std_test_score": "cv_std",
        })
        .sort_values("cv_mean_score", ascending=False)
        .reset_index(drop=True)
    )
    return best_per_model


def plot_cv_results(
    clf_results_table: pd.DataFrame,
    scoring: str = "f1",
) -> plt.Figure:

    if clf_results_table.empty:
        raise ValueError("No hay resultados para graficar.")

    order = clf_results_table.sort_values("cv_mean_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        order["model"],
        order["cv_mean_score"],
        xerr=order["cv_std"],
        color="#5B8FF9",
        alpha=0.8,
        ecolor="#1f1f1f",
        capsize=4,
    )
    ax.set_xlabel(f"{scoring} (mean ± std)")
    ax.set_ylabel("Modelo")
    ax.set_title("Comparativa de modelos de clasificación")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
