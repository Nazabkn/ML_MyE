from __future__ import annotations
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_splits(
    model_input_table: pd.DataFrame,
    params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
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
               X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cols = {"cat": cat_cols, "num": num_cols}
    return X_train, X_test, y_train, y_test, cols


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
    raise ValueError(f"Estimator '{name}' not supported.")


def model_selection(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    cols: Dict,
    params_processing: dict,
    models_cfg: dict,
):

    prep = _preprocessor(cols["num"], cols["cat"])

    scoring = models_cfg.get("scoring", "f1")
    cv = int(models_cfg.get("cv", 5))
    n_jobs = int(models_cfg.get("n_jobs", -1))

    rows = []
    all_cv_rows = []
    best_model = None
    best_score = -np.inf
    best_name = ""


    for model_name, block in models_cfg.items():
        if model_name in {"scoring", "cv", "n_jobs", "num_features", "cat_features"}:
            continue

        est_key = str(block.get("estimator"))
        est = _estimator_by_name(est_key)

        pipe = SkPipeline(steps=[("prep", prep), ("clf", est)])

        grid = block.get("params", {})
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


        best_pipe = gs.best_estimator_
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
        y_hat = (y_prob >= 0.5).astype(int)

        row = {
            "model": model_name,
            "best_params": gs.best_params_,
            "cv_best_score": float(gs.best_score_),
            "f1": float(f1_score(y_test, y_hat, zero_division=0)),
            "recall": float(recall_score(y_test, y_hat, zero_division=0)),
            "precision": float(precision_score(y_test, y_hat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        rows.append(row)

        #acumular los resultados del grids
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df.insert(0, "model", model_name)
        all_cv_rows.append(cv_df)

        #trackear global por la metrica principal
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = best_pipe
            best_name = model_name

    results_df = pd.DataFrame(rows).sort_values("cv_best_score", ascending=False)
    cv_results = pd.concat(all_cv_rows, ignore_index=True)

    #guardando el nombre C:
    results_df.insert(0, "best_model", [best_name] + [""] * (len(results_df) - 1))

    return best_model, results_df, cv_results



def predict(pipe: SkPipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)
    return pd.DataFrame({"y_pred": y_hat, "y_proba": y_prob})


def export_leaderboard(eval_metrics: pd.DataFrame) -> pd.DataFrame:

    return eval_metrics
