from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



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
    if name == "linreg":
        return LinearRegression()
    if name == "ridge":
        return Ridge(random_state=None)
    if name == "lasso":
        return Lasso(max_iter=10000)
    if name == "rf":
        return RandomForestRegressor(random_state=42)
    if name == "svr":
        return SVR(kernel="rbf")
    raise ValueError(f"Regressor '{name}' no soportado.")



def build_splits_reg(
    model_input_table: pd.DataFrame,
    params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
   
    target = params.get("target_reg", params["target"])
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    df = model_input_table.copy()
    y = df[[target]]
    X = df.drop(columns=[target])

   
    user_cats = params.get("cat_features")
    cat_cols = [c for c in (user_cats or []) if c in X.columns] or \
               X.select_dtypes(include=["object", "category"]).columns.tolist()

    user_nums = params.get("numeric_features")
    num_cols = [c for c in (user_nums or []) if c in X.columns] or \
               X.select_dtypes(include=["number", "float", "int", "bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cols = {"cat": cat_cols, "num": num_cols}
    return X_train, X_test, y_train, y_test, cols


def model_selection_reg(
    X_train_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    reg_cols: Dict[str, List[str]],
    params_processing: dict,  
    models_cfg: dict,
):
   
    pre = _preprocessor(reg_cols["num"], reg_cols["cat"])

    scoring = models_cfg.get("scoring", "neg_mean_absolute_error")
    cv = int(models_cfg.get("cv", 5))
    n_jobs = int(models_cfg.get("n_jobs", -1))

    best_model = None
    best_score = -np.inf

    all_cv_rows = []

    for model_name, spec in models_cfg.items():
        if model_name in {"scoring", "cv", "n_jobs", "num_features", "cat_features"}:
            continue

        est = _estimator_by_name(spec.get("estimator"))
        pipe = SkPipeline([("prep", pre), ("reg", est)])

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
        gs.fit(X_train_reg, y_train_reg.values.ravel())

        
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df.insert(0, "model", model_name)
        all_cv_rows.append(cv_df)

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_

    reg_cv_results = pd.concat(all_cv_rows, ignore_index=True)
    return best_model, reg_cv_results


def predict_reg(best_regressor: SkPipeline, X_test_reg: pd.DataFrame) -> pd.DataFrame:
    
    y_pred = best_regressor.predict(X_test_reg)
    return pd.DataFrame({"y_pred_reg": y_pred})


def export_reg_leaderboard(reg_cv_results: pd.DataFrame) -> pd.DataFrame:

    needed = {"model", "rank_test_score", "mean_test_score", "std_test_score", "params"}
    missing = needed - set(reg_cv_results.columns)
    if missing:
        raise ValueError(f"Faltan columnas en reg_cv_results: {sorted(missing)}")

    best_per_model = (
        reg_cv_results
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
