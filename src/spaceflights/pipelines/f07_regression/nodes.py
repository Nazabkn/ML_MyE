from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
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
    if name == "knn":
        return KNeighborsRegressor()
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


def evaluate_regression(
    y_test_reg: pd.DataFrame,
    y_pred_reg: pd.DataFrame,
) -> pd.DataFrame:

    y_true = y_test_reg.squeeze().astype(float)
    y_pred = y_pred_reg.squeeze().astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    metrics = pd.DataFrame([
        {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "support": int(len(y_true)),
        }
    ])
    return metrics


def plot_regression_cv(
    reg_results_table: pd.DataFrame,
    scoring: str = "neg_mean_absolute_error",
) -> plt.Figure:

    if reg_results_table.empty:
        raise ValueError("No hay resultados de regresión para graficar.")

    order = reg_results_table.sort_values("cv_mean_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        order["model"],
        order["cv_mean_score"],
        xerr=order["cv_std"],
        color="#5AD8A6",
        alpha=0.8,
        ecolor="#1f1f1f",
        capsize=4,
    )
    ax.set_xlabel(f"{scoring} (mean ± std)")
    ax.set_ylabel("Modelo")
    ax.set_title("Comparativa de modelos de regresión")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_regression_predictions(
    y_test_reg: pd.DataFrame,
    y_pred_reg: pd.DataFrame,
) -> plt.Figure:

    y_true = y_test_reg.squeeze().astype(float)
    y_pred = y_pred_reg.squeeze().astype(float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#5B8FF9")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#333333")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Predicción")
    ax.set_title("Predicción vs. valor real")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig