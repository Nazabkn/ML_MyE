from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)


def _to_series(y) -> pd.Series:
  
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]
    if isinstance(y, np.ndarray):
        return pd.Series(y)
    return y

def _sigmoid(x: np.ndarray) -> np.ndarray:
 
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def _predict_proba_safe(model, X_test: pd.DataFrame) -> np.ndarray:
 
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]
        else:
      
            y_proba = proba.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if isinstance(scores, (list, tuple)):
            scores = np.asarray(scores)
        y_proba = _sigmoid(scores.astype(float))
    else:
      
        y_pred = model.predict(X_test)
        y_proba = (pd.Series(y_pred).astype(int).values).astype(float)

    y_proba = np.asarray(y_proba, dtype=float)

    y_proba = np.nan_to_num(y_proba, nan=0.0, neginf=0.0, posinf=1.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)
    return y_proba


def evaluate_model(
    hazardous_clf,                
    X_test: pd.DataFrame,
    y_test,                       
    threshold: float = 0.5,        
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    
    y_true = _to_series(y_test).astype(int)

 
    y_proba = _predict_proba_safe(hazardous_clf, X_test)
    
    y_pred = (y_proba >= float(threshold)).astype(int)

    
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
    }
    final_eval_metrics = pd.DataFrame([metrics])

    final_predictions = pd.DataFrame(
        {"y_true": y_true.values, "y_pred": y_pred, "y_proba": y_proba}
    )


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    final_confusion = pd.DataFrame(
        cm,
        index=pd.Index([0, 1], name="true"),
        columns=pd.Index([0, 1], name="pred"),
    )


    fpr, tpr, thr = roc_curve(y_true, y_proba)
    final_roc_curve = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})

    return final_eval_metrics, final_predictions, final_confusion, final_roc_curve


def export_classification_report(
    final_predictions: pd.DataFrame
) -> pd.DataFrame:
   
    y_true = pd.Series(final_predictions["y_true"]).astype(int).values
    y_pred = pd.Series(final_predictions["y_pred"]).astype(int).values
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    rows = []
    for label, stats in rep.items():
        if isinstance(stats, dict):
            row = {"label": label}
            row.update({k: float(v) for k, v in stats.items()})
            rows.append(row)
    return pd.DataFrame(rows)
