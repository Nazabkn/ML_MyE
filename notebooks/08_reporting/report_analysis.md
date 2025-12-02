# Discusión de resultados de clasificación y regresión

Este documento complementa el cuaderno '00_report.ipynb' con una lectura más detallada de los experimentos ejecutados mediante los pipelines de Kedro.

## Clasificación

- **Selección de modelos**: Sse comparan al menos cinco algoritmos (Regresión Logística, KNN, Árbol de Decisión, Random Forest, SVC y Gradient Boosting). El nodo de selección usa 'GridSearchCV' con la metrica objetivo definida en 'conf/base/parameters_models.yml' (por defecto 'f1' para balancear precisión y recall en clases desbalanceadas).
- **Cómo identificar el mejor**: el leaderboard 'data/08_reporting/clf_results_table.csv' ordena los modelos por 'cv_mean_score' (media del 'f1' en validación cruzada) y reporta la desviación estándar. El primer renglon corresponde al modelo con mejor rendimiento medio y menor rango.
- **Interpretación**: un 'cv_mean_score' alto indica un modelo con buen equilibrio entre falsos positivos y falsos negativos. Para consolidar el resultado, el pipeline de evaluación calcula métricas de 'f1', 'precision', 'recall' y 'roc_auc' sobre el conjunto de prueba y genera la matriz de confusión y la curva ROC (guardadas en 'data/08_reporting/final_confusion_matrix.csv' y 'data/08_reporting/final_roc_curve.csv').

## Regresión

- **Selección de modelos**: se comparan regresores lineales y de ensamble (p. ej. Ridge, Lasso, Random Forest, SVR, KNN) con 'GridSearchCV'. La métrica principal es 'neg_mean_absolute_error', pero ahora el leaderboard expone también **MAE**, **RMSE** y **R²** para el mejor modelo.
- **Métricas explícitas**: 
Tras entrenar, el nodo 'evaluate_regression' calcula MAE, RMSE y R² sobre el conjunto de prueba y guarda los resultados en 'data/08_reporting/reg_eval_metrics.csv'. Estas métricas se incorporan al leaderboard 'data/08_reporting/reg_results_table.csv' en las columna 'mae', 'rmse', 'r2' y 'support', haciendo evidente el rendimiento del modelo ganador.
- **Cómo interpretar el mejor modelo**: el modelo con menor MAE/ RMSE y mayor R² (preferentemente cercano a 1) será el de mejor ajuste. La dispersión entre 'cv_mean_score' y las métricas sobre prueba permite validar si hay sobreajuste.
- **Visualizaciones**: el pipeline genera 'reg_results_plot.png' (comparativa CV) y 'reg_predictions_plot.png' (predicciones vs. valores reales) para contrastar error sistemático o sesgos.

## Recomendaciones

1. Ejecutar 'kedro run '--pipeline=f05_classifica' despues de sincronizar los datos con 'dvc pull'
2. revisar los CSV de resultados en 'data/08_reporting/' para documentar explicitamente que modelo quedó en prmer lugar y anexar los valores de métricas en el reporte final.
3. Si las metricas entre entrenamiento y prueba divergen, tambien ajustar la complejidad del modelo (profundidad de árboles, regularización) o reevaluar la selección de variables.