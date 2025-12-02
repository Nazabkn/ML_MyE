# Pipeline de Aprendizaje No Supervisado

Este módulo agrupa las tareas de exploración sin etiquetas para complementar el modelado supervisado.

## Clustering
- **Algoritmos**: K-Means, DBSCAN y Gaussian Mixture Models.
- **Salidas**:
  - 'unsupervised_cluster_labels': etiquetas por algoritmo y 'cluster_label' principal.
  - 'unsupervised_cluster_metrics': Silhouette, Calinski-Harabasz y Davies-Bouldin
  - 'unsupervised_cluster_plots': gráficos 2D (PCA) de cada método.
  - 'unsupervised_model_input_with_clusters': tabla original con la feature 'cluster_label' añadida.

## Reducción de dimensionalidad
- **PCA**: componentes principales, varianza explicada y biplot de cargas
- **UMAP**: embedding 2D estable para visualización
- **Artefactos**: 'pca_components', 'pca_variance', 'pca_loadings', 'pca_variance_plot', 'pca_biplot', 'umap_embedding', 'umap_plot'.

## Integración con supervisado
La feature 'cluster_label' se usa en el pipeline de características (f04_feature) para enriquecer los datasets de clasificación y regresión antes de entrenar modelos

## Próximos pasos
- Implementar detección de anomalías con 'pyod' y reglas de asociación con 'mlxtend'.
- Documentar metricas adicionales y experimentos en los notebooks '05_unsupervised_learning.ipynb' y '06_final_analysis.ipynb'.