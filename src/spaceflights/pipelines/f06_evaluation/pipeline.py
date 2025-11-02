from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_model,
    export_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=[
                "hazardous_clf",
                "X_test",
                "y_test",
                "params:parameters_data_processing.threshold",  
            ],
            outputs=[
                "final_eval_metrics",
                "final_predictions",
                "final_confusion",
                "final_roc_curve",
            ],
            name="evaluate_model_node",
        ),
        node(
            func=export_classification_report,
            inputs="final_predictions",
            outputs="final_classification_report",
            name="export_classification_report_node",
        ),
        node(
            func=plot_confusion_matrix,
            inputs="final_confusion",
            outputs="confusion_matrix_plot",
            name="plot_confusion_matrix_node",
        ),
        
        node(
            func=plot_roc_curve,
            inputs="final_roc_curve",
            outputs="roc_curve_plot",
            name="plot_roc_curve_node",
        ),
    ])
