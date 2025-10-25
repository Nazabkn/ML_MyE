from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model, export_classification_report

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
    ])
