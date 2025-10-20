from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_data,
            inputs=["model_input_table", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train", "params:model_options", "params:model_options"],
            outputs="classifier",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["classifier", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node",
        ),
    ])
