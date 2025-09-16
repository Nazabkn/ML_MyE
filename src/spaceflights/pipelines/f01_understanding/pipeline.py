from kedro.pipeline import Pipeline, node
from .nodes import summarize_dataframe

def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=summarize_dataframe,
            inputs=dict(df="neo", name="params:labels.neo"),
            outputs="report_neo_summary",
            name="summarize_neo",
        ),
        node(
            func=summarize_dataframe,
            inputs=dict(df="meteorite_landings", name="params:labels.meteorites"),
            outputs="report_meteorites_summary",
            name="summarize_meteorites",
        ),
    ])
