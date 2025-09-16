from kedro.pipeline import Pipeline, node
from .nodes import aggregate_meteorites_by_year, unify_by_year

def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            aggregate_meteorites_by_year,
            inputs="meteorites_clean",
            outputs="meteorites_by_year",
            name="aggregate_meteorites_by_year",
        ),
        node(
            unify_by_year,
            inputs=["neo_clean", "meteorites_by_year"],
            outputs="model_input_table",
            name="unify_by_year",
        ),
    ])
