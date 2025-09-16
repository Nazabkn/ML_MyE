from kedro.pipeline import Pipeline, node
from .nodes import clean_neo, clean_meteorites

def create_pipeline() -> Pipeline:
    return Pipeline([
        node(clean_neo, inputs="neo", outputs="neo_clean", name="clean_neo"),
        node(clean_meteorites, inputs="meteorite_landings", outputs="meteorites_clean", name="clean_meteorites"),
    ])
