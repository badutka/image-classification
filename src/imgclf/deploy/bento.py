from pathlib import Path

import bentoml

from imgclf.models.model_manager import load_model
from imgclf.config.settings import settings


def load_model_and_save_it_to_bento(path_to_file: str, file_name: str) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = load_model(path_to_file, file_name)
    bento_model = bentoml.keras.save_model("keras_model", model)
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(settings.model_path, settings.model_file)
