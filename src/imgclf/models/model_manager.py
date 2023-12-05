import keras
import os

from imgclf.common.logger import logger


def save_model(model, path_to_file: str, file_name: str):
    file = os.path.join(path_to_file, file_name)
    model.save(file)
    logger.info(f'Successfully saved model file: {file}.')


def load_model(path_to_file: str, file_name: str):
    file = os.path.join(path_to_file, file_name)
    model = keras.models.load_model(file)
    logger.info(f'Successfully loaded model file: {file}.')
    return model
