import os
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from functools import reduce

from imgclf.common.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox.

    :param path_to_yaml: Path to the yaml file.
    :raises ValueError: if yaml file is empty.
    :returns ConfigBox: returns object with nested properties instead of a dictionary.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Settings file [{path_to_yaml}] loaded successfully.\n{30 * '*'}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Settings file [{path_to_yaml}] is empty.")
    except Exception as e:
        raise e
