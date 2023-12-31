import pydantic
from pydantic import BaseModel, Field, model_validator, constr
from pathlib import Path
import typing
import os
from box import ConfigBox

from imgclf.common.utils import read_yaml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UnexpectedPropertyValidator(BaseModel):
    class Config:
        frozen = True
        protected_namespaces = ()

    @model_validator(mode='before')
    def check_unexpected_properties(cls, values):
        expected_properties = set(cls.__annotations__.keys())
        unexpected_properties = set(values) - expected_properties
        if unexpected_properties:
            raise ValueError(f"Unexpected properties: {', '.join(unexpected_properties)}")
        return values


class DataIngestionSettings(UnexpectedPropertyValidator):
    root_dir: Path


class Conv_1_Settings(UnexpectedPropertyValidator):
    epochs: int


class ModelsSettings(UnexpectedPropertyValidator):
    conv_1: Conv_1_Settings


class PlotsSettings(UnexpectedPropertyValidator):
    metrics_layout: dict
    metrics_xaxis: dict
    metrics_yaxis: dict


class Settings(UnexpectedPropertyValidator):
    artifacts_root: Path
    logs_dir: Path
    model_path: Path
    model_file: Path

    data_ingestion: DataIngestionSettings
    models: ModelsSettings
    plots: PlotsSettings


class SettingsManager(metaclass=Singleton):
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        file_path = Path(os.path.join(os.path.dirname(current_file_path), "settings.yaml"))
        self.settings = read_yaml(file_path)

    def initialize_settings(self) -> Settings:
        return Settings(**self.settings)


settings = SettingsManager().initialize_settings()
