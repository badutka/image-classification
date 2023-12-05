import os
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from imgclf.config.settings import settings
from imgclf.common.logger import logger
from imgclf.data import load_data
from imgclf.models.networks import NeuralNetwork
from imgclf.models import model_manager


def main():
    input_shape = (28, 28, 3)
    num_classes = 10
    batch_size = 128
    img_size = (28, 28)
    validation_split_seed = 42
    shuffle_seed = 42
    data_dir = r"artifacts/datasets/mnist"
    data_dir_labeled = r"artifacts/datasets/mnist_split"

    ds_train, ds_val, ds_test = load_data.run_ds_loading(data_dir_labeled, img_size, batch_size, validation_split_seed, shuffle_seed)

    neural_network = NeuralNetwork()
    model = neural_network.conv_net_1(input_shape=input_shape, num_classes=num_classes)
    callbacks = neural_network.get_callbacks('default')
    cb_history = neural_network.train_model(model=model, ds_train=ds_train, ds_val=ds_val, num_epochs=settings.models.conv_1.epochs, callbacks=callbacks)
    model_manager.save_model(model, settings.model_path, settings.model_file)


if __name__ == "__main__":
    main()
