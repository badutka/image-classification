import os
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from imgclf.config.settings import settings
from imgclf.data import load_data
from imgclf.models.networks import NeuralNetwork


def main():
    ds_train, ds_val, ds_test = load_data.run_ds_loading()
    neural_network = NeuralNetwork()
    model = neural_network.conv_net_1(input_shape=(28, 28, 3), num_classes=10)
    callbacks = neural_network.get_callbacks('default')
    cb_history = neural_network.train_model(model=model, ds_train=ds_train, ds_val=ds_val, num_epochs=settings.models.conv_1.epochs, callbacks=callbacks)


if __name__ == "__main__":
    main()
