import tensorflow as tf
import keras

from typing import List, Tuple

from imgclf.config.settings import settings
from imgclf.common.logger import logger


class NeuralNetwork():
    def __init__(self):
        pass

    def benchmark_1(self):
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 3)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def conv_net_1(self, input_shape: Tuple, num_classes: int):
        """
        Builds a simple ConvNet model.

        Args:
            input_shape: The shape of the input data.
            num_classes: The number of classes to classify.

        Returns:
            A Keras model.
        """
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        # model.add(keras.layers.Dropout(0.5)),
        model.add(keras.layers.Dense(num_classes))  # , activation="softmax")

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        # model.compile(
        #     optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        # )

        return model

    def train_model(self,
                    model: keras.Model,
                    ds_train: tf.data.Dataset,
                    ds_val: tf.data.Dataset,
                    num_epochs: int,
                    callbacks: List[keras.src.callbacks.TensorBoard]
                    ) -> keras.Model:
        """
        Trains a model.

        Args:
            model: The model to train.
            ds_train: The training dataset.
            ds_val: The validation dataset.
            num_epochs: The number of epochs to train for.
            callbacks: A list of applicable tensorflow callbacks.

        Returns:
            The trained model.
        """
        model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_val,
            callbacks=callbacks
        )

        return model

    def get_callbacks(self, name):
        """
        Retrieves a list of callbacks by case name.

        Args:
            name: case name to retrieve specific callbacks list.

        Returns:
            List of tensorflow callbacks.
        """

        match name:
            case 'default':
                callbacks = [keras.callbacks.TensorBoard(log_dir=settings.logs_dir)]
            case other:
                callbacks = []

        return callbacks
