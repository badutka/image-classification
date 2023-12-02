import tensorflow as tf
import keras

from imgclf.config.settings import settings


class NeuralNetwork():
    def __init__(self):
        pass

    def benchmark_1(self):
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 3)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        return model

    def comp_benchmark(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def conv_1(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(28, 28, 3)))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(10))

        return model

    def comp_conv_1(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def fit_conv_1(self, model, ds_train, ds_val, callbacks=None):
        cb_history = model.fit(
            ds_train,
            epochs=settings.models.conv_1.epochs,
            validation_data=ds_val,
            callbacks=callbacks
        )
        return cb_history

    def get_callbacks(self, name):

        match name:
            case 'conv_1':
                callbacks = [tf.keras.callbacks.TensorBoard(log_dir=settings.logs_dir)]
            case other:
                callbacks = []

        return callbacks
