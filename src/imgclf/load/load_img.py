import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import shutil


def load_imgs_from_dir():
    # Define parameters
    batch_size = 32
    img_size = (28, 28)
    data_dir = 'artifacts/datasets/mnist_split/'  # Replace with the path to your MNIST dataset directory

    # Create an ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        # shear_range=0.2,  # Shear transformation
        # zoom_range=0.2,  # Zoom transformation
        # horizontal_flip=True  # Horizontal flip
    )

    # Create the training dataset
    ds_train = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # 'sparse' for integer labels
        subset='training'
    )

    # Create the validation dataset
    ds_val = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # n_samples = ds_train.samples, ds_val.samples
    #
    # # Convert the generators to TensorFlow datasets
    # ds_train = tf.data.Dataset.from_generator(lambda: ds_train, output_signature=(tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int64)))
    # ds_val = tf.data.Dataset.from_generator(lambda: ds_val, output_signature=(tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int64)))
    #
    # # Optionally, prefetch data to improve performance
    # ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    # ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val#, n_samples


def save_split_imgs():
    output_folder = r"artifacts/datasets/mnist"
    output_folder_split = r"artifacts/datasets/mnist_split"

    # Create the temporary directory if it doesn't exist
    os.makedirs(output_folder_split, exist_ok=True)

    # Move images to subdirectories based on their labels
    move_imgs(output_folder, output_folder_split)


def move_imgs(from_dir, to_dir):
    for file in os.listdir(from_dir):
        if not file.endswith(".png"):
            continue

        label = int(file.split("_label_")[1].split(".")[0])
        label_folder = os.path.join(to_dir, str(label))
        os.makedirs(label_folder, exist_ok=True)
        shutil.move(os.path.join(from_dir, file), os.path.join(label_folder, file))
