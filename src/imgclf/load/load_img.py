import os
import tensorflow as tf
import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import to_categorical
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import shutil
import matplotlib.pyplot as plt


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

    return ds_train, ds_val  # , n_samples


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


def save_imgs():
    # Load MNIST dataset
    (ds_train, ds_val), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Define a function to save images
    def save_images(dataset, num_images, output_folder):
        for i, (image, label) in enumerate(dataset.take(num_images)):
            # image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            image = tf.squeeze(image)
            image = Image.fromarray(image.numpy())
            image.save(f"{output_folder}/image_{i + 1}_label_{label.numpy()}.png")

    # Specify the output folder
    output_folder = r"artifacts/datasets/mnist"

    # Create the output folder if it doesn't exist
    tf.io.gfile.makedirs(output_folder)

    # # Save images from the training dataset
    save_images(ds_train, 60000, output_folder)

    # # Save images from the test dataset
    save_images(ds_val, 10000, output_folder)


def dataset_info(ds_train):
    batch = iter(ds_train).next()
    # batch = ds_train.take(1)
    print(f'shape of a single batch (batch size, height, width, no. channels): {batch[0].shape}')
    print(f'max value: {batch[0].max()}, min value: {batch[0].min()}')

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
