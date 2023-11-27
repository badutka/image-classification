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
    batch_size = 32
    img_size = (28, 28)
    data_dir = 'artifacts/datasets/mnist_split/'

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    ds_train = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # 'sparse' for integer labels
        subset='training'
    )

    ds_val = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    return ds_train, ds_val


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
    batch = next(iter(ds_train))
    # batch = ds_train.take(1)
    print(f'shape of a single batch (batch size, height, width, no. channels): {batch[0].shape}')
    print(f'max value: {batch[0].max()}, min value: {batch[0].min()}')

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
