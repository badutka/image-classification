import shutil
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple

import tensorflow_datasets as tfds
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from imgclf.common.logger import logger


def run_image_saving(data_dir, data_dir_labeled):
    image_saver = ImageSaver()
    ds_train, ds_val = image_saver.load_dataset()
    tf.io.gfile.makedirs(data_dir)
    image_saver.save_imgs_by_label(ds_train, len(ds_train), data_dir)
    image_saver.save_imgs_by_label(ds_val, len(ds_val), data_dir)
    os.makedirs(data_dir_labeled, exist_ok=True)
    image_saver.move_imgs(data_dir, data_dir_labeled)


def run_ds_loading(data_dir_labeled, img_size, batch_size, validation_split_seed, shuffle_seed):
    dataset_loader = DatasetLoader()
    ds = dataset_loader.load_dataset(data_dir_labeled, batch_size, img_size, validation_split_seed)
    dataset_loader.dataset_info(ds)
    ds_train, ds_val, ds_test = dataset_loader.get_dataset_partitions(ds, batch_size=batch_size, shuffle_seed=shuffle_seed)
    ds_train, ds_val, ds_test = dataset_loader.standardize_data(ds_train, ds_val, ds_test)
    ds_train, ds_val, ds_test = dataset_loader.optimize_data(ds_train, ds_val, ds_test)
    return ds_train, ds_val, ds_test


def sample_random_mnist_data_point() -> Tuple[np.ndarray, np.ndarray]:
    # Path to the main directory containing subdirectories for each label
    main_dir = 'artifacts/datasets/mnist_split'

    # Get a list of subdirectories (assuming each subdirectory represents a label)
    label_dirs = [os.path.join(main_dir, label) for label in os.listdir(main_dir)]

    # Randomly select a label
    random_label_dir = random.choice(label_dirs)

    # # Get a list of image files in the selected label directory
    image_files = [os.path.join(random_label_dir, img) for img in os.listdir(random_label_dir) if img.endswith('.png')]

    # Randomly select an image file
    random_image_file = random.choice(image_files)

    # Load the image using TensorFlow
    image = load_img(random_image_file, target_size=(28, 28, 3))  # Adjust target_size as needed
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Display or process the randomly selected label and image
    logger.info(("Random Label:", os.path.basename(random_label_dir)))
    logger.info(("Random Image:", os.path.basename(random_image_file)))

    return image_array, os.path.basename(random_label_dir)


class DatasetLoader():
    def optimize_data(self, ds_train, ds_val, ds_test=None):
        ds_train = ds_train.cache()
        ds_val = ds_val.cache()
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        if ds_test is None:
            return ds_train, ds_val

        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_val, ds_test

    def standardize_data(self, ds_train, ds_val, ds_test=None):
        datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        ds_train = ds_train.map(lambda x, y: (datagen.standardize(x), y))
        ds_val = ds_val.map(lambda x, y: (datagen.standardize(x), y))

        if ds_test is None:
            return ds_train, ds_val

        ds_test = ds_test.map(lambda x, y: (datagen.standardize(x), y))
        return ds_train, ds_val, ds_test

    def get_dataset_partitions(self, dataset, ds_size=None, batch_size=128, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_seed=42):

        if train_split + test_split + val_split != 1:
            raise ValueError(f'Split partitions do not add up to 1.')

        if ds_size == None:
            ds_size = len(dataset)

        if shuffle:
            dataset = dataset.shuffle(ds_size * batch_size, seed=shuffle_seed)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        test_size = int(test_split * ds_size)

        ds_train = dataset.take(train_size)
        ds_val = dataset.skip(train_size).take(val_size)
        ds_test = dataset.skip(train_size + val_size).take(test_size)

        return ds_train, ds_val, ds_test

    def load_dataset(self, data_dir_labeled, batch_size, img_size, validation_split_seed):
        dataset = keras.utils.image_dataset_from_directory(
            directory=data_dir_labeled,
            batch_size=batch_size,
            image_size=img_size,
            seed=validation_split_seed,
            # color_mode='grayscale',
        )

        return dataset

    def load_dataset_split(self, data_dir_labeled, batch_size, img_size, validation_split_seed):
        ds_train, ds_val = keras.utils.image_dataset_from_directory(
            directory=data_dir_labeled,
            batch_size=batch_size,
            image_size=img_size,
            validation_split=0.2,
            subset='both',
            seed=validation_split_seed,
            # color_mode='grayscale',
        )

        return ds_train, ds_val

    def dataset_info(self, dataset):
        # ds_train is _PrefetchDataset, batch is tf.Tensor
        batch = next(iter(dataset.take(1)))
        print(f'shape of a single batch (batch size, height, width, no. channels): {batch[0].shape}')
        print(f'max value: {batch[0].numpy().max()}, min value: {batch[0].numpy().min()}')

        fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img)
            ax[idx].title.set_text(batch[1][idx].numpy())


class ImageSaver():
    def __init__(self):
        pass

    def load_dataset(self):
        (ds_train, ds_val), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        return ds_train, ds_val

    def save_imgs_by_label(self, dataset, num_images, output_folder):
        for i, (image, label) in enumerate(dataset.take(num_images)):
            # image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            image = tf.squeeze(image)
            image = Image.fromarray(image.numpy())
            image.save(f"{output_folder}/image_{i + 1}_label_{label.numpy()}.png")

    def move_imgs(self, from_dir, to_dir):
        for file in os.listdir(from_dir):
            if not file.endswith(".png"):
                continue

            label = int(file.split("_label_")[1].split(".")[0])
            label_folder = os.path.join(to_dir, str(label))
            os.makedirs(label_folder, exist_ok=True)
            shutil.move(os.path.join(from_dir, file), os.path.join(label_folder, file))
