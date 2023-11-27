import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import os

batch_size = 128
img_size = (28, 28)
validation_split_seed = 42
shuffle_seed = 42
data_dir = r"artifacts/datasets/mnist"
data_dir_labeled = r"artifacts/datasets/mnist_split"


def run_image_saving():
    image_saver = ImageSaver()
    ds_train, ds_val = image_saver.load_dataset()
    tf.io.gfile.makedirs(data_dir)
    image_saver.save_imgs_by_label(ds_train, 60000, data_dir)
    image_saver.save_imgs_by_label(ds_train, 10000, data_dir)
    os.makedirs(data_dir_labeled, exist_ok=True)
    image_saver.move_imgs(data_dir, data_dir_labeled)


def run_ds_loading():
    dataset_loader = DatasetLoader()
    ds = dataset_loader.load_dataset()
    dataset_loader.dataset_info(ds)
    ds_train, ds_val, ds_test = dataset_loader.get_dataset_partitions(ds)
    ds_train, ds_val, ds_test = dataset_loader.standardize_data(ds_train, ds_val, ds_test)
    ds_train, ds_val, ds_test = dataset_loader.optimize_data(ds_train, ds_val, ds_test)
    return ds_train, ds_val, ds_test


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

    def get_dataset_partitions(self, dataset, ds_size=None, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
        if train_split + test_split + val_split != 1:
            raise ValueError(f'Split portions do not add up to 1.')

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

    def load_dataset(self):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir_labeled,
            batch_size=batch_size,
            image_size=img_size,
            seed=validation_split_seed,
            # color_mode='grayscale',
        )

        return dataset

    def load_dataset_split(self):
        ds_train, ds_val = tf.keras.utils.image_dataset_from_directory(
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
