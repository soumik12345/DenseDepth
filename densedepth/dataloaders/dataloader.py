import os
from glob import glob
import tensorflow as tf
from typing import List
from sklearn.model_selection import train_test_split


class NYUDepthV2DataLoader:

    def __init__(
            self, data_dir: str, image_size: List[int],
            val_split: float, single_image_overfit: bool = False):
        self.train_rgb, self.train_depth = [], []
        self.val_rgb, self.val_depth = [], []
        assert 0 <= val_split <= 1.0
        self._populate_data_list(data_dir=data_dir, val_split=val_split)
        if single_image_overfit:
            self.train_rgb = [self.train_rgb[0]]
            self.train_depth = [self.train_depth[0]]
            self.val_rgb = [self.val_rgb[0]]
            self.val_depth = [self.val_depth[0]]
        self.rgb_size = image_size
        self.depth_size = [size // 2 for size in image_size]

    def _populate_data_list(self, data_dir: str, val_split: float):
        self.train_rgb = glob(str(os.path.join(data_dir, 'nyu2_train/*/*.jpg')))
        self.train_depth = [file_name.replace('jpg', 'png') for file_name in self.train_rgb]
        (
            self.train_rgb, self.val_rgb,
            self.train_depth, self.val_depth
        ) = train_test_split(
            self.train_rgb, self.train_depth,
            test_size=val_split, random_state=42
        )

    def summarize(self):
        print('Train RGB Images:', len(self.train_rgb))
        print('Train Depth Images:', len(self.train_depth))
        print('validation RGB Images:', len(self.val_rgb))
        print('Validation Depth Images:', len(self.val_depth))

    def _parse_rgb(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, self.rgb_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    def _parse_depth(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, self.depth_size)
        image = tf.image.convert_image_dtype(image / 255.0, dtype=tf.float32)
        image = 1000 / tf.clip_by_value(image * 1000, 10, 1000)
        return image

    def _map_function(self, rgb, depth):
        return self._parse_rgb(rgb), self._parse_depth(depth)

    def _configure_dataset(self, rgb_images: List[str], depth_images: List[str], batch_size: int):
        dataset = tf.data.Dataset.from_tensor_slices((rgb_images, depth_images))
        dataset = dataset.map(
            map_func=self._map_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(
            buffer_size=len(rgb_images),
            reshuffle_each_iteration=True
        )
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def get_datasets(self, batch_size: int):
        train_dataset = self._configure_dataset(
            rgb_images=self.train_rgb, depth_images=self.train_depth, batch_size=batch_size)
        val_dataset = self._configure_dataset(
            rgb_images=self.val_rgb, depth_images=self.val_depth, batch_size=batch_size)
        return train_dataset, val_dataset
