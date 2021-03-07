from typing import List
import tensorflow as tf


class NYUTFRecordLoader:

    def __init__(self, train_file_pattern: str, val_file_pattern: str, image_size: List[int]):
        self.train_file_pattern = train_file_pattern
        self.val_file_pattern = val_file_pattern
        self.rgb_size = image_size
        self.depth_size = [size // 2 for size in image_size]

    def _preprocess_rgb(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.rgb_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    def _preprocess_depth(self, image):
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, self.depth_size)
        image = tf.image.convert_image_dtype(image / 255.0, dtype=tf.float32)
        image = 1000 / tf.clip_by_value(image * 1000, 10, 1000)
        return image

    @tf.function
    def parse_example(self, example):
        tfrec_format = {
            'rgb_image': tf.io.FixedLenFeature([], tf.string),
            'depth_image': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, tfrec_format)
        rgb_image = self._preprocess_rgb(example.pop('rgb_image'))
        depth_image = self._preprocess_depth(example.pop('depth_image'))
        return rgb_image, depth_image

    def _configure_datasets(self, file_paths, batch_size: int):
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(
            self.parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def get_datasets(self, batch_size: int):
        train_dataset = self._configure_datasets(
            tf.io.gfile.glob(self.train_file_pattern),
            batch_size=batch_size
        )
        val_dataset = self._configure_datasets(
            tf.io.gfile.glob(self.val_file_pattern),
            batch_size=batch_size
        )
        return train_dataset, val_dataset
