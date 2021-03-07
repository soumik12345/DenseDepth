from typing import List
import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def create_example(rgb_path, depth_path):
    rgb_image = tf.io.read_file(rgb_path)
    rgb_image = tf.image.decode_image(rgb_image, channels=3)
    depth_image = tf.io.read_file(depth_path)
    depth_image = tf.image.decode_jpeg(depth_image)
    return tf.train.Example(
        features=tf.train.Features(feature={
            'rgb_image': bytes_feature(rgb_image),
            'depth_image': bytes_feature(depth_image)
        })
    )


def split_list(given_list: List, chunk_size: int):
    return [
        given_list[offs: offs + chunk_size]
        for offs in range(
            0, len(given_list), chunk_size
        )
    ]
