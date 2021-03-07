import os
from typing import List
import tensorflow as tf
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from .commons import create_example, split_list


class TFRecordCreator:

    def __init__(
            self, output_directory: str, dataset_name: str,
            log_interval: int = 100, shard_size: int = 128, is_notebook: bool = False):
        self.output_directory = output_directory
        self.dataset_name = dataset_name
        self.log_interval = log_interval
        self.shard_size = shard_size
        self.is_notebook = is_notebook

    def _create_tfrecord_files(self, data_shards: List[List[str]], split: str):
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        progress_bar = tqdm_notebook if self.is_notebook else tqdm
        for shard, data_shard in enumerate(progress_bar(data_shards)):
            shard_size = len(data_shard)
            record_name = self.dataset_name + '-{}-{:02d}-{}.tfrec'.format(split, shard, shard_size)
            file_path = os.path.join(self.output_directory, record_name)
            with tf.io.TFRecordWriter(file_path) as out_file:
                for data in data_shard:
                    example = create_example(data)
                    out_file.write(example.SerializeToString())
                if shard % self.log_interval == 0:
                    print('Data written at file {} containing {} records'.format(file_path, shard_size))

    def create(self, image_files: List[str], split: str):
        image_shards = split_list(image_files, chunk_size=self.shard_size)
        print('Creating {} TFRecords....'.format(split))
        self._create_tfrecord_files(image_shards, split=split)
