import os
import gdown
import wandb
import zipfile
import subprocess
from typing import List
import tensorflow as tf
import tensorflow_addons as tfa

from densedepth import DenseDepth, DenseDepthLoss


class Trainer:

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.train_dataset, self.val_dataset = None, None
        self.model = None

    @staticmethod
    def download_dataset(dataset_name: str, dataset_access_key: str):
        gdown.download(
            'https://drive.google.com/uc?id={}'.format(dataset_access_key),
            '{}.zip'.format(dataset_name), quiet=False
        )
        os.system('mkdir -p data')
        with zipfile.ZipFile('{}.zip'.format(dataset_name), 'r') as zip_ref:
            zip_ref.extractall('data')
        subprocess.run(['rm', '{}.zip'.format(dataset_name)])

    def init_wandb(self, project_name: str, entity: str, wandb_api_key: str):
        if project_name is not None and self.experiment_name is not None:
            os.environ['WANDB_API_KEY'] = wandb_api_key
            wandb.init(
                project=project_name, entity=entity,
                name=self.experiment_name, sync_tensorboard=True
            )

    def compile(
            self, train_dataset, val_dataset, model_build_shape: List[int],
            learning_rate: float, weight_decay: float, strategy=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if strategy is not None:
            with strategy.scope():
                self.model = DenseDepth()
                self.model.compile(
                    optimizer=tfa.optimizers.AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay
                    ), loss=DenseDepthLoss(
                        lambda_weight=0.1, depth_max_val=1000.0 / 10.0
                    )
                )
        else:
            self.model = DenseDepth()
            self.model.build(model_build_shape)
            self.model.compile(
                optimizer=tfa.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay
                ), loss=DenseDepthLoss(
                    lambda_weight=0.1, depth_max_val=1000.0 / 10.0
                )
            )

    def train(self, epochs: int, log_dir: str):
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1,
                update_freq=50, write_images=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                './logs/train/' + self.experiment_name + '_{epoch}.ckpt',
                save_weights_only=True
            )
        ]
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            epochs=epochs, callbacks=callbacks
        )
        return history
