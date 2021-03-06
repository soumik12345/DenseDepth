import os
import wandb
import gdown


def init_wandb(
        project_name: str, experiment_name: str,
        entity: str, wandb_api_key: str):
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(
            project=project_name, entity=entity,
            name=experiment_name, sync_tensorboard=True
        )


def download_dataset(dataset_name: str, dataset_access_key: str):
    gdown.download(
        'https://drive.google.com/uc?id={}'.format(dataset_access_key),
        '{}.zip'.format(dataset_name), quiet=False
    )
