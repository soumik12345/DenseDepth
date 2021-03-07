from densedepth import Trainer, NYUDepthV2DataLoader


experiment_name = 'NYU_Depth_V2_single_image_overfit'
trainer = Trainer(experiment_name=experiment_name)
trainer.download_dataset(
    dataset_name='nyu_data',
    dataset_access_key='1nQd4hcsQVnX33vTv4dF2'
)
trainer.init_wandb(
    project_name='densedepth', entity='19soumik-rakshit96',
    wandb_api_key='cf0947ccde62903d4df0742a58b8a54ca4c11673'
)
loader = NYUDepthV2DataLoader(
    data_dir='C:\\Workspace\\nyu_data',
    image_size=[480, 640], val_split=0.2, single_image_overfit=True
)
loader.summarize()
train_dataset, val_dataset = loader.get_datasets(batch_size=1)
print(train_dataset)
print(val_dataset)
trainer.compile(
    train_dataset=train_dataset, val_dataset=val_dataset,
    model_build_shape=[480, 640, 3], learning_rate=1e-4
)
trainer.train(epochs=50)
