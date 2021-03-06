from densedepth.dataloader import NYUDepthV2DataLoader


loader = NYUDepthV2DataLoader(
    data_dir='C:\\Workspace\\nyu_data',
    image_size=[480, 640], val_split=0.2
)
loader.summarize()
train_dataset, val_dataset = loader.get_datasets(batch_size=8)
print(train_dataset)
print(val_dataset)
