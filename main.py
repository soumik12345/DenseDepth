import tensorflow as tf
from datetime import datetime
from wandb.keras import WandbCallback

from densedepth import (
    NYUDepthV2DataLoader,
    DenseDepth, DenseDepthLoss
)
from densedepth.utils import init_wandb


experiment_name = 'NYU_Depth_V2_single_image_overfit'

# init_wandb(
#     project_name='densedepth', entity='19soumik-rakshit96',
#     experiment_name=experiment_name, wandb_api_key='cf0947ccde62903d4df0742a58b8a54ca4c11673'
# )

loader = NYUDepthV2DataLoader(
    data_dir='C:\\Workspace\\nyu_data',
    image_size=[480, 640], val_split=0.2, single_image_overfit=True
)
loader.summarize()
train_dataset, val_dataset = loader.get_datasets(batch_size=4)
print(train_dataset)
print(val_dataset)

model = DenseDepth()
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4, amsgrad=True
    ), loss=DenseDepthLoss(
        lambda_weight=0.1, depth_max_val=1000.0 / 10.0
    )
)

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs/train/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
        histogram_freq=1, update_freq=50, write_images=True
    ),
    # WandbCallback(),
    tf.keras.callbacks.ModelCheckpoint(
        './logs/train/' + experiment_name + '_{epoch}.ckpt', save_weights_only=True
    )
]

history = model.fit(
    train_dataset, validation_data=val_dataset,
    epochs=15, callbacks=callbacks
)
