import tensorflow as tf


class DenseDepthLoss(tf.keras.losses.Loss):

    def __init__(self, lambda_weight: float, depth_max_val: float, **kwargs):
        super(DenseDepthLoss, self).__init__(**kwargs)
        self.lambda_weight = lambda_weight
        self.depth_max_val = depth_max_val

    def call(self, y_true, y_pred):
        depth_loss = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
        ssim_loss = tf.keras.backend.clip(
            (1 - tf.image.ssim(
                y_true, y_pred, self.depth_max_val
            )) * 0.5, 0, 1
        )
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        gradient_loss = tf.reduce_mean(
            tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1
        )
        return self.lambda_weight * depth_loss + gradient_loss + ssim_loss
