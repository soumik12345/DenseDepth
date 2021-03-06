import tensorflow as tf

from .upsample import UpSampleBlock


class Decoder(tf.keras.Model):

    def __init__(self, filters: int, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.convolution_1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, padding='same'
        )
        self.upsample_1 = UpSampleBlock(filters=filters // 2)
        self.upsample_2 = UpSampleBlock(filters=filters // 4)
        self.upsample_3 = UpSampleBlock(filters=filters // 8)
        self.upsample_4 = UpSampleBlock(filters=filters // 16)
        self.convolution_2 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, strides=1, padding='same'
        )

    def call(self, inputs, training=None, mask=None):
        x, pool_1, pool_2, pool_3, conv = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        x = self.convolution_1(x)
        x = self.upsample_1([x, pool_3])
        x = self.upsample_2([x, pool_2])
        x = self.upsample_3([x, pool_1])
        x = self.upsample_4([x, conv])
        x = self.convolution_2(x)
        return x
