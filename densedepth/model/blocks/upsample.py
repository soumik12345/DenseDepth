import tensorflow as tf


class UpSampleBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat = tf.keras.layers.Concatenate()
        self.convolution_1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, strides=1, padding='same'
        )
        self.convolution_2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, strides=1, padding='same'
        )

    def call(self, inputs, **kwargs):
        x = self.upsample(inputs[0])
        x = self.concat([x, inputs[1]])
        x = self.convolution_1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.convolution_2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x
