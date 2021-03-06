import tensorflow as tf

from .blocks import Encoder, Decoder


class DenseDepth(tf.keras.Model):

    def __init__(self, **kwargs):
        super(DenseDepth, self).__init__(**kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(
            filters=int(self.encoder.layers[-1].output[0].shape[-1] // 2)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x
