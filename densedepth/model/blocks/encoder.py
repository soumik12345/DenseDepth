import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.backbone = tf.keras.applications.DenseNet169(
            input_shape=[None, None, 3], include_top=False, weights='imagenet'
        )
        backbone_layers = [
            'pool1', 'pool2_pool', 'pool3_pool', 'conv1/relu'
        ]
        outputs = [self.backbone.outputs[-1]]
        for layer in backbone_layers:
            outputs.append(self.backbone.get_layer(layer).output)
        self.encoder = tf.keras.Model(
            inputs=self.base_model.inputs, outputs=outputs
        )

    def call(self, inputs, training=None, mask=None):
        return self.encoder(inputs)
