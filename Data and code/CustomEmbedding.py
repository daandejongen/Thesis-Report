# Master Thesis Daan de Jong 
# Negation scope detection with Neural Networks

# Custom layer for the creation of cue embeddings

import tensorflow as tf
from tensorflow import keras
    

class CueEmbeddingLayer(keras.layers.Layer):
    def __init__(self, embDim, mask_value):
        super(CueEmbeddingLayer, self).__init__()
        self.embDim = embDim
        self.mask_value = mask_value
        
    def build(self, embDim):
        self.w = self.add_weight(shape=self.embDim, 
                                 initializer=keras.initializers.Ones(),
                                 trainable=False)

    def call(self, inputs):
        return tf.tensordot(inputs, self.w, axes=0)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.not_equal(inputs, self.mask_value)