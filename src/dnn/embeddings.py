# coding: utf-8


"""
Embeddings

Based on https://github.com/CyberZHG/keras-embed-sim
Framework was changed from keras to tf.keras to compile TPU model using keras-support.

EmbeddingWithWeights, which returns normal embedings with base weights, was
widely rewritten according to TPUEmbedding in tf.keras.

A flag to stop mask propagation is added into EmbeddingSim to escape 
loss calculation related to sample weights and masks. 
(Currently keras masks don't seem to be enabled on TPU)
"""


import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


__all__ = ['EmbeddingWithWeights', 'EmbeddingSim', 'get_custom_objects']


def get_custom_objects():
    return {
        'EmbeddingWithWeights': EmbeddingWithWeights,
        'EmbeddingSim': EmbeddingSim,
    }


class EmbeddingWithWeights(tf.keras.layers.Embedding):
    """
    Inputs: [Input:(batch_size, seq_len)]
    Outputs: [(batch_size, seq_len, feat_dim), embeddings:(token_num, embed_dim)]
    """
    
    def get_config(self):
        config = {}
        config.update(super(EmbeddingWithWeights, self).get_config())
        return config
    
    def compute_output_shape(self, input_shape):
        return [
            super(EmbeddingWithWeights, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim),
        ]
    
    def compute_mask(self, inputs, mask=None):
        return [mask, None]

    def call(self, inputs):
        
        # TPU compatible look-up
        if K.dtype(inputs) != 'int32':
            inputs = math_ops.cast(inputs, 'int32')
        inputs = array_ops.one_hot(inputs, self.input_dim)
        ret = math_ops.tensordot(inputs, self.embeddings, 1)
        
        # Embeddings are multiplied by sqrt(d_model) while weights are not.
        ret *= K.sqrt(tf.cast(K.shape(self.embeddings)[-1], tf.float32))
        
        return [
                ret,
                tf.cast(self.embeddings, tf.float32),
                # Note: cast is a dummy operation to wrap ReplicatedVariable.
                # This makes an error in the last layer which uses embedding out surpress. 
                # Howerver, the side effect of this wrapping is unknown.
            ]


class EmbeddingSim(tf.keras.layers.Layer):
    """
    inputs: [feature_inputs, token_embedding_matrix]
    outputs: [token_probabilities]
    """

    def __init__(
            self,
            propagate_mask=False,
            use_bias=True,
            initializer='zeros',
            regularizer=None,
            constraint=None,
            **kwargs
        ):
        super(EmbeddingSim, self).__init__(**kwargs)
        
        self.propagate_mask = propagate_mask
        self.supports_masking = True
        self.bias = None
        self.use_bias = use_bias
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)
        
    def get_config(self):
        config = {
            'propagate_mask': self.propagate_mask,
            'use_bias': self.use_bias,
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
            'constraint': tf.keras.constraints.serialize(self.constraint),
        }
        config.update(super(EmbeddingSim, self).get_config())
        return config

    def build(self, input_shape):
        if self.use_bias:
            _, embed_shape = input_shape
            token_num = embed_shape[0]
            self.bias = self.add_weight(
                shape=(token_num,),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
                name='bias',
            )
        super(EmbeddingSim, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_shape, embed_shape = input_shape
        token_num = embed_shape[0]
        return feature_shape[:-1] + (token_num,)

    def compute_mask(self, inputs, mask=None):
        if not self.propagate_mask:
            return None
        
        if isinstance(mask, list):
            return mask[0]
        
        return None
    
    def call(self, inputs, mask=None):
        features, embeddings = inputs
        outputs = K.dot(features, K.transpose(embeddings))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        
        return outputs
