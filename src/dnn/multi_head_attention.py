# coding: utf-8


"""
Multi-head attention layer

Based on https://github.com/CyberZHG/keras-multi-head
Framework was changed from keras to tf.keras to compile TPU model using keras-support.

ToDo: content
"""


import tensorflow as tf
import tensorflow.keras.backend as K


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    
    Inputs are masked based on group_ids to restrict attentions within each groups
    Input pattern 1:
        inputs=[query=key=value, group_ids]
        mask= group_ids[:, None] == group_ids[None, :]
    Input pattern 2:
        inputs=[query, query_group_ids, key, value, key_group_ids]
        mask= query_group_ids[:, None] == key_group_ids[None, :]
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.history_only = history_only
        
        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return input_shape[0]
        elif len(input_shape) == 5:
            q, _, _, v, _ = input_shape
            return q[:-1] + (int(v[-1]),)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if len(input_shape) == 5:
            q, _, k, v, _ = input_shape
        else:
            q = k = v = input_shape[0]
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)
    
    @staticmethod
    def _get_shape(x):
        shape_x = K.shape(x)
        x_tensor_shape = x.shape
        return tuple(
                shape_x[i] if ts.value is None else ts
                for i, ts in enumerate(x_tensor_shape)
            )
    
    @staticmethod
    def _reshape_to_batches(x, head_num):
        _, seq_len, feature_dim = MultiHeadAttention._get_shape(x)
        head_dim = feature_dim // head_num
        x = K.reshape(x, [-1, seq_len, head_num, head_dim])
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        x = K.reshape(x, [-1, seq_len, head_dim])
        return x

    @staticmethod
    def _reshape_from_batches(x, head_num):
        _, seq_len, feature_dim = MultiHeadAttention._get_shape(x)
        x = K.reshape(x, [-1, head_num, seq_len, feature_dim])
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (-1, seq_len, feature_dim * head_num))
    
    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        _, seq_len = MultiHeadAttention._get_shape(mask)
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, K.stack([1, head_num, 1]))
        return K.reshape(mask, [-1, seq_len])
    
    @staticmethod
    def scaled_dot_product_attention(
            inputs,      
            mask=None, 
            return_attention=False, 
            history_only=False
        ):
        
        query, key, value, query_group_ids, key_group_ids = inputs
        
        if isinstance(mask, list):
            mask = mask[1]
        
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        
        group_mask = tf.equal(query_group_ids[:, :, None], key_group_ids[:, None, :])
        e -= (1.0 - tf.cast(group_mask, tf.float32)) * 1e9
        
        if history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            ones = tf.ones((query_len, key_len))
            e -= (ones - tf.matrix_band_part(ones, -1, 0)) * 1e9
        
        if mask is not None:
            e -= (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx())) * 1e9
    
        a = tf.keras.activations.softmax(e)
        v = K.batch_dot(a, value, axes=[2, 1])
    
        if return_attention:
            return [v, a]
        return v

    def call(self, inputs, mask=None):
        if len(inputs)==5:
            q, q_group_ids, k, v, k_group_ids = inputs
        else:
            q = k = v = inputs[0]
            q_group_ids = k_group_ids = inputs[1]
        
        if isinstance(mask, list):
            if len(inputs)==5:
                q_mask, _, k_mask, v_mask, _ = mask
            else:
                q_mask = k_mask = v_mask = mask[0]
        else:
            q_mask = k_mask = v_mask = mask
        
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        
        y = self.scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
                self._reshape_mask(q_group_ids, self.head_num),
                self._reshape_mask(k_group_ids, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
            history_only=self.history_only,
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        
        return y
