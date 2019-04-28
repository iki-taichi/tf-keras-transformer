# coding: utf-8


"""
Transformer Model Creation
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import keras_support

from .layer_normalization import LayerNormalization
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .trig_pos_embd import TrigPosEmbedding
from .embeddings import EmbeddingWithWeights, EmbeddingSim


__all__ = [
    'get_custom_objects', 
    'gelu',
    'compile_transformer_model',
    'get_transformer_model',
    'get_encoders', 
    'get_decoders', 
    'get_encoder_component', 
    'get_decoder_component', 
]


def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingWithWeights': EmbeddingWithWeights,
        'EmbeddingSim': EmbeddingSim,
        'gelu': gelu,
    }


def gelu(x):
    return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))


def get_transformer_model(config, model_path=None, weights_only=True, custom_objects=None):
    _custom_objects = get_custom_objects()
    if custom_objects is not None:
        _custom_objects.update(custom_objects)
    with tf.keras.utils.custom_object_scope(_custom_objects):
        
        if model_path is None:
            model = _get_transformer_model(**config.asdict())
        else:
            if weights_only:
                model = _get_transformer_model(**config.asdict())
                model.load_weights(model_path)
            else:
                model = tf.keras.models.load_model(model_path)        
                
        if config.use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(config.tpu_grpc_url)
            strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
            model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)  
    
    return model


def _get_transformer_model(
        token_num,
        input_len,
        embed_dim,
        block_num,
        head_num,
        hidden_dim,
        attention_activation=None,
        feed_forward_activation='relu',
        dropout_rate=0.0,
        use_same_embed=True,
        embed_weights=None,
        embed_trainable=None,
        trainable=True,
        **kwargs,
    ):
    """
    Get full model without compilation.
     
    args:
        token_num: Number of distinct tokens for encoder/decoder, tuple or int.
        embed_dim: Dimension of token embedding.
        block_num: Number of encoder/decoder components, tuple or int.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        dropout_rate: Dropout rate.
        use_same_embed: Whether to use the same token embedding layer. `token_num`, `embed_weights` and
                           `embed_trainable` should be lists of two elements if it is False.
        embed_weights: Initial weights of token embedding.
        embed_trainable: Whether the token embedding is trainable. It will automatically set to False if the given
                            value is None when embedding weights has been provided.
        trainable: Whether the layers are trainable.
    return:
        Keras model.
    """
    
    if not isinstance(token_num, (tuple, list)):
        token_num = [token_num, token_num]
    encoder_token_num, decoder_token_num = token_num

    if not isinstance(embed_weights, (tuple, list)):
        embed_weights = [embed_weights, embed_weights]
    encoder_embed_weights, decoder_embed_weights = embed_weights
    if encoder_embed_weights is not None:
        encoder_embed_weights = [encoder_embed_weights]
    if decoder_embed_weights is not None:
        decoder_embed_weights = [decoder_embed_weights]

    if not isinstance(embed_trainable, (tuple, list)):
        embed_trainable = [embed_trainable, embed_trainable]
    encoder_embed_trainable, decoder_embed_trainable = embed_trainable
    if encoder_embed_trainable is None:
        encoder_embed_trainable = encoder_embed_weights is None
    if decoder_embed_trainable is None:
        decoder_embed_trainable = decoder_embed_weights is None
    
    if not isinstance(input_len, (tuple, list)):
        input_len = (input_len, input_len)
    encoder_input_len, decoder_input_len = input_len
    
    if not isinstance(block_num, (tuple, list)):
        block_num = (block_num, block_num)
    encoder_num, decoder_num = block_num
    
    input_names=[]
    inputs = {}
    for key in ('Token-Ids', 'Pos-Ids', 'Group-Ids'):
        name = 'Encoder-'+key
        inputs[name] = tf.keras.layers.Input(
                dtype=tf.int32, 
                shape=(encoder_input_len,), 
                name=name,
            )
        input_names.append(name)
    for key in ('Token-Ids', 'Pos-Ids', 'Group-Ids'):
        name = 'Decoder-'+key
        inputs[name] = tf.keras.layers.Input(
                dtype=tf.int32, 
                shape=(decoder_input_len,), 
                name=name,
            )
        input_names.append(name)
    
    if use_same_embed:
        encoder_embed_layer = decoder_embed_layer = EmbeddingWithWeights(
                input_dim=encoder_token_num,
                output_dim=embed_dim,
                weights=encoder_embed_weights,
                trainable=encoder_embed_trainable,
                name='Token-Embedding',
            )
    else:
        encoder_embed_layer = EmbeddingWithWeights(
                input_dim=encoder_token_num,
                output_dim=embed_dim,
                weights=encoder_embed_weights,
                trainable=encoder_embed_trainable,
                name='Encoder-Token-Embedding',
            )
        decoder_embed_layer = EmbeddingWithWeights(
                input_dim=decoder_token_num,
                output_dim=embed_dim,
                weights=decoder_embed_weights,
                trainable=decoder_embed_trainable,
                name='Decoder-Token-Embedding',
            )
    
    encoder_raw_embed = encoder_embed_layer(inputs['Encoder-Token-Ids'])[0]
    encoder_embed = TrigPosEmbedding(
            mode=TrigPosEmbedding.MODE_ADD,
            name='Encoder-Embedding',
        )([encoder_raw_embed, inputs['Encoder-Pos-Ids']])
    
    encoded_layer = get_encoders(
            encoder_num=encoder_num,
            input_layer=[encoder_embed, inputs['Encoder-Group-Ids']],
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
    )
    
    decoder_raw_embed, decoder_embed_weights = \
            decoder_embed_layer(inputs['Decoder-Token-Ids'])
    
    decoder_embed = TrigPosEmbedding(
            mode=TrigPosEmbedding.MODE_ADD,
            name='Decoder-Embedding',
        )([decoder_raw_embed, inputs['Decoder-Pos-Ids']])
    
    decoded_layer = get_decoders(
            decoder_num=decoder_num,
            input_layer=[
                    decoder_embed, 
                    inputs['Encoder-Group-Ids'], 
                    inputs['Decoder-Group-Ids']
                ],
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    
    dense_layer = EmbeddingSim(
            propagate_mask=False,
            trainable=trainable,
            name='Output',
        )([decoded_layer, decoder_embed_weights])
    
    model = tf.keras.models.Model(
            inputs=[inputs[_] for _ in input_names], 
            outputs=dense_layer
        )
    
    return model

    
def get_encoders(
        encoder_num,
        input_layer,
        head_num,
        hidden_dim,
        attention_activation=None,
        feed_forward_activation='relu',
        dropout_rate=0.0,
        trainable=True
    ):
    """
    Get encoders.
    
    args:
        encoder_num: Number of encoder components.
        input_layer: Embedding Input layer.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    return:
        Output layer.
    """
    
    last_layer = input_layer[0]
    encoder_group_ids = input_layer[1]
    
    for i in range(encoder_num):
        last_layer = get_encoder_component(
                name='Encoder-%d' % (i + 1),
                input_layer=[last_layer, encoder_group_ids],
                head_num=head_num,
                hidden_dim=hidden_dim,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
                trainable=trainable,
            )
    return last_layer


def get_decoders(
        decoder_num,
        input_layer,
        encoded_layer,
        head_num,
        hidden_dim,
        attention_activation=None,
        feed_forward_activation='relu',
        dropout_rate=0.0,
        trainable=True
    ):
    """
    Get decoders.
    args:
        decoder_num: Number of decoder components.
        input_layer: Input layer.
        encoded_layer: Encoded layer from encoder.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    return: 
        Output layer.
    """
    
    last_layer = input_layer[0]
    group_ids = input_layer[1:]
    
    for i in range(decoder_num):
        last_layer = get_decoder_component(
                name='Decoder-%d' % (i + 1),
                input_layer=[last_layer]+group_ids,
                encoded_layer=encoded_layer,
                head_num=head_num,
                hidden_dim=hidden_dim,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
                trainable=trainable,
            )
    return last_layer


def get_encoder_component(
        name,
        input_layer,
        head_num,
        hidden_dim,
        attention_activation=None,
        feed_forward_activation='relu',
        dropout_rate=0.0,
        trainable=True
    ):
    """
    Multi-head self-attention and feed-forward layer.
    
    args:
        name: Prefix of names for internal layers.
        input_layer: Input layer.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    return: 
        Output layer.
    """
    
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    
    attention_layer = get_residual_unit(
            name=attention_name,
            input_layer=input_layer,
            build_func=lambda x: MultiHeadAttention(
                    head_num=head_num,
                    name=attention_name,
                    activation=attention_activation,
                    history_only=False,
                    trainable=trainable,
                )(x),
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    
    feed_forward_layer = get_residual_unit(
            name=feed_forward_name,
            input_layer=attention_layer,
            build_func=lambda x: FeedForward(
                    name=feed_forward_name,
                    units=hidden_dim,
                    activation=feed_forward_activation,
                    trainable=trainable,
                )(x),
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    
    return feed_forward_layer


def get_decoder_component(name,
                          input_layer,
                          encoded_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True):
    """
    Multi-head self-attention, multi-head query attention and feed-forward layer.
    args:
        name: Prefix of names for internal layers.
        input_layer: Input layer.
        encoded_layer: Encoded layer from encoder.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    return: 
        Output layer.
    """
    
    last_layer, encoder_group_ids, decoder_group_ids = input_layer
    
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    
    self_attention_layer = get_residual_unit(
            name=self_attention_name,
            input_layer=[last_layer, decoder_group_ids],
            build_func=lambda x: MultiHeadAttention(
                    head_num=head_num,
                    name=self_attention_name,
                    activation=attention_activation,
                    history_only=True,
                    trainable=trainable,
                )(x),
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    
    query_attention_layer = get_residual_unit(
            name=query_attention_name,
            input_layer=[
                    self_attention_layer, 
                    decoder_group_ids,
                    encoded_layer, 
                    encoded_layer,
                    encoder_group_ids,
                ],
            build_func=lambda x: MultiHeadAttention(
                    head_num=head_num,
                    name=query_attention_name,
                    activation=attention_activation,
                    history_only=False,
                    trainable=trainable,
                )(x),
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    
    feed_forward_layer = get_residual_unit(
            name=feed_forward_name,
            input_layer=query_attention_layer,
            build_func=lambda x: FeedForward(
                    name=feed_forward_name,
                    units=hidden_dim,
                    activation=feed_forward_activation,
                    trainable=trainable,
                )(x),
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return feed_forward_layer


def get_residual_unit(name, input_layer, build_func, dropout_rate=0.0, trainable=True):
    """
    residual unit contains residual, normalization and dropout.
    args:
        name: Prefix of names for internal layers.
        input_layer: Input layer.
        build_func: A callable that takes the input tensor and generates the output tensor.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    return:
        Output layer.
    """
    
    build_output = build_func(input_layer)
    
    if dropout_rate > 0.0:
        dropout_layer = tf.keras.layers.Dropout(
                rate=dropout_rate,
                name='%s-Dropout' % name,
            )(build_output)
    else:
        dropout_layer = build_output
    
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    
    add_layer = tf.keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = LayerNormalization(
            trainable=trainable,
            name='%s-Norm'%name,
        )(add_layer)
    
    return normal_layer
