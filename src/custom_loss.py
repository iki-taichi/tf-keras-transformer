# coding: utf-8


import tensorflow as tf


def masked_softmax_cross_entropy_with_logit(y_true, y_pred):
    """
    Masks elements that is not equal to -1 in y_true and
    averages alive-elements.
    
    Sample weight and the standard mask of layers should not be enabled.
    If they work averaging will be done twice time. 
    
    Standard mask system of Kreras does not seem to work correctly on TPU
    So this loss is a temporary measure.
    """
        
    if len(y_true.shape)+1 != len(y_pred.shape):
        y_true = tf.squeeze(y_true, axis=[-1])
           
    mask = tf.not_equal(y_true, -1)
    safe_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    safe_y_true = tf.cast(safe_y_true, dtype=tf.int32)
           
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=safe_y_true,
            logits=y_pred,
        )  
           
    mask = tf.cast(mask, tf.float32)
           
    sum_cross_entropy = tf.reduce_sum(cross_entropy*mask)
    sum_mask = tf.reduce_sum(mask)
           
    averaged_loss = tf.div_no_nan(sum_cross_entropy, sum_mask)
           
    return averaged_loss
           
           
def masked_sparse_categorical_accuracy(y_true, y_pred):
    """
    Counter part of masked_softmax_cross_entropy_with_logit
    """
    
    if len(y_true.shape)+1 != len(y_pred.shape):
        y_true = tf.squeeze(y_true, axis=[-1])
           
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.equal(y_true, y_pred_id)
    matches = tf.cast(matches, tf.float32)
           
    mask = tf.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32)
           
    averaged = tf.div_no_nan(tf.reduce_sum(matches*mask), tf.reduce_sum(mask))
           
    return averaged