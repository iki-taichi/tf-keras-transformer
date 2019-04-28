# coding: utf-8


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler.
    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """      
           
    def __init__(self, schedule, verbose=0):
        super(BatchLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.offset = 0
           
    def on_epoch_begin(self, epoch, logs=None):
        self.offset = epoch*self.params['steps']
           
    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
           
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.schedule(batch + self.offset)
           
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                    'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: BatchLearningRateScheduler reducing learning '
                    'rate to %s.' % (batch + 1, lr))
           
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
           
           
class NBatchLogger(tf.keras.callbacks.Callback):
    """    
    A Logger that log average performance per `display` steps.
    """    
    def __init__(self, display, custom_metrics=None):
        self.epoch_offset = 0
        self.step = 0
        self.display = display
        self.custom_metrics = custom_metrics or []
        self.metric_cache = {}
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_offset = epoch*self.params['steps']
           
    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics'] + self.custom_metrics:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' %s=%.4f'%(k, val)
                else:
                    metrics_log += ' %s=%.4e'%(k, val)
            total_step = self.epoch_offset + batch + 1
            print('step_run={} step_total={} step_per_epoch={}{}'.format(
                    self.step,
                    total_step,
                    self.params['steps'],
                    metrics_log
                ))
            self.metric_cache.clear()    