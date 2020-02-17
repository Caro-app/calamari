from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.python.framework import ops

class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    Arguments:
        schedule: a function that takes an iteration index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self._total_batches_seen_lr = 0


    def on_train_batch_begin(self, batch, logs=None):

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(self._total_batches_seen_lr, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(self._total_batches_seen_lr)
        if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                            'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nIteration %05d: LearningRateScheduler reducing learning '
                    'rate to %s.' % (self._total_batches_seen_lr + 1, lr))


    def on_train_batch_end(self, batch, logs=None):
        self._total_batches_seen_lr += 1
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)