import tensorflow as tf
import numpy as np
import attr


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
    """
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))


@attr.s
class CyclicLR(LearningRateScheduler):
    """
    Cyclical learning rate policy (CLR)

    References:
    - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
    - [Cyclical Learning Rate Keras callback] (https://github.com/bckenstler/CLR)
    """

    base_lr = attr.ib(default=0.001)
    max_lr = attr.ib(default=0.006)
    step_size = attr.ib(default=2000.)
    scale_mode = attr.ib(default='cycle')
    scale_fn = attr.ib(default=None)
    mode = attr.ib(default='triangular')

    def schedule(self, epochs):
        if self.scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = self.scale_fn
            self.scale_mode = self.scale_mode

        cycle = np.floor(1 + epochs / (2 * self.step_size))
        x = np.abs(epochs / self.step_size - 2 * cycle + 1)

        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(epochs)

