from tensorflow import keras

class LearningRateScheduler(keras.callbacks.Callback):
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

    lr = float(keras.backend.get_value(self.model.optimizer.lr))
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


