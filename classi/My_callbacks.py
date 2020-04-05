import os
import keras
import datetime

# --------------------- Definizione Callback -----------------
class My_callbacks():

    def __init__(self, run_folder):
        self.run_folder = run_folder
        if not os.path.exists('logs'):
            os.makedirs('logs')

    def callbacks_ReduceLROnPlateau(self):
        # factor: factor by which the learning rate will be reduced.new_lr = lr * factor

        # This callback “monitors” a quantity and if no improvement is seen
        # for a ‘patience‘ number of epochs, the learning rate is reduced by the “factor” specified.Improvement is specified
        # by the “min_delta” argument. No improvement is considered if the change in the monitored quantity is less than the
        # min_delta specified.This also has an option whether you want to start evaluating the new LR instantly or give some
        # time to the optimizer to crawl with the new LR and then evaluate the monitored quantity.This is done using the
        # “cooldown” argument.You can also set the lower bound on the LR using the “min_lr” argument.No matter how many
        # epochs or what reduction factor you use, the LR will never decrease beyond “min_lr“.

        lor = keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_loss',
                factor = 0.2,
                mode='auto',
                patience = 10,
                min_lr = 0.00005,
                verbose=1)

        return lor

    def callbacks_TensorBoard(self, idx):

        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + "__fold{}".format(idx)
        tboard = keras.callbacks.TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                write_graph = True,
                write_images = False,
                profile_batch = 0)

        return tboard
