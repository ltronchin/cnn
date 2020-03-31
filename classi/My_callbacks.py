import os
import keras
import datetime

# --------------------- Definizione Callback -----------------
class My_callbacks():

    def __init__(self, run_folder):
        self.run_folder = run_folder
        if not os.path.exists('logs'):
            os.makedirs('logs')

    def callbacks_list(self, k):
        log_dir = "logs\\fit\\"+ datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")+"__fold{}".format(k)
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor = 'accuracy',
                patience = 100,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.run_folder,"model/model_{}.h5".format(k)),
                monitor='val_accuracy',
                #save_weights_only = True,
                save_best_only = True,
            ),
            #keras.callbacks.ReduceLROnPlateau(
            #    monitor = 'val_loss',
            #    factor = 0.6,
            #    patience = 20,
            #    mode = min,
            #    min_lr = 0.00005
            # ),
            #factor: factor by which the learning rate will be reduced.new_lr = lr * factor

            keras.callbacks.TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                write_graph = True,
                write_images = False,
                profile_batch = 0,
            )
        ]

        return callbacks_list
