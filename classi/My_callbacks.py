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
            keras.callbacks.TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                write_graph = True,
                write_images = False,
                profile_batch = 0,
            )
        ]

        return callbacks_list
