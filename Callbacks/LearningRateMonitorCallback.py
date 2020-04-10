
from tensorflow import keras
import matplotlib.pyplot as plt
import os

class LearningRateMonitorCallback(keras.callbacks.Callback):

    def __init__(self, run_folder, iter):
        super().__init__()
        self.run_folder = run_folder
        self.iter = iter

    # Callback invocata all'inizio del training della rete
    def on_train_begin(self, logs = None):
        self.lrates = []

    # Callback chiamata alla fine di ogni epoca
    def on_epoch_end(self, epoch, logs=None):
        # Prendo il LR dal ottimizzatore del modello e lo salvo in una lista
        lrate = float(keras.backend.get_value(self.model.optimizer.lr))
        print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, lrate))
        self.lrates.append(lrate)

    # Callback chiamata alla fine del training
    def on_train_end(self, logs=None):
        epochs = range(1, len(self.lrates) + 1)

        plt.figure()
        plt.plot(epochs, self.lrates, 'm', label='LR')

        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.legend()
        plt.grid(True)
        plt.title('Learning rate fold: {}'.format(self.iter))
        plt.savefig(os.path.join(self.run_folder, "plot/lr_{}.png".format(self.iter)), dpi=1200, format='png')

        plt.show()
        plt.close()
