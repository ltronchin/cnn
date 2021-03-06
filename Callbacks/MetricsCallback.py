
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import math

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.preprocessing.image import array_to_img

from tensorflow import keras

class MetricsCallback(keras.callbacks.Callback):

    def __init__(self, val_data, batch_size, run_folder, iter):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.run_folder = run_folder
        self.iter = iter

    # Callback invocata all'inizio del training della rete
    def on_train_begin(self, logs=None):

        self.val_specificity = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_g = []
        # Initialize the best as 0
        self.best_g = 0
        self.best_f1 = 0

    # Callback chiamata alla fine di ogni epoca
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation data.')

        batches = len(self.validation_data)
        #total = batches * self.batch_size

        # Liste che contengono rispettivamente label predetta e verità per ogni batch
        Y_pred = []
        Y_true = []
        for batch in range(batches):
            # generazione di batch di immagini -- se nel set di validazioni ci sono meno immagini rispetto a "total" (calcolato
            # come il numero di batch per il numero di immagini per batch), all'interno dell'ultima batch ci saranno solo
            # le immagini necessarie affinchè si raggiunga la dimensionalità del set di validazione
            # Un discorso analogo vale per le immagini di training
            X_val, Y_val = next(self.validation_data)

            predictions = self.model.predict(X_val)
            Y_pred.append(predictions.round())
            Y_true.append(Y_val)

        val_pred = np.asarray(list(itertools.chain.from_iterable(Y_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(Y_true)))

        # Calcolo delle metriche
        conf_mat = confusion_matrix(val_true, val_pred)
        tn, fp, fn, tp = conf_mat.ravel()
        _val_accuracy = accuracy_score(val_true, val_pred)
        _val_specificity = tn / (tn + fp)
        _val_f1 = f1_score(val_true, val_pred)
        _val_precision = precision_score(val_true, val_pred)
        _val_recall = recall_score(val_true, val_pred)
        _val_g = math.sqrt(_val_recall * _val_specificity)

        self.val_specificity.append(_val_specificity)
        self.val_f1s.append(_val_f1)
        self.val_precisions.append(_val_precision)
        self.val_recalls.append(_val_recall)
        self.val_g.append(_val_g)

        print("\n EPOCH: %d"
              "— val_acc: % f "
              "— val_f1: % f "
              "— val_precision: % f "
              "— val_recall % f "
              "— val_g % f "
              "— val_specificity % f" % (epoch, _val_accuracy,
                                         _val_f1, _val_precision,
                                         _val_recall, _val_g, _val_specificity))

        # Model checkpoint
        if epoch >= 100:
            current_g = _val_g
            if np.greater(current_g, self.best_g):
                self.best_g = current_g
                self.model.save(os.path.join(self.run_folder, "model/best_model_gscore%d_fold%d.h5" % (epoch, self.iter)))
                self.model.save(os.path.join(self.run_folder, "model/best_model_gscore_fold%d.h5" % (self.iter)))
            current_f1 = _val_f1
            if np.greater(current_f1, self.best_f1):
                self.best_f1 = current_f1
                self.model.save(os.path.join(self.run_folder, "model/best_model_f1score%d_fold%d.h5" % (epoch, self.iter)))
                self.model.save(os.path.join(self.run_folder, "model/best_model_f1score_fold%d.h5" % (self.iter)))

        return

    # Callback chiamata alla fine del training
    def on_train_end(self, logs=None):

        epochs = range(1, len(self.val_g) + 1)

        plt.figure()
        plt.plot(epochs, self.val_g, 'k', label='Val mean of accuracy')
        plt.plot(epochs, self.val_f1s, 'y', label='Val f1-score')
        plt.xlabel('Epochs')
        plt.ylabel('Validation score')
        plt.legend()
        plt.grid(True)
        plt.title('Validation score fold: {}'.format(self.iter))
        plt.savefig(os.path.join(self.run_folder, "plot/val_score_{}.png".format(self.iter)), dpi=1200, format='png')

        plt.show()
        plt.close()