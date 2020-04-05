
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import math
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow import keras


class Custom_callbacks(keras.callbacks.Callback):

    def __init__(self, val_data, batch_size, run_folder, iter):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.run_folder = run_folder
        self.iter = iter

    def on_train_begin(self, logs=None):

        self.val_specificity = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_g = []
        # Initialize the best as 0
        self.best = 0

    # Chiamata alla fine di ogni epoca
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation data.')

        batches = len(self.validation_data)
        #total = batches * self.batch_size

        Y_pred = []
        Y_true = []
        for batch in range(batches):
            # generazione di batch di immagini -- se nel set di validazioni ci sono meno immagini rispetto a "total" (calcolato
            # come il numero di batch per il numero di immagini per batch), all'interno dell'ultima batch ci saranno solo
            # le immagini necessarie affinchè si raggiunga la dimensionalità del set di validazione
            # Un discorso analogo vale per le immagini di training
            X_val, Y_val = next(self.validation_data)

            Y_pred_batch = []
            Y_true_batch = []

            # argmax restituisce l'indice del massimo valore lungo un asse
            predictions = self.model.predict(X_val)
            for idx in range(predictions.shape[0]):
                Y_pred_batch.append(np.argmax(predictions[idx]))
                Y_true_batch.append(np.argmax(Y_val[idx]))

            Y_pred.append(Y_pred_batch)
            Y_true.append(Y_true_batch)

        val_pred = np.asarray(list(itertools.chain.from_iterable(Y_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(Y_true)))

        conf_mat = confusion_matrix(val_true, val_pred, labels=[0, 1])
        tn, fp, fn, tp = conf_mat.ravel()
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

        print("\n— val_f1: % f "
              "— val_precision: % f "
              "— val_recall % f "
              "— val_g % f "
              "— val_specificity % f" % (_val_f1, _val_precision,_val_recall, _val_g, _val_specificity))

        # Model checkpoint
        current = _val_g
        if np.greater(current, self.best):
            self.best = current
            self.model.save(os.path.join(self.run_folder, "model/best_model_fold%d.h5" % (self.iter)))
        return

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