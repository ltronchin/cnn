from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import numpy as np
import os
import math as math


class Score():

    def __init__(self, X_test, Y_test, k, alexnet, run_folder, paziente_test, ID_paziente_slice_test, label_paziente_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.k = k
        self.alexnet = alexnet
        self.run_folder = run_folder
        self.paziente_test = paziente_test
        self.ID_paziente_slice_test = ID_paziente_slice_test
        self.label_paziente_test = label_paziente_test

    def metrics(self, Y_true, Y_pred):
        conf_mat = confusion_matrix(Y_true, Y_pred, labels=[0, 1])
        print("\nMatrice di confusione (colonne -> classe predetta, righe-> verità)\n{}".format(conf_mat))

        # Calcolo:
        # true negative (tn)
        # false positive (fp)
        # false negative (fn)
        # true positive (tp)

        # Matrice di conf:
        #    predicted
        #
        #    TN     FP
        #
        #    FN     TP
        #

        tn, fp, fn, tp = conf_mat.ravel()
        neg_true = tn + fp
        pos_true = tp + fn

        print("True negative: {}"
              "\nFalse positive: {}"
              "\nTrue positive: {}"
              "\nFalse negative: {}"
              "\nTotali veri negativi: {}"
              "\nTotali veri positivi: {}".format(tn, fp, tp, fn, neg_true, pos_true))

        # accuracy: tp + tn / positivi veri + negativi veri -> quantifica il numero di volte che il classificatore ha
        # dato un risposta corretta (non consente di capire la distribuzione dei tp e tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Se nel set di test ci sono SOLO pazienti positivi (PFS maggiore uguale 11) o SOLO paziente negativi (PFS minore uguale 11 mesi)
        # non ha senso calcolare gli score per singola classe, per questo vengono messi a nan
        if pos_true != 0 and neg_true != 0:
            # sensibilità/recall/true positive rate -> misura la frazione di positivi correttamente riconosciuta, risponde alla domanda:
            # "di tutti i campioni VERAMENTE positivi quale percentuale è stata classificata correttamente?" e si calcola
            # come il rapporto tra i positivi correttamente predetti e tutti i veri positivi
            recall = tp / (tp + fn)

            # precision: tp / tp + fp -> quantifica quanti dei positivi predetti sono effettivamente positivi, risponde alla domanda
            # "di tutti i campioni PREDETTI positivi quali erano veramente positivi?"
            precision = tp / (tp + fp)

            # f1_score è una media ponderata delle metriche Precision e Recall - se anche solo una tra precisione e recall è
            # bassa, l'f1-score sarà basso -
            f1_score = (2 * recall * precision) / (recall + precision)

            # specificità/true negative rate -> misura la frazione di negativi correttamente riconosciuta
            specificity = tn / (tn + fp)

            # Media geometrica delle accuratezze: radice quadrata delle recall calcolate per classe (non dipende dalla
            # probabilità a priori)
            g = math.sqrt(recall * specificity)
        else:
            print("Pazienti di una sola classe, gli score sulle singole classi vengono posti a nan")
            recall = math.nan
            precision = math.nan
            f1_score = math.nan
            specificity = math.nan
            g = math.nan

        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy, precision, recall, specificity, f1_score, g))

        return accuracy, precision, recall, f1_score, specificity, g, pos_true, neg_true

    def predictions(self):
        # ---------- Calcolo predizioni slice ------------
        # L'output di model predict è una distribuzione di probabilità sulle 2 classi -> per ogni campione in input
        # la rete fornisce in output un vettore bidimensionale
        predictions = self.alexnet.predict(self.X_test)
        # predictions -> tensore 2D (N slice di TEST, Numero classi)
        # print('Shape tensore predictions: {}'.format(predictions.shape))
        # ogni riga in predictions è un tensore 1D di dimensionalità 2 (2 sole classi), la somma dei coefficienti in ogni
        # tensore è pari a 1
        # print("Somma delle probabilità per ogni slice di test: {}".format(np.sum(predictions[0])))
        # print(true_label)
        Y_pred = []
        Y_true = self.Y_test[:, 0]  # verità

        # l'indice del valore più grande nel tensore 1D è la classe predetta -- la classe con la più alta probabilità --
        # la funzione argmax restituisce l'indice del massimo valore lungo un asse del tensore
        for idx in range(predictions.shape[0]):
            Y_pred.append(np.argmax(predictions[idx]))

        Y_pred = np.asarray(Y_pred)

        accuracy, precision, recall, f1_score, specificity, g, pos_true, neg_true = self.metrics(Y_true, Y_pred)

        if pos_true != 0 and neg_true != 0:
            auc = self.roc_curve(predictions, Y_true)
        else:
            auc = math.nan

        # 'a' apertura del file in scrittura
        file = open(os.path.join(self.run_folder, "score.txt"), "a")
        file.write("\n-------------- FOLD: {} --------------".format(self.k))
        file.write("\nScore slice:\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}"
                   "\nAUC: {}\n".format(accuracy, precision, recall, f1_score, specificity, g, auc))
        file.close()

        accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente,\
        specificity_paziente, g_paziente = self.predictions_pazienti(Y_pred)

        return (accuracy, precision, recall, f1_score, specificity, g, accuracy_paziente, precision_paziente,
                recall_paziente, f1_score_paziente, specificity_paziente, g_paziente)

    def predictions_pazienti(self, Y_pred):
        # ----------- Metriche su pazienti ------------
        pred_paziente_test_list = []

        for n in range(self.paziente_test.shape[0]):
            zero_pred = 0
            uno_pred = 0
            # Voglio contare per il SINGOLO paziente il numero di zero e di uno predetti
            # Ciclo su tutte le slice di test
            for idx in range(self.ID_paziente_slice_test.shape[0]):
                # Se la slice di test appartiene al paziente di test in considerazione
                if self.ID_paziente_slice_test[idx] == self.paziente_test[n]:
                    # print('Slice di test {} del paziente di test {}'.format(idx, n))
                    # print(paziente_test[n], ID_paziente_slice_test[idx])
                    # print('Verità slice: {}, predizione slice: {}'.format(Y_true[idx], Y_pred[idx]))
                    if Y_pred[idx] == 0:
                        zero_pred += 1
                    else:
                        uno_pred += 1
                    # print(zero_pred, uno_pred)
            if zero_pred > uno_pred:
                pred_paziente_test = 0
            else:
                pred_paziente_test = 1

            pred_paziente_test_list.append(pred_paziente_test)
            # print('\n---------------------------------------------------\n')
            # print('Paziente: {}, verità paziente: {}, predizione paziente: {} (zero totali: {}, '
            #      'uno totali: {})'.format(paziente_test[n], lab_test[n], pred_paziente_test_list[n], zero_pred, uno_pred))
            # print('\n---------------------------------------------------\n')

        pred_paziente_test = np.asarray(pred_paziente_test_list)
        # print('\n---------------------------------------------------\n')
        accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente,\
        specificity_paziente, g_paziente, pos_true_paziente, neg_true_paziente  = self.metrics(self.label_paziente_test, pred_paziente_test)

        file = open(os.path.join(self.run_folder, "score.txt"), "a")
        file.write("\nScore paziente:\nAccuratezza_paz: {}"
                   "\nPrecisione_paz: {}"
                   "\nRecall_paz: {}"
                   "\nF1_score_paz: {}"
                   "\nSpecificità_paz: {}"
                   "\nMedia delle accuratezze_paz:  {}\n".format(accuracy_paziente, precision_paziente, recall_paziente,
                                                    f1_score_paziente, specificity_paziente, g_paziente))
        file.close()
        return accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente, specificity_paziente, g_paziente

    def roc_curve(self, predictions, Y_true):
        x = np.linspace(0, 1, 10)
        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(Y_true, predictions[:,1])
        # Calcolo AUC
        auc = roc_auc_score(Y_true, predictions[:,1])
        print('AUC: %.3f' % auc)
        plt.plot(fpr, tpr,  color='b', label = 'AUC='+str(auc))
        plt.plot(x, x, 'k-')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/roc_curve{}.png".format(self.k)), dpi=1200, format='png')

        plt.show()
        return auc

    def plot_loss_accuracy(self, loss_history, val_loss_history, acc_history, val_acc_history):
        # Plot della loss per traning e validation
        epochs = range(1, len(loss_history) + 1)

        plt.figure()
        plt.plot(epochs, loss_history, 'r', label = 'Training loss')
        plt.plot(epochs, val_loss_history, 'b', label = 'Validation loss')
        plt.title('Training and validation loss fold: {}'.format(self.k))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/train_val_loss_{}.png".format(self.k)), dpi=1200, format='png')

        # Plot dell'accuratezza per training e validation
        plt.figure()
        plt.plot(epochs, acc_history, 'r', label = 'Training accuracy')
        plt.plot(epochs, val_acc_history, 'b', label = 'Validation accuracy')
        plt.title('Training and validation accuracy fold: {}'.format(self.k))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/train_val_accuracy_{}.png".format(self.k)), dpi=1200, format='png')
        plt.show()
        plt.close()



