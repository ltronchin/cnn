from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import numpy as np
import os
import math as math


class Score():

    def __init__(self,
                 X_test,
                 Y_test,
                 ID_paziente_slice_test,
                 idx,
                 alexnet,
                 run_folder,
                 paziente_test,
                 label_paziente_test,
                 model_to_evaluate):

        self.X_test = X_test
        self.Y_test = Y_test
        self.idx = idx
        self.alexnet = alexnet
        self.run_folder = run_folder
        self.paziente_test = paziente_test
        self.ID_paziente_slice_test = ID_paziente_slice_test
        self.label_paziente_test = label_paziente_test
        self.model_to_evaluate = model_to_evaluate

    def metrics(self, Y_true, Y_pred, subject):
        conf_mat = confusion_matrix(Y_true, Y_pred)
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

        file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
        file.write("\n-------------- FOLD: {} --------------".format(self.idx))
        file.write("\nScore {}:"
                   "\nMatrice di confusione (colonne -> classe predetta, righe-> verità)\n{}".format(subject, conf_mat))
        file.write("\nTrue negative: {}"
                   "\nFalse positive: {}"
                   "\nTrue positive: {}"
                   "\nFalse negative: {}"
                   "\nTotali veri negativi: {}"
                   "\nTotali veri positivi: {}\n".format(tn, fp, tp, fn, neg_true, pos_true))
        file.close()

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
            if np.isnan(recall):
                recall = 0
                file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
                file.write("\nRecall is nan")
                file.close()

            # precision: tp / tp + fp -> quantifica quanti dei positivi predetti sono effettivamente positivi, risponde alla domanda
            # "di tutti i campioni PREDETTI positivi quali erano veramente positivi?"
            precision = tp / (tp + fp)
            if np.isnan(precision):
                precision = 0
                file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
                file.write("\nPrecision is nan")
                file.close()

            # f1_score è una media ponderata delle metriche Precision e Recall - se anche solo una tra precisione e recall è
            # bassa, l'f1-score sarà basso -
            f1_score = (2 * recall * precision) / (recall + precision)
            if np.isnan(f1_score):
                f1_score = 0
                file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
                file.write("\nF1-score is nan")
                file.close()

            # specificità/true negative rate -> misura la frazione di negativi correttamente riconosciuta
            specificity = tn / (tn + fp)
            if np.isnan(specificity):
                specificity = 0
                file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
                file.write("\nSpecificity is nan")
                file.close()

            # Media geometrica delle accuratezze: radice quadrata delle recall calcolate per classe (non dipende dalla
            # probabilità a priori)
            g = math.sqrt(recall * specificity)
        else:
            print("Pazienti di una sola classe, gli score sulle singole Class vengono posti a 0")
            recall = 0
            precision = 0
            f1_score = 0
            specificity = 0
            g = 0

        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy, precision, recall, specificity, f1_score, g))

        # 'a' apertura del file in scrittura
        file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
        file.write("\nScore {}:"
                   "\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}".format(subject, accuracy, precision, recall, f1_score, specificity, g))
        file.close()

        return accuracy, precision, recall, f1_score, specificity, g, pos_true, neg_true

    def predictions(self):

        predictions = self.alexnet.predict(self.X_test)
        print('Shape tensore predictions: {}'.format(predictions.shape))

        Y_pred = predictions.round()
        print(self.Y_test.shape)

        accuracy,\
        precision,\
        recall, \
        f1_score, \
        specificity,\
        g, pos_true, \
        neg_true = self.metrics(self.Y_test, Y_pred, "Slice")

        if pos_true != 0 and neg_true != 0:
            auc = self.roc_curve(predictions, self.Y_test)
            # Se l'area sotto la curva ROC è inferiore del 50% si ribaltano le etichette
            if auc < 0.5:
                print("\n ---- Ribalto etichette ---- \n")
                predictions = 1 - predictions

                Y_pred = predictions.round()
                accuracy,\
                precision,\
                recall, \
                f1_score,\
                specificity,\
                g, pos_true,\
                neg_true = self.metrics(self.Y_test, Y_pred, "Slice")
                self.roc_curve(predictions, self.Y_test)

        accuracy_paziente, \
        precision_paziente, \
        recall_paziente, \
        f1_score_paziente,\
        specificity_paziente,\
        g_paziente = self.predictions_pazienti(Y_pred)

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
            #print('Paziente: {}, verità paziente: {}, predizione paziente: {} (zero totali: {}, '
            #      'uno totali: {})'.format(self.paziente_test[n], self.label_paziente_test[n], pred_paziente_test_list[n], zero_pred, uno_pred))
            # print('\n---------------------------------------------------\n')

        pred_paziente_test = np.array(pred_paziente_test_list)
        accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente,\
        specificity_paziente, g_paziente, pos_true_paziente, neg_true_paziente  = self.metrics(self.label_paziente_test, pred_paziente_test, "Paziente")

        return accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente, specificity_paziente, g_paziente

    def roc_curve(self, predictions, Y_true):
        x = np.linspace(0, 1, 10)
        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(Y_true, predictions)
        # Calcolo AUC
        auc = roc_auc_score(Y_true, predictions)
        print('AUC: %.3f' % auc)
        plt.plot(fpr, tpr,  color='b', label = 'AUC='+str(auc))
        plt.plot(x, x, 'k-')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/roc_curve{}.png".format(self.idx)), dpi=1200, format='png')

        plt.show()
        return auc

    def plot_loss_accuracy(self, loss_history, val_loss_history, acc_history, val_acc_history, mean):
        if mean == True:
            title = 'mean'
        else:
            title = self.idx
        # Plot della loss per traning e validation
        epochs = range(1, len(loss_history) + 1)

        plt.figure()
        plt.plot(epochs, loss_history, 'r', label = 'Training loss')
        plt.plot(epochs, val_loss_history, 'b', label = 'Validation loss')
        plt.title('Training and validation loss fold: {}'.format(title))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/train_val_loss_{}.png".format(title)), dpi=1200, format='png')

        # Plot dell'accuratezza per training e validation
        plt.figure()
        plt.plot(epochs, acc_history, 'r', label = 'Training accuracy')
        plt.plot(epochs, val_acc_history, 'b', label = 'Validation accuracy')
        plt.title('Training and validation accuracy fold: {}'.format(title))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_folder, "plot/train_val_accuracy_{}.png".format(title)), dpi=1200, format='png')
        plt.show()
        plt.close()


