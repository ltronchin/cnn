from tensorflow.keras import models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import os
import math as math

class EvaluateConvNet():

    def __init__(self, run_folder):
        self.run_folder = run_folder

        self.accuracy_his = []
        self.precision_his = []
        self.recall_his = []
        self.f1_score_his = []
        self.specificity_his = []
        self.g_his = []

        self.accuracy_paziente_his = []
        self.precision_paziente_his = []
        self.recall_paziente_his = []
        self.f1_score_paziente_his = []
        self.specificity_paziente_his = []
        self.g_paziente_his = []

    def use_best_model_on_val_set(self, k):

        for idx in range(k):
            # ------------------------------ CARICAMENTO DATI DA CARTELLA DATI SLICE ----------------------------------
            print("\n ------ CARICAMENTO: {} ------ ".format(idx))
            # Caricamento del modello trainato
            path = self.run_folder + '/model/model_{}.h5'.format(idx)
            alexnet = models.load_model(path)

            # Caricamento label di test
            path = self.run_folder + '/data/true_label_val_{}.h5.npy'.format(idx)
            Y_test = np.load(path)
            Y_true = Y_test[:, 0]  # verità
            print('Shape tensore Y_true: {}'.format(Y_true.shape))

            # Caricamento ID paziente slice di test
            path = self.run_folder + '/data/ID_paziente_slice_val_{}.h5.npy'.format(idx)
            ID_paziente_slice_test = np.load(path)
            print('Shape tensore ID_paziente_slice_test: {}'.format(ID_paziente_slice_test.shape))

            # Caricamento immagini di test
            path = self.run_folder + '/data/X_val_{}.h5.npy'.format(idx)
            X_test = np.load(path)
            print('Shape tensore X_test: {}'.format(X_test.shape))

            # PAZIENTE
            # Caricamento label pazienti di test per slice
            path = self.run_folder + '/data_pazienti/lab_paziente_val_{}.h5.npy'.format(idx)
            lab_paziente_test = np.load(path)
            print('Shape tensore lab_paziente_test: {}'.format(lab_paziente_test.shape))

            # Caricamento ID pazienti di test
            path = self.run_folder + '/data_pazienti/paziente_val_{}.h5.npy'.format(idx)
            paziente_test = np.load(path)
            print('Shape tensore paziente_test: {}'.format(paziente_test.shape))

            # -------------------------------------- CLASSIFICAZIONE ---------------------------------------------------
            # Calcolo predizioni slice
            # L'output di model predict è una distribuzione di probabilità sulle 2 classi -> per ogni campione in input
            # la rete fornisce in output un vettore bidimensionale
            predictions = alexnet.predict(X_test)
            print('Shape tensore predictions: {}'.format(predictions.shape))

            Y_pred = []
            for j in range(predictions.shape[0]):
                Y_pred.append(np.argmax(predictions[j]))

            Y_pred = np.array(Y_pred)

            # ---------------------------------------- CALCOLO PERFORMANCE SLICE ---------------------------------------
            print("\nSLICE")
            accuracy, precision, recall, f1_score, specificity, g, pos_true, neg_true = self.metrics(Y_true, Y_pred, idx)

            self.accuracy_his.append(accuracy)
            self.precision_his.append(precision)
            self.recall_his.append(recall)
            self.f1_score_his.append(f1_score)
            self.specificity_his.append(specificity)
            self.g_his.append(g)
            # -------------------------------------- CALCOLO PERFORMANCE PAZIENTE ---------..---------------------------
            print("\nPAZIENTE")
            accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente, \
            specificity_paziente, g_paziente = self.predictions_pazienti(Y_pred, paziente_test, ID_paziente_slice_test,
                                                                         lab_paziente_test, idx)

            self.accuracy_paziente_his.append(accuracy_paziente)
            self.precision_paziente_his.append(precision_paziente)
            self.recall_paziente_his.append(recall_paziente)
            self.f1_score_paziente_his.append(f1_score_paziente)
            self.specificity_paziente_his.append(specificity_paziente)
            self.g_paziente_his.append(g_paziente)

            self.roc_curve(predictions[:,1], Y_true, idx)

        # --------------------------------------------- VALORI MEDI ----------------------------------------------------
        accuracy_average = np.nanmean([x for x in self.accuracy_his])
        precision_average = np.nanmean([x for x in self.precision_his])
        recall_average = np.nanmean([x for x in self.recall_his])
        f1_score_average = np.nanmean([x for x in self.f1_score_his])
        specificity_average = np.nanmean([x for x in self.specificity_his])
        g_average = np.nanmean([x for x in self.g_his])

        print('\n ------------------ METRICHE MEDIE SLICE ------------------')
        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy_average, precision_average, recall_average,
                                                     specificity_average,
                                                     f1_score_average, g_average))

        accuracy_paziente_average = np.nanmean([x for x in self.accuracy_paziente_his])
        precision_paziente_average = np.nanmean([x for x in self.precision_paziente_his])
        recall_paziente_average = np.nanmean([x for x in self.recall_paziente_his])
        f1_score_paziente_average = np.nanmean([x for x in self.f1_score_paziente_his])
        specificity_paziente_average = np.nanmean([x for x in self.specificity_paziente_his])
        g_paziente_average = np.nanmean([x for x in self.g_paziente_his])

        print('\n ------------------ METRICHE MEDIE PAZIENTI ------------------')
        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy_paziente_average, precision_paziente_average,
                                                     recall_paziente_average,
                                                     specificity_paziente_average, f1_score_paziente_average,
                                                     g_paziente_average))

        # Scrittura su file
        file = open(os.path.join(self.run_folder, "score_best_on_val_set.txt"), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD\n".format(idx + 1))
        file.write("\nScore slice:\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}\n".format(accuracy_average, precision_average, recall_average,
                                                            f1_score_average, specificity_average, g_average))
        file.write("\nScore paziente:"
                   "\nAccuratezza_paz: {}"
                   "\nPrecisione_paz: {}"
                   "\nRecall_paz: {}"
                   "\nF1_score_paz:{}"
                   "\nSpecificità_paz: {}"
                   "\nMedia delle accuratezze_paz: {}\n".format(accuracy_paziente_average, precision_paziente_average,
                                                                recall_paziente_average, f1_score_paziente_average,
                                                                specificity_paziente_average, g_paziente_average))
        file.close()

    def predictions_pazienti(self, Y_pred, paziente_test, ID_paziente_slice_test, label_paziente_test, idx):
        pred_paziente_test_list = []

        for n in range(paziente_test.shape[0]):
            zero_pred = 0
            uno_pred = 0
            # Voglio contare per il SINGOLO paziente il numero di zero e di uno predetti
            # Ciclo su tutte le slice di test
            for j in range(ID_paziente_slice_test.shape[0]):
                # Se la slice di test appartiene al paziente di test in considerazione
                if ID_paziente_slice_test[j] == paziente_test[n]:
                    #print('Slice di test {} del paziente di test {}'.format(idx, n))
                    #print(paziente_test[n], ID_paziente_slice_test[idx])
                    #print('Verità slice: {}, predizione slice: {}'.format(Y_true[idx], Y_pred[idx]))
                    if Y_pred[j] == 0:
                        zero_pred += 1
                    else:
                        uno_pred += 1
            #print(zero_pred, uno_pred)
            if zero_pred > uno_pred:
                pred_paziente_test = 0
            else:
                pred_paziente_test = 1

            pred_paziente_test_list.append(pred_paziente_test)

            #print('\n---------------------------------------------------\n')
            #print('Paziente: {}, verità paziente: {}, predizione paziente: {} (zero totali: {}, '
            #      'uno totali: {})'.format(paziente_test[n], label_paziente_test[n], pred_paziente_test_list[n], zero_pred, uno_pred))
            #print('\n---------------------------------------------------\n')

        pred_paziente_test = np.array(pred_paziente_test_list)
        accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente,\
        specificity_paziente, g_paziente, pos_true_paziente, neg_true_paziente  = self.metrics(label_paziente_test, pred_paziente_test, idx)

        return accuracy_paziente, precision_paziente, recall_paziente, f1_score_paziente, specificity_paziente, g_paziente

    def metrics(self, Y_true, Y_pred, idx):
        conf_mat = confusion_matrix(Y_true, Y_pred, labels=[0, 1])
        print("Matrice di confusione (colonne -> classe predetta, righe-> verità)\n{}".format(conf_mat))

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

        file = open(os.path.join(self.run_folder, "score_best_on_val_set.txt"), "a")
        file.write("\n-------------- FOLD: {} --------------".format(idx))
        file.write("\nScore:"
                   "\nMatrice di confusione (colonne -> classe predetta, righe-> verità)\n{}".format(conf_mat))
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

        file = open(os.path.join(self.run_folder, "score_best_on_val_set.txt"), "a")
        file.write("\nScore:"
                   "\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}\n".format(accuracy, precision, recall, f1_score, specificity, g))
        file.close()

        return accuracy, precision, recall, f1_score, specificity, g, pos_true, neg_true

    def roc_curve(self, predictions, Y_true, idx):
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

        plt.savefig(os.path.join(self.run_folder, "plot/roc_curve_{}_best_on_val_set.png".format(idx)), dpi=1200,
                    format='png')

        plt.show()
        return auc

