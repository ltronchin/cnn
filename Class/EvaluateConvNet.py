from tensorflow.keras import models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import math as math

class EvaluateConvNet():

    def __init__(self, run_folder):
        self.run_folder = run_folder

        self.predictions_slice_iterator = []
        self.true_slice_iterator = []
        self.pred_paziente_iterator = []
        self.true_paziente_iterator = []

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
            path = self.run_folder + '/model/best_model_gscore_fold{}.h5'.format(idx)
            alexnet = models.load_model(path)

            # Caricamento label di test
            path = self.run_folder + '/data/Y_val_{}.h5.npy'.format(idx)
            Y_test = np.load(path)

            print('Shape tensore Y_true: {}'.format(Y_test.shape))

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
            predictions = alexnet.predict(X_test)
            print('Shape tensore predictions: {}'.format(predictions.shape))

            self.predictions_slice_iterator.append(predictions)
            self.true_slice_iterator.append(Y_test)

            Y_pred = predictions.round()

            self.metrics(Y_test, Y_pred, idx, 'slice')

            self.predictions_pazienti(Y_pred, paziente_test, ID_paziente_slice_test, lab_paziente_test, idx)

            self.roc_curve(predictions, Y_test, idx)

        # --------------------------------------------- VALORI MEDI ----------------------------------------------------
        accuracy_average = np.nanmean([x for x in self.accuracy_his])
        precision_average = np.nanmean([x for x in self.precision_his])
        recall_average = np.nanmean([x for x in self.recall_his])
        f1_score_average = np.nanmean([x for x in self.f1_score_his])
        specificity_average = np.nanmean([x for x in self.specificity_his])
        g_average = np.nanmean([x for x in self.g_his])

        accuracy_paziente_average = np.nanmean([x for x in self.accuracy_paziente_his])
        precision_paziente_average = np.nanmean([x for x in self.precision_paziente_his])
        recall_paziente_average = np.nanmean([x for x in self.recall_paziente_his])
        f1_score_paziente_average = np.nanmean([x for x in self.f1_score_paziente_his])
        specificity_paziente_average = np.nanmean([x for x in self.specificity_paziente_his])
        g_paziente_average = np.nanmean([x for x in self.g_paziente_his])

        # Scrittura su file
        file = open(os.path.join(self.run_folder, "score_best_on_val_set.txt"), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD\n".format(k))
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

        self.metrics(np.concatenate(self.true_slice_iterator), np.concatenate(self.predictions_slice_iterator).round(), 'all_predictions', None)
        self.metrics(np.concatenate(self.true_paziente_iterator), np.concatenate(self.pred_paziente_iterator), 'all_predictions' , None)

    def predictions_pazienti(self, Y_pred, paziente_test, ID_paziente_slice_test, label_paziente_test, idx):
        pred_paziente_test_list = []

        for n in range(paziente_test.shape[0]):
            zero_pred = 0
            uno_pred = 0
            for j in range(ID_paziente_slice_test.shape[0]):
                if ID_paziente_slice_test[j] == paziente_test[n]:
                    if Y_pred[j] == 0:
                        zero_pred += 1
                    else:
                        uno_pred += 1

            if zero_pred > uno_pred:
                pred_paziente_test = 0
            else:
                pred_paziente_test = 1

            pred_paziente_test_list.append(pred_paziente_test)

        pred_paziente_test = np.array(pred_paziente_test_list)
        self.pred_paziente_iterator.append(pred_paziente_test)
        self.true_paziente_iterator.append(label_paziente_test)

        self.metrics(label_paziente_test, pred_paziente_test, idx, 'paziente')

    def metrics(self, Y_true, Y_pred, idx, metrics_slice_paziente):
        conf_mat = confusion_matrix(Y_true, Y_pred)

        tn, fp, fn, tp = conf_mat.ravel()
        neg_true = tn + fp
        pos_true = tp + fn

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

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = (2 * recall * precision) / (recall + precision)
        specificity = tn / (tn + fp)
        g = math.sqrt(recall * specificity)

        file = open(os.path.join(self.run_folder, "score_best_on_val_set.txt"), "a")
        file.write("\nScore:"
                   "\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}\n".format(accuracy, precision, recall, f1_score, specificity, g))
        file.close()

        if metrics_slice_paziente == 'slice':
            self.accuracy_his.append(accuracy)
            self.precision_his.append(precision)
            self.recall_his.append(recall)
            self.f1_score_his.append(f1_score)
            self.specificity_his.append(specificity)
            self.g_his.append(g)
        elif(metrics_slice_paziente == 'paziente'):
            self.accuracy_paziente_his.append(accuracy)
            self.precision_paziente_his.append(precision)
            self.recall_paziente_his.append(recall)
            self.f1_score_paziente_his.append(f1_score)
            self.specificity_paziente_his.append(specificity)
            self.g_paziente_his.append(g)


    def roc_curve(self, predictions, Y_true, idx):
        x = np.linspace(0, 1, 10)
        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(Y_true, predictions)
        # Calcolo AUC
        auc = roc_auc_score(Y_true, predictions)
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

