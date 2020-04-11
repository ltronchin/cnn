from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import os
import math as math


class Score():

    def __init__(self,
                 run_folder,
                 model_to_evaluate,
                 num_epochs):

        self.predictions_slice_iterator = []
        self.true_slice_iterator = []
        self.pred_paziente_iterator = []
        self.true_paziente_iterator = []

        self.run_folder = run_folder
        self.model_to_evaluate = model_to_evaluate
        self.num_epochs = num_epochs

        self.all_acc_history = []
        self.all_loss_history = []
        self.all_val_acc_history = []
        self.all_val_loss_history = []

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

    def metrics(self, Y_true, Y_pred, metrics_slice_paziente):
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
        file.write("\nSCORE {}:"
                   "\nMatrice di confusione (colonne -> classe predetta, righe-> verità)\n{}".format(metrics_slice_paziente, conf_mat))
        file.write("\nTrue negative: {}"
                   "\nFalse positive: {}"
                   "\nTrue positive: {}"
                   "\nFalse negative: {}"
                   "\nTotali veri negativi: {}"
                   "\nTotali veri positivi: {}\n".format(tn, fp, tp, fn, neg_true, pos_true))
        file.close()

        # accuracy: tp + tn / positivi veri + negativi veri -> quantifica il numero di volte che il classificatore ha
        # dato un risposta corretta (non consente di capire la distribuzione dei tp e tn)
        accuracy = accuracy_score(Y_true, Y_pred)

        # sensibilità/recall/true positive rate (tp / (tp + fn)) -> misura la frazione di positivi correttamente riconosciuta, risponde alla domanda:
        # "di tutti i campioni VERAMENTE positivi quale percentuale è stata classificata correttamente?" e si calcola
        # come il rapporto tra i positivi correttamente predetti e tutti i veri positivi
        recall = recall_score(Y_true, Y_pred)

        # precision (tp / tp + fp) -> quantifica quanti dei positivi predetti sono effettivamente positivi, risponde alla domanda
        # "di tutti i campioni PREDETTI positivi quali erano veramente positivi?"
        precision = precision_score(Y_true, Y_pred)

        # f1_score ((2 * recall * precision) / (recall + precision)) è una media ponderata delle metriche Precision e Recall - se anche solo una tra precisione e recall è
        # bassa, l'f1-score sarà basso -
        f1 = f1_score(Y_true, Y_pred)

        # specificità/true negative rate -> misura la frazione di negativi correttamente riconosciuta
        specificity = tn / (tn + fp)

        # Media geometrica delle accuratezze: radice quadrata delle recall calcolate per classe (non dipende dalla
        # probabilità a priori)
        g = math.sqrt(recall * specificity)

        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy, precision, recall, specificity, f1, g))

        # 'a' apertura del file in scrittura
        file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
        file.write("\nAccuratezza: {}"
                   "\nPrecisione: {}"
                   "\nRecall: {}"
                   "\nF1_score: {}"
                   "\nSpecificità: {}"
                   "\nMedia delle accuratezze: {}\n".format(accuracy, precision, recall, f1, specificity, g))
        file.close()

        if metrics_slice_paziente == 'slice':
            self.accuracy_his.append(accuracy)
            self.precision_his.append(precision)
            self.recall_his.append(recall)
            self.f1_score_his.append(f1)
            self.specificity_his.append(specificity)
            self.g_his.append(g)
        elif metrics_slice_paziente == 'paziente':
            self.accuracy_paziente_his.append(accuracy)
            self.precision_paziente_his.append(precision)
            self.recall_paziente_his.append(recall)
            self.f1_score_paziente_his.append(f1)
            self.specificity_paziente_his.append(specificity)
            self.g_paziente_his.append(g)

    def predictions(self, alexnet, X_test, Y_test, ID_paziente_slice_test, paziente_test, label_paziente_test, idx):

        self.alexnet = alexnet
        self.X_test = X_test
        self.Y_test = Y_test
        self.ID_paziente_slice_test = ID_paziente_slice_test

        self.paziente_test = paziente_test
        self.label_paziente_test = label_paziente_test
        self.idx = idx

        print('\n ------------------ METRICHE SLICE E PAZIENTE ({}) ------------------'.format(self.model_to_evaluate))
        file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
        file.write("\n-------------- FOLD: {} --------------".format(self.idx))
        file.close()

        predictions = self.alexnet.predict(self.X_test)

        self.predictions_slice_iterator.append(predictions)
        self.true_slice_iterator.append(self.Y_test)

        Y_pred = predictions.round()

        auc = self.roc_curve(predictions, self.Y_test)
        # Se l'area sotto la curva ROC è inferiore del 50% si ribaltano le etichette
        if auc < 0.5:
            print("---- Ribalto etichette ----")
            predictions = 1 - predictions
            Y_pred = predictions.round()
            self.roc_curve(predictions, self.Y_test)
            self.metrics(self.Y_test, Y_pred, "slice")
        else:
            self.metrics(self.Y_test, Y_pred, "slice")

        self.predictions_pazienti(Y_pred)

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

        self.pred_paziente_iterator.append(pred_paziente_test)
        self.true_paziente_iterator.append(self.label_paziente_test)

        self.metrics(self.label_paziente_test, pred_paziente_test, "paziente")

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

    def save_single_score(self, history):

        # Plot di accuratezza e loss per la singola fold
        # Oggetto history -> l'oggetto ha come membro "history", che è un dizionario contenente lo storico di ciò
        # che è successo durante il training
        # Valori di accuratezza e di loss per ogni epoca
        acc_history = history.history['accuracy']
        val_acc_history = history.history['val_accuracy']

        loss_history = history.history['loss']
        val_loss_history = history.history['val_loss']

        self.plot_loss_accuracy(loss_history, val_loss_history, acc_history, val_acc_history, False)

        # Creazione liste dei valori di accuratezza e loss per ogni fold
        self.all_acc_history.append(acc_history)
        self.all_loss_history.append(loss_history)
        self.all_val_acc_history.append(val_acc_history)
        self.all_val_loss_history.append(val_loss_history)

    def save_mean_score(self, n):
        # ----------------------------------------------- SCORE  MEDIE -------------------------------------------------
        print('\n---------------  Media valori loss e accuratezza sulle varie fold ({}) ----------'.format(self.model_to_evaluate))

        average_acc_history = [np.mean([x[i] for x in self.all_acc_history]) for i in range(self.num_epochs)]
        average_loss_history = [np.mean([x[i] for x in self.all_loss_history]) for i in range(self.num_epochs)]
        average_val_acc_history = [np.mean([x[i] for x in self.all_val_acc_history]) for i in range(self.num_epochs)]
        average_val_loss_history = [np.mean([x[i] for x in self.all_val_loss_history]) for i in range(self.num_epochs)]

        self.plot_loss_accuracy(average_loss_history, average_val_loss_history,  average_acc_history, average_val_acc_history, mean = True)

        # ------------------------ SLICE -------------------------
        accuracy_average = np.mean([x for x in self.accuracy_his])
        precision_average = np.mean([x for x in self.precision_his])
        recall_average = np.mean([x for x in self.recall_his])
        f1_score_average = np.mean([x for x in self.f1_score_his])
        specificity_average = np.mean([x for x in self.specificity_his])
        g_average = np.mean([x for x in self.g_his])

        print('\n ------------------ METRICHE MEDIE SLICE ------------------')
        print('\nAccuratezza: {}'
              '\nPrecisione: {}'
              '\nRecall: {}'
              '\nSpecificità: {}'
              '\nF1-score: {}'
              '\nMedia delle accuratezze: {}'.format(accuracy_average, precision_average, recall_average, specificity_average,
                                                     f1_score_average, g_average))
        # --------------------- PAZIENTI ------------------------
        accuracy_paziente_average = np.mean([x for x in self.accuracy_paziente_his])
        precision_paziente_average = np.mean([x for x in self.precision_paziente_his])
        recall_paziente_average = np.mean([x for x in self.recall_paziente_his])
        f1_score_paziente_average = np.mean([x for x in self.f1_score_paziente_his])
        specificity_paziente_average = np.mean([x for x in self.specificity_paziente_his])
        g_paziente_average = np.mean([x for x in self.g_paziente_his])

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
        file = open(os.path.join(self.run_folder, "score_{}.txt".format(self.model_to_evaluate)), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD\n".format(n))
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

        self.metrics(np.concatenate(self.true_slice_iterator), np.concatenate(self.predictions_slice_iterator).round(),'all_predictions slice')
        self.metrics(np.concatenate(self.true_paziente_iterator), np.concatenate(self.pred_paziente_iterator),'all_predictions paziente')

