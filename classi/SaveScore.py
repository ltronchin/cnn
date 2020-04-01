import numpy as np
import os

class SaveScore():
    def __init__(self, idx, run_folder, num_epochs):

        self.num_epochs = num_epochs
        self.run_folder = run_folder
        self.idx = idx

        self.all_history = []
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

    def save_single_score(self, score, history, best_on_val_set):

        self.all_history.append(history)

        # Plot di accuratezza e loss per la singola fold
        # Oggetto history -> l'oggetto ha come membro "history", che è un dizionario contenente lo storico di ciò
        # che è successo durante il training
        # Valori di accuratezza per ogni epoca
        acc_history = history.history['accuracy']  # contiene i valori di accuratezza per ogni epoca
        loss_history = history.history['loss']
        val_acc_history = history.history['val_accuracy']
        val_loss_history = history.history['val_loss']

        # Creazione liste dei valori di accuratezza e loss (per ogni epoca) per ogni fold
        self.all_acc_history.append(acc_history)  # contiene le liste dei valori di accuratezza per ogni fold
        self.all_loss_history.append(loss_history)
        self.all_val_acc_history.append(val_acc_history)
        self.all_val_loss_history.append(val_loss_history)

        score.plot_loss_accuracy(loss_history, val_loss_history, acc_history, val_acc_history, False)

        print('\n ------------------ METRICHE SLICE E PAZIENTE ({}) ------------------'.format(best_on_val_set))
        accuracy, precision, recall, f1_score, specificity, g, accuracy_paziente, precision_paziente, recall_paziente, \
        f1_score_paziente, specificity_paziente, g_paziente = score.predictions()

        self.accuracy_his.append(accuracy)
        self.precision_his.append(precision)
        self.recall_his.append(recall)
        self.f1_score_his.append(f1_score)
        self.specificity_his.append(specificity)
        self.g_his.append(g)

        self.accuracy_paziente_his.append(accuracy_paziente)
        self.precision_paziente_his.append(precision_paziente)
        self.recall_paziente_his.append(recall_paziente)
        self.f1_score_paziente_his.append(f1_score_paziente)
        self.specificity_paziente_his.append(specificity_paziente)
        self.g_paziente_his.append(g_paziente)

    def save_mean_score(self, score, best_on_val_set):
        # ----------------------------------------------- SCORE  MEDIE -------------------------------------------------
        print('\n---------------  Media valori loss e accuratezza sulle varie fold ({}) ----------'.format(best_on_val_set))

        average_acc_history = [np.mean([x[i] for x in self.all_acc_history]) for i in range(self.num_epochs)]
        average_loss_history = [np.mean([x[i] for x in self.all_loss_history]) for i in range(self.num_epochs)]
        average_val_acc_history = [np.mean([x[i] for x in self.all_val_acc_history]) for i in range(self.num_epochs)]
        average_val_loss_history = [np.mean([x[i] for x in self.all_val_loss_history]) for i in range(self.num_epochs)]

        score.plot_loss_accuracy(average_loss_history,
                                      average_val_loss_history,
                                      average_acc_history,
                                      average_val_acc_history,
                                      mean = True)

        # ------------------------ SLICE -------------------------
        # nanmean: compute the arithmetic mean along the specified axis, ignoring NaNs
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
              '\nMedia delle accuratezze: {}'.format(accuracy_average, precision_average, recall_average, specificity_average,
                                                     f1_score_average, g_average))
        # --------------------- PAZIENTI ------------------------
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
        file = open(os.path.join(self.run_folder, "score_{}.txt".format(best_on_val_set)), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD\n".format(self.idx + 1))
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









