# Bootstrap: estrazione casuale dei campioni con reintroduzione per creare N "nuovi"
# dataset a partire da quello di partenza (numero iterazioni Bootstrap)

# Nella formazione del nuovo dataset, ad ogni estrazione randomica il campione
# estratto viene poi reinserito nel set di partenza per poi procedere ad una nuova estrazione.
# Quindi reinserendo i campioni nel set originale, nulla vieta che questi possano essere
# nuovamente estratti per far parte del "nuovo" dataset. Il numero di estrazioni per
# la formazione del dataset è pari al numero di campioni inzialmente contenuti nel
# set di partenza, questo perché il nuovo dataset deve avere la stessa dimensione del dataset di partenza.

# CREAZIONE TRAINING E TEST
# TRAINING: con un dataset di dime campioni, occorrono dime estrazioni con reintroduzione
# per costruire il set di training. Non è detto che tutti i campioni vengano estratti,
# i campioni NON estratti dal set di partenza vanno a formare il set di TEST

from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img

from classi.Slices import Slices
from classi.Score import Score


class Run_net():

    def __init__(self,validation_method, ID_paziente, label_paziente, slices, labels, ID_paziente_slice, num_epochs, batch, boot_iter,
                 k_iter, n_patient_test, augmented, alexnet, my_callbacks, run_folder, load, augmented_crop, data_aug):
        self.validation_method = validation_method,
        self.ID_paziente = ID_paziente
        self.label_paziente = label_paziente

        self.slices = slices
        self.labels = labels
        self.ID_paziente_slice = ID_paziente_slice

        self.num_epochs = num_epochs
        self.batch = batch
        self.boot_iter = boot_iter
        self.k_iter = k_iter
        self.n_patient_test = n_patient_test
        self.augmented = augmented
        self.augmented_crop = augmented_crop
        self.data_aug = data_aug

        self.alexnet = alexnet
        self.my_callbacks = my_callbacks
        self.run_folder = run_folder
        self.load = load

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

    def run(self):
        if self.validation_method[0] == 'bootstrap':
            # Creazione di una lista di 125 indici, da 0 a 124
            index = []
            for i in range(self.ID_paziente.shape[0]):
                index.append(i)
            iter = self.boot_iter
        else:
            # Calcolo del numero di campioni per il set di validazione
            num_val_samples = len(self.ID_paziente) // self.k_iter  # // si usa per approssimare al più piccolo
            iter = self.k_iter

        first_iter = 0
        # random_state: se posto ad un intero forza il generatore di numeri random ad estrarre sempre gli stessi valori.
        # In questo caso si fa in modo che ogni volta che viene lanciato il codice vengano generate sempre
        # gli stessi boot_iter set dal metodo bootstrap
        np.random.seed(42)
        for idx in range(iter):
            if self.validation_method[0] == 'bootstrap':
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list = self.bootstrap(idx, index)
            else:
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list = self.kfold(idx, num_val_samples)

            self.write_excel(lab_paziente_val, paziente_val, lab_paziente_train, paziente_train, idx)

            # --------------------------------------------- SLICE -----------------------------------------------------
            # Divisione slice in TRAINING e VALIDATION in base allo split dei pazienti: le slice di un paziente non possono
            # trovarsi sia nel validation che nel train
            # Creazione istanza per selezione slice
            create_slices = Slices(ID_paziente_slice = self.ID_paziente_slice,
                                   slices = self.slices,
                                   labels = self.labels,
                                   paziente_train = paziente_train,
                                   paziente_val = paziente_val,
                                   paziente_test = paziente_val,
                                   run_folder = self.run_folder,
                                   k = idx)

            X_val, Y_val, true_label_val, ID_paziente_slice_val = create_slices.val()
            X_train, Y_train, true_label_train = create_slices.train()

            print("[INFO] -- Numero slice per la validazione: {}, label: {}".format(X_val.shape[0], Y_val.shape[0]))
            print("[INFO] -- Numero slice per il training: {}, label: {}".format(X_train.shape[0], Y_train.shape[0]))

            # --------------------------------- GENERAZIONE BATCH DI IMMAGINI ------------------------------------------
            # Data augmentation
            if self.augmented == 0:
                train_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True)
                step_per_epoch = int(X_train.shape[0] / self.batch)  # ad ogni epoca si fa in modo che tutti i campioni di training passino per la rete
            else:
                if self.augmented_crop == 1:
                    X_train, Y_train = self.data_aug.add_crop_slices(paziente_train, X_train, Y_train)
                    print("\n[INFO] -- Numero slice per il training con augmentation: {}, label: {}".format(X_train.shape[0], Y_train.shape[0]))

                train_datagen = ImageDataGenerator(rotation_range=45,
                                                   width_shift_range=0.20,
                                                   height_shift_range=0.20,
                                                   shear_range=25,
                                                   zoom_range=0.25,
                                                   horizontal_flip='true',
                                                   fill_mode='constant',
                                                   cval=0)

                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True)
                step_per_epoch = int(X_train.shape[0] / self.batch)
                print("\nTRAINING DELLA RETE \n[INFO] -- Step per epoca: {}".format(step_per_epoch))

            test_datagen = ImageDataGenerator()
            validation_generator = test_datagen.flow(X_val, Y_val, batch_size = self.batch, shuffle = True)

            self.how_generator_work(train_datagen, X_train)
            self.how_generator_work(test_datagen, X_val)

            # ---------------------------------------------- MODELLO ---------------------------------------------------
            # Costruzione del modello
            model = self.alexnet.build_alexnet()
            # Salvataggio struttura rete e parametri modello
            if first_iter == 0:
                self.alexnet.save(self.run_folder, model)
                first_iter = 1
            # Fit del modello
            history = model.fit(train_generator,
                              steps_per_epoch = step_per_epoch,
                              epochs = self.num_epochs,
                              validation_data = validation_generator,
                              validation_steps = (X_val.shape[0] / self.batch),
                              callbacks = callbacks_list,
                              verbose=0)
            self.all_history.append(history)
            model.save(os.path.join(self.run_folder, "model/model_end_of_train_{}.h5".format(idx)))

            # ---------------------------------------- SCORE SINGOLA FOLD ---------------------------------------------
            score = Score(X_test = X_val,
                          Y_test = true_label_val,
                          k = idx,
                          alexnet = model,
                          run_folder = self.run_folder,
                          paziente_test = paziente_val,
                          ID_paziente_slice_test = ID_paziente_slice_val,
                          label_paziente_test = lab_paziente_val)

            # Plot di accuratezza e loss per la singola fold
            # Oggetto history -> l'oggetto ha come membro "history", che è un dizionario contenente lo storico di ciò
            # che è successo durante il training
            # Valori di accuratezza per ogni epoca
            acc_history = history.history['accuracy']  # contiene i valori di accuratezza per ogni epoca
            loss_history = history.history['loss']
            val_acc_history = history.history['val_accuracy']
            val_loss_history = history.history['val_loss']

            score.plot_loss_accuracy(loss_history, val_loss_history, acc_history, val_acc_history)

            print('\n ------------------ METRICHE SLICE E PAZIENTE ------------------')
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

            # Creazione liste dei valori di accuratezza e loss (per ogni epoca) per ogni fold
            self.all_acc_history.append(acc_history)  # contiene le liste dei valori di accuratezza per ogni fold
            self.all_loss_history.append(loss_history)
            self.all_val_acc_history.append(val_acc_history)
            self.all_val_loss_history.append(val_loss_history)

        # -------------------------------------- SCORE SULLE K FOLD --------------------------------------
        print('\n---------------  Media valori loss e accuratezza sulle varie fold ----------')
        average_acc_history = [np.mean([x[i] for x in self.all_acc_history]) for i in range(self.num_epochs)]
        average_loss_history = [np.mean([x[i] for x in self.all_loss_history]) for i in range(self.num_epochs)]
        average_val_acc_history = [np.mean([x[i] for x in self.all_val_acc_history]) for i in range(self.num_epochs)]
        average_val_loss_history = [np.mean([x[i] for x in self.all_val_loss_history]) for i in range(self.num_epochs)]

        score = Score('', '', 'mean', '', self.run_folder, '', '', '')

        score.plot_loss_accuracy(average_loss_history,
                                 average_val_loss_history,
                                 average_acc_history,
                                 average_val_acc_history)

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
              '\nMedia delle accuratezze: {}'.format(accuracy_paziente_average, precision_paziente_average, recall_paziente_average,
                                      specificity_paziente_average, f1_score_paziente_average, g_paziente_average))

        # Scrittura su file
        file = open(os.path.join(self.run_folder, "score.txt"), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD".format(self.boot_iter))
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

    # Codice per eliminare duplicati da una lista
    def remove(self, duplicate_list):
        final_list = []
        for ele in duplicate_list:
            if ele not in final_list:
                final_list.append(ele)
        return final_list

    def write_excel(self, lab_paziente_val, paziente_val, lab_paziente_train, paziente_train, idx):

        # Creazione di un nuovo file Excel e aggiunta di un foglio di lavoro
        workbook_val = xlsxwriter.Workbook(os.path.join(self.run_folder, "val_{}.xlsx".format(idx)))
        worksheet_val = workbook_val.add_worksheet()
        workbook_train = xlsxwriter.Workbook(os.path.join(self.run_folder, "train_{}.xlsx".format(idx)))
        worksheet_train = workbook_train.add_worksheet()

        # Allargamento della prima e seconda colonna
        worksheet_val.set_column('A:B', 20)
        worksheet_train.set_column('A:B', 20)

        # Aggiunta del formato grassetto
        bold_val = workbook_val.add_format({'bold': True})
        bold_train = workbook_train.add_format({'bold': True})

        worksheet_val.write('A1', 'Validation Label', bold_val)
        worksheet_val.write('B1', 'Validation ID Paziente', bold_val)
        worksheet_train.write('A1', 'Train Label', bold_train)
        worksheet_train.write('B1', 'Train ID Paziente', bold_train)

        for i in range(lab_paziente_val.shape[0]):
            worksheet_val.write(i + 1, 0, lab_paziente_val[i, 0])
            worksheet_val.write(i + 1, 1, paziente_val[i, 0])

        for i in range(lab_paziente_train.shape[0]):
            worksheet_train.write(i + 1, 0, lab_paziente_train[i, 0])
            worksheet_train.write(i + 1, 1, paziente_train[i, 0])

        workbook_val.close()
        workbook_train.close()

    def how_generator_work(self, datagen, X):
        #im = X[1]
        #im = im.reshape((1,) + im.shape)
        generator = datagen.flow(X, batch_size = 64, shuffle = True)
        # Iterator restituisce un batch di immagini per ogni iterazione
        i = 0
        plt.figure(figsize=(12, 12))
        for X_batch in generator:
            img_batch = X_batch

            plt.figure(i, figsize=(12, 12))
            for idx in range(img_batch.shape[0]):
                plt.subplot(8, 8, idx + 1)
                plt.axis('off')
                image = img_batch[idx]
                plt.imshow(array_to_img(image), cmap='gray')
            i += 1
            if i % 4 == 0:
                break

        plt.tight_layout()
        plt.show()

    def bootstrap(self, idx, index):
        # ------------------------------------ DIVISIONE SET PER PAZIENTI -------------------------------------
        print('\n ------------------------ Processing BOOTSTRAP iter #{} ------------------------'.format(idx + 1))
        callbacks_list = self.my_callbacks.callbacks_list(idx)
        # n_sample: numero di campioni da estrarre.
        # replace -> se True, estrazione con reintroduzione
        # Estrazione random degli indici di 15 pazienti con reintroduzione (potrebbero essere meno di 15 pazienti
        # per effetto della reintroduzione)
        test_set_index = resample(index, replace = True, n_samples = self.n_patient_test)
        # Eliminazione degli indici duplicati dal test_set
        test_set_index = self.remove(test_set_index)
        # Creazione degli indici di training: il set di training è creato come la differenza tra il set completo e il
        # test set
        train_set_index = [x for x in index if x not in test_set_index]
        # Conversione da lista ad array
        test_set_index = np.asarray(test_set_index)
        train_set_index = np.asarray(train_set_index)
        print(test_set_index, len(test_set_index))
        print(train_set_index, len(train_set_index), "\n")

        paziente_val = self.ID_paziente[test_set_index]
        lab_paziente_val = self.label_paziente[test_set_index]
        paziente_train = self.ID_paziente[train_set_index]
        lab_paziente_train = self.label_paziente[train_set_index]
        print("[INFO] -- Numero pazienti per la validazione: {}".format(paziente_val.shape))
        print("[INFO] -- Numero pazienti per il training: {}".format(paziente_train.shape))

        return paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list

    def kfold(self, idx, num_val_samples):
        callbacks_list = self.my_callbacks.callbacks_list(idx)
        print('\n ------------------------ Processing KCROSSVAL fold #{} ------------------------'.format(idx + 1))
        # ------------------------------------------- PAZIENTI --------------------------------------------------
        # Si effettua la divisione dei set di training, validation e test sui PAZIENTI e NON sulle singole slice: questo per
        # evitare che slice di uno stesso paziente siano sia nel set di training che nel set di test introducendo un bias
        # Preparazione set di validazione: selezione dei pazienti dalla partizione kesima

        paziente_val = self.ID_paziente[idx * num_val_samples: (idx + 1) * num_val_samples]
        lab_paziente_val = self.label_paziente[idx * num_val_samples: (idx + 1) * num_val_samples]
        # Preparazione set di training: selezione dei pazienti da tutte le altre partizioni (k-1 partizioni)
        paziente_train = np.concatenate([self.ID_paziente[: idx * num_val_samples],
                                         self.ID_paziente[(idx + 1) * num_val_samples:]],
                                        axis = 0)
        lab_paziente_train = np.concatenate([self.label_paziente[: idx * num_val_samples],
                                             self.label_paziente[(idx + 1) * num_val_samples:]],
                                             axis = 0)
        print("[INFO] -- Numero pazienti per la validazione: {}, pazienti da {} a {}".format(paziente_val.shape,
                                                                                   idx * num_val_samples,
                                                                                   (idx + 1) * num_val_samples))
        print("[INFO] -- Numero pazienti per il training: {}".format(paziente_train.shape))

        return paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list