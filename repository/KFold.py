# ------------------- k-fold CROSS VALIDATION ---------------------
import numpy as np
import os
import xlsxwriter
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

from classi.Slices import Slices
from classi.Score import Score

class KFold():

    def __init__(self, ID_paziente, label_paziente, slices, labels, ID_paziente_slice, num_epochs, batch, factor,  k, augmented, alexnet, my_callbacks, run_folder):
        self.paziente_kfold = ID_paziente
        self.lab_paziente_kfold = label_paziente

        self.slices = slices
        self.labels = labels
        self.ID_paziente_slice = ID_paziente_slice

        self.num_epochs = num_epochs
        self.batch = batch
        self.factor = factor
        self.k = k
        self.augmented = augmented

        self.alexnet = alexnet
        self.my_callbacks = my_callbacks
        self.run_folder = run_folder

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

        self.accuracy_paziente_his = []
        self.precision_paziente_his = []
        self.recall_paziente_his = []
        self.f1_score_paziente_his = []
        self.specificity_paziente_his = []

    def k_fold_cross_validation(self):
        first_iter = 0
        # Calcolo del numero di campioni per il set di validazione
        num_val_samples = len(self.paziente_kfold) // self.k  # // si usa per approssimare al più piccolo
        # Ciclo sulle k fold
        for idx in range(self.k):
            callbacks_list = self.my_callbacks.callbacks_list(idx)
            print('\n ------------------------ Processing fold #{} ------------------------'.format(idx + 1))
            # ------------------------------------------- PAZIENTI --------------------------------------------------
            # Si effettua la divisione dei set di training, validation e test sui PAZIENTI e NON sulle singole slice: questo per
            # evitare che slice di uno stesso paziente siano sia nel set di training che nel set di test introducendo un bias
            # Preparazione set di validazione: selezione dei pazienti dalla partizione kesima

            paziente_val = self.paziente_kfold[idx * num_val_samples: (idx + 1) * num_val_samples]
            lab_paziente_val = self.lab_paziente_kfold[idx * num_val_samples: (idx + 1) * num_val_samples]
            # Preparazione set di training: selezione dei pazienti da tutte le altre partizioni (k-1 partizioni)
            paziente_train = np.concatenate([self.paziente_kfold[: idx * num_val_samples],
                                            self.paziente_kfold[(idx + 1) * num_val_samples:]],
                                            axis=0)
            print("Numero pazienti per la validazione: {}, pazienti da {} a {}".format(paziente_val.shape,
                                                                                       idx * num_val_samples,
                                                                                       (idx + 1) * num_val_samples))
            print("Numero pazienti per il training: {}".format(paziente_train.shape))

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
            X_train, Y_train, true_label_train, ID_paziente_slice_train = create_slices.train()

            self.write_excel(true_label_val, ID_paziente_slice_val, true_label_train, ID_paziente_slice_train, idx)

            print("Numero slice per la validazione: {}, label: {}".format(X_val.shape[0], Y_val.shape[0]))
            print("Numero slice per il training: {}, label: {}".format(X_train.shape[0], Y_train.shape[0]))

            # --------------------------------- GENERAZIONE BATCH DI IMMAGINI ------------------------------------------
            # Data augmentation
            if self.augmented == 0:
                train_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True, seed = 42)
                step_per_epoch = int(X_train.shape[0] / self.batch) # ad ogni epoca si fa in modo che tutti i campioni di training passino
                # per la rete
                print('Step per epoca: {}'.format(step_per_epoch))
            else:
                train_datagen = ImageDataGenerator(rotation_range = 45,
                                                   width_shift_range = 0.2,
                                                   height_shift_range= 0.2,
                                                   shear_range = 0.2)
                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True, seed = 42)
                step_per_epoch = int(X_train.shape[0] / self.batch) * self.factor
                print(step_per_epoch)
            test_datagen = ImageDataGenerator()
            validation_generator = test_datagen.flow(X_val, Y_val, batch_size = self.batch, shuffle = True, seed = 42)

            # ---------------------------------------------- MODELLO ---------------------------------------------------
            # Costruzione del modello
            model = self.alexnet.build_alexnet()
            # Salvataggio struttura rete e parametri modello
            if first_iter == 0:
                self.alexnet.save(self.run_folder, model)
                first_iter = 1
            # Fit del modello
            history = model.fit_generator(train_generator,
                                          steps_per_epoch = step_per_epoch,
                                          epochs = self.num_epochs,
                                          validation_data = validation_generator,
                                          validation_steps = (X_val.shape[0] / self.batch),
                                          callbacks = callbacks_list,
                                          verbose = 0)
            self.all_history.append(history)
            model.save(os.path.join(self.run_folder,"model/model_end_of_train_{}.h5".format(idx)))

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
            accuracy, precision, recall, f1_score, specificity, accuracy_paziente, precision_paziente, recall_paziente,\
                                                            f1_score_paziente, specificity_paziente = score.predictions()

            self.accuracy_his.append(accuracy)
            self.precision_his.append(precision)
            self.recall_his.append(recall)
            self.f1_score_his.append(f1_score)
            self.specificity_his.append(specificity)

            self.accuracy_paziente_his.append(accuracy_paziente)
            self.precision_paziente_his.append(precision_paziente)
            self.recall_paziente_his.append(recall_paziente)
            self.f1_score_paziente_his.append(recall_paziente)
            self.specificity_paziente_his.append(specificity_paziente)

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
        score.plot_loss_accuracy(average_loss_history, average_val_loss_history, average_acc_history, average_val_acc_history)

        #------------------------ SLICE -------------------------
        accuracy_average = np.mean([x for x in self.accuracy_his])
        precision_average = np.mean([x for x in self.precision_his])
        recall_average = np.mean([x for x in self.recall_his])
        f1_score_average = np.mean([x for x in self.f1_score_his])
        specificity_average = np.mean([x for x in self.specificity_his])
        print('\n ------------------ METRICHE MEDIE SLICE ------------------')
        print('\nAccuratezza: {}\nPrecisione: {}\nRecall: {}\nSpecificità: '
              '{}\nF1-score: {}'.format(accuracy_average,
                                        precision_average,
                                        recall_average,
                                        specificity_average,
                                        f1_score_average))
        # --------------------- PAZIENTI ------------------------
        accuracy_paziente_average = np.mean([x for x in self.accuracy_paziente_his])
        precision_paziente_average = np.mean([x for x in self.precision_paziente_his])
        recall_paziente_average = np.mean([x for x in self.recall_paziente_his])
        f1_score_paziente_average = np.mean([x for x in self.f1_score_paziente_his])
        specificity_paziente_average = np.mean([x for x in self.specificity_paziente_his])
        print('\n ------------------ METRICHE MEDIE PAZIENTI ------------------')
        print('\nAccuratezza: {}\nPrecisione: {}\nRecall: {}\nSpecificità:'
              ' {}\nF1-score: {}'.format(
                                        accuracy_paziente_average,
                                        precision_paziente_average,
                                        recall_paziente_average,
                                        specificity_paziente_average,
                                        f1_score_paziente_average))


        file = open(os.path.join(self.run_folder, "score.txt"), "a")
        file.write("\nMEDIA SCORE SULLE {} FOLD".format(self.k))
        file.write("\nScore slice:\nAccuratezza: {}\nPrecisione: {}\nRecall: {}\nF1_score:"
                   " {}\nSpecificità: {}\n".format(
                                                    accuracy_average,
                                                    precision_average,
                                                    recall_average,
                                                    f1_score_average,
                                                    specificity_average))
        file.write("\nScore paziente:\nAccuratezza_paz: {}\nPrecisione_paz: {}\nRecall_paz: {}\nF1_score_paz:"
                   " {}\nSpecificità_paz: {}\n".format(accuracy_paziente_average,
                                                       precision_paziente_average,
                                                       recall_paziente_average,
                                                       f1_score_paziente_average,
                                                       specificity_paziente_average))
        file.close()


    def write_excel(self, true_label_val, ID_paziente_slice_val, true_label_train, ID_paziente_slice_train, idx):

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

        for i in range(true_label_val.shape[0]):
            worksheet_val.write(i + 1, 0, true_label_val[i, 0])
            worksheet_val.write(i + 1, 1, ID_paziente_slice_val[i, 0])

        for i in range(true_label_train.shape[0]):
            worksheet_train.write(i + 1, 0, true_label_train[i, 0])
            worksheet_train.write(i + 1, 1, ID_paziente_slice_train[i, 0])

        workbook_val.close()
        workbook_train.close()

    def how_generator_work(self, datagen, X, ID, name):

        generator = datagen.flow(X, ID, batch_size = 64, shuffle = True, seed=42)
        # Iterator restituisce un batch di immagini per ogni iterazione
        i = 0
        for X_batch, Y_batch in generator:
            img_batch = X_batch
            ID_batch = Y_batch
            #print(img_batch.shape)
            #print(ID_batch)

            plt.figure(i, figsize=(12, 12))

            for idx in range(img_batch.shape[0]):
                plt.subplot(8, 8, idx + 1)
                plt.axis('off')
                image = img_batch[idx]
                plt.imshow(array_to_img(image), cmap='gray')
            i += 1
            if i % 1 == 0:
                break
        plt.tight_layout()
        plt.show()