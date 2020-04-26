from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
import os
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img

from Class.Slices import Slices
from Class.Score import Score
from Callbacks.MetricsCallback import MetricsCallback
from Callbacks.LearningRateMonitorCallback import LearningRateMonitorCallback
from Callbacks.LearningRateScheduler import LearningRateScheduler

class Run_net():

    def __init__(self,
                 validation_method,
                 ID_paziente,
                 label_paziente,
                 slices,
                 labels,
                 ID_paziente_slice,
                 num_epochs,
                 batch,
                 boot_iter,
                 k_iter,
                 n_patient_test,
                 augmented,
                 fill_mode,
                 alexnet,
                 my_callbacks,
                 run_folder,
                 lr_decay):

        self.validation_method = validation_method
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
        self.fill_mode = fill_mode

        self.alexnet = alexnet
        self.my_callbacks = my_callbacks
        self.run_folder = run_folder

        self.score = Score(run_folder=self.run_folder,
                           model_to_evaluate='end_of_training',
                           num_epochs=self.num_epochs)

        # (epoch to start, learning rate) tuples
        self.LR_SCHEDULE = [(10, 0.0001), (60, 0.00001), (10, 0.000001)]
        self.lr_decay = lr_decay

    def lr_schedule(self, epoch, lr):
        if epoch < self.LR_SCHEDULE[0][0] or epoch > self.LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(self.LR_SCHEDULE)):
            if epoch == self.LR_SCHEDULE[i][0]:
                return self.LR_SCHEDULE[i][1]
        return lr

    def run(self):
        if self.validation_method[0] == 'bootstrap':
            # Creazione di una lista di indici quanti sono il numero di pazienti
            index = []
            for i in range(self.ID_paziente.shape[0]):
                index.append(i)
            # boot_iter contiene il numero di iterazioni da effettuare per il metodo bootstrap
            iter = self.boot_iter
        else:
            # Calcolo del numero di campioni per il set di validazione
            num_val_samples = len(self.ID_paziente) // self.k_iter  # // si usa per approssimare al più piccolo
            iter = self.k_iter

        # random_state: se posto ad un intero forza il generatore di numeri random ad estrarre sempre gli stessi valori.
        # In questo caso si fa in modo che ogni volta che viene lanciato il codice vengano generate sempre
        # le stesse iterazioni dal metodo bootstrap
        np.random.seed(42)
        for self.idx in range(iter):
            if self.validation_method[0] == 'bootstrap':
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val = self.bootstrap(self.idx, index)
            else:
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val = self.kfold(self.idx, num_val_samples)


            # Salvataggio dei set creati ad ogni iterazione di kfold o bootstrap
            np.save(os.path.join(self.run_folder, "data_pazienti/paziente_val_{}.h5".format(self.idx)), paziente_val, allow_pickle = False)
            np.save(os.path.join(self.run_folder, "data_pazienti/lab_paziente_val_{}.h5".format(self.idx)), lab_paziente_val, allow_pickle=False)

            # Chiamata per la creazione di un foglio excel che mostra la suddivisione dei pazienti in training e test
            self.write_excel(lab_paziente_val, paziente_val, lab_paziente_train, paziente_train, self.idx)

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
                                   idx = self.idx)

            X_val, Y_val, ID_paziente_slice_val = create_slices.val()
            X_train, Y_train, ID_paziente_slice_train = create_slices.train()

            # Salvataggio dei set creati ad ogni iterazione di kfold o bootstrap
            np.save(os.path.join(self.run_folder, "data/X_train_{}.h5".format(self.idx)), X_train, allow_pickle =False)
            np.save(os.path.join(self.run_folder, "data/Y_train_{}.h5".format(self.idx)), Y_train, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/ID_paziente_slice_train_{}.h5".format(self.idx)), ID_paziente_slice_train, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/X_val_{}.h5".format(self.idx)), X_val, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/Y_val_{}.h5".format(self.idx)), Y_val, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/ID_paziente_slice_val_{}.h5".format(self.idx)), ID_paziente_slice_val, allow_pickle=False)

            print("\n[INFO] -- Numero slice per la validazione: {}, label: {}".format(X_val.shape[0], Y_val.shape[0]))
            print("[INFO] -- Numero slice per il training: {}, label: {}".format(X_train.shape[0], Y_train.shape[0]))

            # --------------------------------- GENERAZIONE BATCH DI IMMAGINI ------------------------------------------
            # Data augmentation

            if self.augmented == 0:
                train_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True)
            else:

                train_datagen = ImageDataGenerator(rotation_range = 175,
                                                   width_shift_range = (-7, +7),
                                                   height_shift_range = (-7, +7),
                                                   shear_range = 20,
                                                   horizontal_flip = 'true',
                                                   vertical_flip='true',
                                                   fill_mode = self.fill_mode,
                                                   cval = 0)

                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle=True)

            test_datagen = ImageDataGenerator()
            validation_generator = test_datagen.flow(X_val, Y_val, batch_size = self.batch, shuffle = False)
            print(self.batch)
            print(X_train.shape[0])
            print(X_val.shape[0])
            step_per_epoch = math.ceil(X_train.shape[0] / (self.batch))
            step_per_epoch_val = math.ceil(X_val.shape[0] / (self.batch))
            print("\nTRAINING DELLA RETE \n[INFO] "
                  "-- Step per epoca set di training: {}\n"
                  "-- Step per epoca set di validazione: {}".format(step_per_epoch, step_per_epoch_val))

            self.how_generator_work(train_datagen, X_train)
            self.how_generator_work(test_datagen, X_val)

            # ---------------------------------------------- MODELLO ---------------------------------------------------
            # Costruzione di un nuovo modello
            model = self.alexnet.build_alexnet()

            lr_monitor = LearningRateMonitorCallback(self.run_folder, self.idx)
            metrics= MetricsCallback(validation_generator, self.batch, self.run_folder, self.idx)
            lr_scheduling = LearningRateScheduler(self.lr_schedule)
            if self.lr_decay == True:
                callbacks_list = [metrics, lr_monitor, lr_scheduling]
            else:
                callbacks_list = [metrics]

            # Fit del modello
            history = model.fit(train_generator,
                                steps_per_epoch = step_per_epoch,
                                epochs = self.num_epochs,
                                validation_data = validation_generator,
                                validation_steps = step_per_epoch_val,
                                callbacks = callbacks_list,
                                verbose=0)

            # Salvataggio del modello alla fine del training
            model.save(os.path.join(self.run_folder, "model/model_end_of_train_{}.h5".format(self.idx)), include_optimizer=False)

            # ---------------------------------------- SCORE ----------------------------------------------
            self.score.predictions(alexnet=model,
                                   X_test=X_val,
                                   Y_test=Y_val,
                                   ID_paziente_slice_test=ID_paziente_slice_val,
                                   paziente_test=paziente_val,
                                   label_paziente_test=lab_paziente_val,
                                   idx = self.idx)

            self.score.save_single_score(history)

        # ---------------------------------------- SCORE MEDIATO SULLE VARIE FOLD ----------------------------------------------
        self.score.save_mean_score(iter)

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

        generator = datagen.flow(X, batch_size = 64, shuffle = True)
        # Iterator restituisce una batch di immagini per ogni iterazione
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
            if i % 1 == 0:
                break

        plt.tight_layout()
        plt.show()

    def bootstrap(self, idx, index):
        # --------------------------------------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------------------------------------

        print('\n ------------------------ Processing BOOTSTRAP iter #{} ------------------------'.format(idx + 1))
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

        return paziente_train, lab_paziente_train, paziente_val, lab_paziente_val

    def kfold(self, idx, num_val_samples):
        # --------------------------------------------------------------------------------------------------------------
        # Si effettua la divisione dei set di training, validation e test sui PAZIENTI e NON sulle singole slice: questo per
        # evitare che slice di uno stesso paziente siano sia nel set di training che nel set di test introducendo un bias
        # Preparazione set di validazione: selezione dei pazienti dalla partizione kesima
        # --------------------------------------------------------------------------------------------------------------

        print('\n ------------------------ Processing KCROSSVAL fold #{} ------------------------'.format(idx + 1))
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

        return paziente_train, lab_paziente_train, paziente_val, lab_paziente_val

