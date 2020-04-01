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
from tensorflow.keras import models

from classi.Slices import Slices
from classi.SaveScore import SaveScore
from classi.Score import Score


class Run_net():

    def __init__(self,validation_method, ID_paziente, label_paziente, slices, labels, ID_paziente_slice, num_epochs, batch, boot_iter,
                 k_iter, n_patient_test, augmented, fill_mode, alexnet, my_callbacks, run_folder, load, data_aug):
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

        first_iter = 0
        # random_state: se posto ad un intero forza il generatore di numeri random ad estrarre sempre gli stessi valori.
        # In questo caso si fa in modo che ogni volta che viene lanciato il codice vengano generate sempre
        # le stesse iterazioni dal metodo bootstrap
        np.random.seed(42)
        for idx in range(iter):
            if self.validation_method[0] == 'bootstrap':
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list = self.bootstrap(idx, index)
            else:
                paziente_train, lab_paziente_train, paziente_val, lab_paziente_val, callbacks_list = self.kfold(idx, num_val_samples)

            # Salvataggio dei set creati ad ogni iterazione di kfold o bootstrap
            np.save(os.path.join(self.run_folder, "data_pazienti/paziente_val_{}.h5".format(idx)), paziente_val, allow_pickle = False)
            np.save(os.path.join(self.run_folder, "data_pazienti/lab_paziente_val_{}.h5".format(idx)), lab_paziente_val, allow_pickle=False)

            # Chiamata per la creazione di un foglio excel che mostra la suddivisione dei pazienti in training e test
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
                                   idx = idx)

            X_val, Y_val, true_label_val, ID_paziente_slice_val = create_slices.val()
            X_train, Y_train, true_label_train, ID_paziente_slice_train = create_slices.train()

            # Salvataggio dei set creati ad ogni iterazione di kfold o bootstrap
            np.save(os.path.join(self.run_folder, "data/X_train_{}.h5".format(idx)), X_train, allow_pickle =False)
            np.save(os.path.join(self.run_folder, "data/true_label_train_{}.h5".format(idx)), true_label_train, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/ID_paziente_slice_train_{}.h5".format(idx)), ID_paziente_slice_train, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/X_val_{}.h5".format(idx)), X_val, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/true_label_val_{}.h5".format(idx)), true_label_val, allow_pickle=False)
            np.save(os.path.join(self.run_folder, "data/ID_paziente_slice_val_{}.h5".format(idx)), ID_paziente_slice_val, allow_pickle=False)

            print("\n[INFO] -- Numero slice per la validazione: {}, label: {}".format(X_val.shape[0], Y_val.shape[0]))
            print("[INFO] -- Numero slice per il training: {}, label: {}".format(X_train.shape[0], Y_train.shape[0]))

            # --------------------------------- GENERAZIONE BATCH DI IMMAGINI ------------------------------------------
            # Data augmentation
            #total_expand_slice = 15000

            if self.augmented == 0:
                train_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle = True)
                step_per_epoch = int(X_train.shape[0] / self.batch)  # ad ogni epoca si fa in modo che tutti i campioni di training passino per la rete
            else:
                # Chiamata alla funzione per l'ESPANSIONE manuale del dataset
                # X_aug, Y_aug = self.expand_dataset(total_expand_slice, paziente_train, X_train, true_label_train, ID_paziente_slice_train)
                # print("\n[INFO] -- Numero slice ottenute con data augmentation: {}, label: {}".format(X_aug.shape, Y_aug.shape))
                # X_train_aug = np.concatenate((X_train, X_aug), axis=0)
                # Y_train_aug = np.concatenate((Y_train, Y_aug), axis=0)
                # print("[INFO] -- slice per il training della rete: {}, label {}".format(X_train_aug.shape, Y_train_aug.shape))

                train_datagen = ImageDataGenerator(rotation_range = 175,
                                                   width_shift_range = (-5, +5),
                                                   height_shift_range = (-5, +5),
                                                   shear_range = 20 ,
                                                   horizontal_flip = 'true',
                                                   vertical_flip='true',
                                                   fill_mode = self.fill_mode,
                                                   cval = 0)

                #train_datagen_elastic = ImageDataGenerator(rotation_range = 175,
                #                                           width_shift_range = (-5, +5),
                #                                           height_shift_range = (-5, +5),
                #                                           horizontal_flip = 'true',
                #                                           vertical_flip='true',
                #                                           preprocessing_function=lambda x: self.data_aug.elastic_transform(x, [30,40], 5, random_state=None),
                #                                           fill_mode=self.fill_mode,
                #                                           cval=0)

                #train_generator = self.multiple_generator(train_datagen, train_datagen_elastic, X_train, Y_train)

                train_generator = train_datagen.flow(X_train, Y_train, batch_size = self.batch, shuffle=True)
                print(self.batch)
                print(X_train.shape[0])
                step_per_epoch = X_train.shape[0] // (self.batch)
                print("\nTRAINING DELLA RETE \n[INFO] -- Step per epoca: {}".format(step_per_epoch))

            test_datagen = ImageDataGenerator()
            validation_generator = test_datagen.flow(X_val, Y_val, batch_size = self.batch, shuffle = True)

            self.how_generator_work(train_datagen, X_train)
            #self.how_generator_work(train_datagen_elastic, X_train)
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
                                validation_steps = X_val.shape[0] // (self.batch),
                                callbacks = callbacks_list,
                                verbose=0)

            # Salvataggio del modello alla fine del training
            model.save(os.path.join(self.run_folder, "model/model_end_of_train_{}.h5".format(idx)), include_optimizer=False)

            # Caricamento del modello checkpoint -- modello che ha permesso di ottenere la massima accuratezza sul set
            # di validazione
            best_on_val_set = models.load_model(self.run_folder + '/model/model_{}.h5'.format(idx))

            score = Score(X_test=X_val,
                          Y_test=true_label_val,
                          ID_paziente_slice_test=ID_paziente_slice_val,
                          idx=idx,
                          alexnet=model,
                          run_folder=self.run_folder,
                          paziente_test=paziente_val,
                          label_paziente_test=lab_paziente_val,
                          best_on_val_set = 'end_of_training')

            # ---------------------------------------- SCORE ----------------------------------------------
            save_score = SaveScore(idx = idx,
                                   run_folder = self.run_folder,
                                   num_epochs = self.num_epochs)

            save_score.save_single_score(score, history, best_on_val_set = 'end_of_training')

        # ---------------------------------------- SCORE MEDIATO SULLE VARIE FOLD ----------------------------------------------
        save_score.save_mean_score(score, best_on_val_set = 'end_of_training')

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
            if i % 1 == 0:
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

    def multiple_generator(self, train_datagen, train_datagen_clone, X_train, label):

        train_generator = train_datagen.flow(X_train, label, batch_size= self.batch, shuffle = True)
        train_generator_clone = train_datagen_clone.flow(X_train, label, batch_size = self.batch, shuffle = True)
        while True:
            X_train = train_generator.next()
            X_train_clone = train_generator_clone.next()
            yield np.concatenate((X_train[0], X_train_clone[0]), axis=0), np.concatenate((X_train[1], X_train_clone[1]), axis=0)

    def expand_dataset(self, total_augmented_slice, paziente_train, X_train, true_label_train, ID_paziente_slice_train):
        # Creazione slice trasformate per la corrente iterazione -- per ogni slice si applicano 7 trasformazioni (vedi
        # DataAugmentation per maggiori info).
        # Il metodo restituisce:
        #
        # X_aug: insieme delle slice trasformate
        # Y_aug: etichette delle slice trasformate
        # idd_aug: id delle slice trasformate

        # Total_augmented_slice rappresenta il numero di slice che si vogliono selezionare per espandere il dataset:
        # queste slice vengono campionate in modo randomico da X_aug facendo in modo che per ogni paziente vengano
        # campionate lo stesso numero di "nuove" slice
        count = 0
        augmented_slice_per_patient = total_augmented_slice // paziente_train.shape[0]
        print("\n[INFO] -- numero di slice da aggiungere per paziente per arrivare a {}: {}".format(total_augmented_slice,
                                                                                                  augmented_slice_per_patient))
        number_of_slice_per_patient = []

        X_train_singleidd = []
        Y_train_singleidd = []
        ID_paziente_slice_train_singleidd = []

        for n in range(0, paziente_train.shape[0]):
            for idx in range(ID_paziente_slice_train.shape[0]):
                if ID_paziente_slice_train[idx] == paziente_train[n]:
                    X_train_singleidd.append(X_train[idx])
                    Y_train_singleidd.append(true_label_train[idx])
                    ID_paziente_slice_train_singleidd.append(ID_paziente_slice_train[idx])
                    count += 1

            number_of_slice_per_patient.append(count)
            count = 0
            print("\n[INFO] -- numero di slice per paziente {}: {}".format(n, number_of_slice_per_patient[n]))
            X_train_singleidd = np.array(X_train_singleidd)
            Y_train_singleidd = np.array(Y_train_singleidd)
            ID_paziente_slice_train_singleidd = np.array(ID_paziente_slice_train_singleidd)
            X_aug_singleidd, Y_aug_singleidd, idd_aug_singleidd = self.data_aug.augment_data(X_train_singleidd, Y_train_singleidd,
                                                                                             ID_paziente_slice_train_singleidd,
                                                                                             augmented_slice_per_patient,
                                                                                             n)
            if n == 0:
                X_aug = X_aug_singleidd
                Y_aug = Y_aug_singleidd
                idd_aug = idd_aug_singleidd
            else:
                X_aug = np.concatenate((X_aug, X_aug_singleidd), axis=0)
                Y_aug = np.concatenate((Y_aug, Y_aug_singleidd), axis=0)
                idd_aug = np.concatenate((idd_aug, idd_aug_singleidd), axis=0)
            X_train_singleidd = []
            Y_train_singleidd = []
            ID_paziente_slice_train_singleidd = []
            print("[INFO] -- Numero slice ottenute con data augmentation: {}, label: {}, idd: {}".format(X_aug.shape, Y_aug.shape, idd_aug.shape))
        return X_aug, Y_aug