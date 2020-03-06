import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


class Slices():

    def __init__(self, ID_paziente_slice, slices, labels, paziente_train, paziente_val, paziente_test, run_folder, k):

        self.ID_paziente_slice = ID_paziente_slice
        self.slices = slices
        self.labels = labels
        self.paziente_train = paziente_train
        self.paziente_val = paziente_val
        self.paziente_test = paziente_test
        self.run_folder = run_folder
        self.k = k

    # -------------------- Creazione set di Training ------------------------
    def train(self):
        X_train = []
        Y_train = []
        ID_paziente_slice_train = []
        # Scorre il vettore dei pazienti di training
        for n in range(self.paziente_train.shape[0]):
            # Scorre il vettore delle slice
            for idx in range(self.slices.shape[0]):
                # Se l'ID paziente relativo alla slice è uguale all'ID paziente presente nel set di training si
                # aggiunge un elemento alla lista
                if self.ID_paziente_slice[idx] == self.paziente_train[n]:
                    X_train.append(self.slices[idx])
                    Y_train.append(self.labels[idx])
                    ID_paziente_slice_train.append(self.ID_paziente_slice[idx])

        # Conversione liste ad array
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        ID_paziente_slice_train = np.asarray(ID_paziente_slice_train)

        # Si aggiunge un ASSE al tensore X_train portandolo ad essere un tensore 4D. La rete in ingresso vuole tensori
        # di dimensione 4D con la forma: (samples dimension x row x column x channels) dove sample dimensione determina
        # il numero di slice, r e c sono rispettivamente le righe e le colonne di ciascuna slice e channels è il
        # numero di canali
        X_train = X_train[:, :, :, np.newaxis]
        Y_train_categ = to_categorical(Y_train, 2, dtype='uint8')

        #plt.imshow(array_to_img(X_train[78]), cmap='gray')
        self.count_label_train(Y_train)

        return X_train, Y_train_categ, Y_train, ID_paziente_slice_train

    def shuffle_in_unison(self, X_train, Y_train, ID_paziente_slice_train):
            n_elem = X_train.shape[0]
            np.random.seed(42)
            indeces = np.random.choice(n_elem, size = n_elem, replace = False)

            return X_train[indeces], Y_train[indeces], ID_paziente_slice_train[indeces]

    # -------------------- Creazione set di Validazione ----------------
    def val(self):
        X_val = []
        Y_val = []
        ID_paziente_slice_val = []

        # Ciclo sui pazienti di validazione
        for n in range(self.paziente_val.shape[0]):
            # Ciclo su tutte le slice: si individuano le slice il cui ID_paziente è nel set di validazione
            for idx in range(self.slices.shape[0]):
                if self.ID_paziente_slice[idx] == self.paziente_val[n]:
                    X_val.append(self.slices[idx])
                    Y_val.append(self.labels[idx])
                    ID_paziente_slice_val.append(self.ID_paziente_slice[idx])

        X_val = np.asarray(X_val)
        Y_val = np.asarray(Y_val)
        ID_paziente_slice_val = np.asarray(ID_paziente_slice_val)

        # Si aggiunge un ASSE al tensore X_val portandolo ad essere un tensore 4D
        X_val = X_val[:, :, :, np.newaxis]
        Y_val_categ = to_categorical(Y_val, 2, dtype='uint8')

        self.count_label_val(Y_val)

        return X_val, Y_val_categ, Y_val, ID_paziente_slice_val

    # -------------------- Creazione set di Test ------------------------
    def test(self):
        X_test = []
        Y_test = []
        ID_paziente_slice_test = []
        for n in range(self.paziente_test.shape[0]):
            idx = 0
            for idx in range(self.slices.shape[0]):
                if self.ID_paziente_slice[idx] == self.paziente_test[n]:
                    X_test.append(self.slices[idx])
                    Y_test.append(self.labels[idx])
                    ID_paziente_slice_test.append(self.ID_paziente_slice[idx])

        # Conversione da lista ad array
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)
        ID_paziente_slice_test = np.asarray(ID_paziente_slice_test)

        # Si aggiunge un ASSE al tensore X_test portandolo ad essere un tensore 4D -> si aggiunge l'asse dei canali (1 canale
        # essendo l'immagine in scala di grigi)
        X_test = X_test[:, :, :, np.newaxis]
        Y_test_categ = to_categorical(Y_test, 2, dtype='uint8')

        self.count_label_test(Y_test)

        return X_test, Y_test_categ, Y_test, ID_paziente_slice_test

    # ------------------ Verifica del bilanciamento del set di training  ------------------
    def count_label_train(self, Y_train):
        one = 0
        zero = 0
        for label in Y_train:
            if (label == 1):
                one += 1
            else:
                zero += 1
        print('------------- TRAIN -------------')
        print('{} è presente {} volte'.format(1, one))
        print('{} è presente {} volte'.format(0, zero))

    # ------------------ Verifica del bilanciamento del set di validation  ------------------
    def count_label_val(self, Y_val):
        one = 0
        zero = 0
        for label in Y_val:
            if (label == 1):
                one += 1
            else:
                zero += 1
        print('------------- VALIDATION -------------')
        print('{} è presente {} volte'.format(1, one))
        print('{} è presente {} volte'.format(0, zero))

    # ------------------ Verifica del bilanciamento del set di test  ------------------
    def count_label_test(self, Y_test):
        one = 0
        zero = 0
        for label in Y_test:
            if (label[1] == 1):
                one += 1
            else:
                zero += 1
        print('------------- TEST -------------')
        print('{} è presente {} volte'.format(1, one))
        print('{} è presente {} volte'.format(0, zero))
