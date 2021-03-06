import scipy.io as sio  # libreria per importare i dati da Matlab
import numpy as np
import copy
from tensorflow.keras.preprocessing.image import img_to_array


class Load():
    def __init__(self, path_pazienti):
        self.path_pazienti = path_pazienti

    def read_from_path(self, path_slices, view):
        # ------------------ Caricamento dati da Matlab ------------------
        # loadmat: metodo per leggere file .mat
        # il metodo loadmat restituisce la struttura dati python dizionario
        load = sio.loadmat(path_slices)
        pazienti = sio.loadmat(self.path_pazienti)

        # con la funzione .keys si leggono le chiavi degli elementi presenti nel dizionario. Ciascun elemento è
        # caratterizzato da una coppia chiave-valore e si può utilizzare la chiave per ottenere il valore corrispondente
        print(load.keys())
        print(pazienti.keys())

        # shape -> descrive quante DIMENSIONI il tensore ha lungo ogni ASSE
        # ndim -> conta quanti ASSI ha il tensore. Scalare -> tensore 0D, vettore -> tensore 1D, matrice -> tensore 2D

        info = pazienti['pazienti_new']  # salvataggio della struct pazienti in info
        data = load['slices_padding_' + view][0]  # salvataggio della struct slices in data

        self.data = data
        self.info = info
        # [0][0]: indici per righe e colonne

    def ID_paziente(self):
        # --------------------- Creazione lista ID_paziente ------------------------
        ID_paziente = []
        lab_paziente = []
        for idx in range(self.info.shape[0]):
            ID_paziente.append(self.info[idx][0][0])
            lab_paziente.append(self.info[idx][0][1][0])

        ID_paziente = np.array(ID_paziente)
        lab_paziente = np.array(lab_paziente)

        print("[INFO]-- Numero pazienti {}".format(ID_paziente.shape))

        ID_paziente_shuffle = copy.deepcopy(ID_paziente)
        lab_paziente_shuffle = copy.deepcopy(lab_paziente)
        ID_paziente_shuffle, lab_paziente_shuffle = self.shuffle_in_unison(ID_paziente_shuffle, lab_paziente_shuffle)

        return ID_paziente_shuffle, lab_paziente_shuffle

    # --- Creazione di tre liste che contengono rispettivamente le immagini, le label e l'ID_paziente per ogni slice ---
    def slices(self):
        slices = []
        labels = []
        ID_paziente_slice = []
        for idx in range(self.data.shape[0]):
            slices.append(img_to_array(self.data[idx][0], dtype='double'))  # immagini
            ID_paziente_slice.append(self.data[idx][1])  # ID_paziente
            labels.append(self.data[idx][2][0])  # delle labels

        # ------------------ Conversione da lista ad array numpy ------------------
        slices = np.array(slices)
        labels = np.array(labels)
        ID_paziente_slice = np.array(ID_paziente_slice)

        print("[INFO]-- Numero e dimensione slice {}".format(slices.shape))
        print(type(slices[1][0][0][0]))

        return slices, labels, ID_paziente_slice

    # Definire 'random.seed' pari ad un valore fisso garantisce che la stessa sequenza di numeri random siano generati
    # ogni volta che viene eseguito il codice. Questo permette di validare i risultati quando il codice è eseguito
    # più volte.
    def shuffle_in_unison(self, ID_paziente_shuffle, lab_paziente_shuffle):
        n_elem = ID_paziente_shuffle.shape[0]
        np.random.seed(42)
        indeces = np.random.choice(n_elem, size=n_elem, replace=False)
        return ID_paziente_shuffle[indeces], lab_paziente_shuffle[indeces]
