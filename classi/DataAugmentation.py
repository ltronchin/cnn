import scipy.io as sio
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import copy
import xlsxwriter
import os

class DataAugmentation():

    def __init__(self, path_slices):

        load = sio.loadmat(path_slices)
        data = load['slices_resize_layer'][0]

        slices = []
        labels = []
        ID_paziente_slice = []
        for idx in range(data.shape[0]):
            slices.append(img_to_array(data[idx][0]))
            ID_paziente_slice.append(data[idx][1])
            labels.append(data[idx][2][0])

        slices = np.array(slices)
        labels = np.array(labels)
        ID_paziente_slice = np.array(ID_paziente_slice)

        self.slices = slices
        self.labels = labels
        self.ID_paziente_slice = ID_paziente_slice

    def augment_data(self, X_train_singleidd, Y_train_singleidd, ID_paziente_slice_train_singleidd, number_of_augmented, n_patient):

        N_rotate = 3
        N_shear = 2
        N_zoom = 2
        N_elastic = 1
        N_crop = 1
        N_flip = 1
        N_crop_zoom_shift = 5

        img_aug_single_slice = []
        # Lista slice augmented
        img_aug_total = []
        # Lista label augmented
        label_aug_total = []
        # Lista IDD augmented
        ID_aug_total = []

        datagen = ImageDataGenerator(fill_mode = 'constant', cval=0)
        n_clone = 0
        idd_clone = 'start'

        # Trasformazioni slice n-esima: N_rot (1 + N_zoom + N_shear + N_flip + N_elastic) + N_crop +  N_crop_zoom_shift
        total = N_rotate * (1 + N_zoom + N_shear + N_flip + N_elastic) + N_crop + N_crop_zoom_shift
        print("[INFO] -- Numero trasformazioni per slice {}".format(total))
        #print("[INFO] -- Numero trasformazioni totali: {}".format(total * X_train_singleidd.shape[0]))

        for n in range(0, X_train_singleidd.shape[0]):
            # Selezione di una slice
            img = X_train_singleidd[n]
            label = Y_train_singleidd[n]
            idd = ID_paziente_slice_train_singleidd[n]

            if idd_clone != idd:
                idd_clone = idd
                n_clone = n
                new_paziente = 1

            #print("Numero slice: {}".format(n))
            #print("Numero prima slice paziente: {}".format(n_clone))

            # CROP
            img_crop = self.crop_slices(idd, n, n_clone)
            img_aug_single_slice.append(img_crop)
            img_aug_total.append(img_crop)
            label_aug_total.append(label)
            ID_aug_total.append(idd)

            # Ad slice 2D è stato effettuato il crop sulla lezione per poi effettuare un resize alla dimensione della
            # slice 2D di partenza. A slice_crop si applica uno zoom out del 60% e poi delle traslazioni randomiche.
            # Questo tipo di precedimento serve per centrare la lesione al centro dell'immagine impedendo così che
            # applicando le traslazioni la lesione "esca" dal campo visivo
            # CROP + ZOOM_OUT + SHIFT
            for idx in range(0, N_crop_zoom_shift):
                img_crop_zoomout_shift = self.crop_zoom_shift_slice(img_crop, datagen)
                img_aug_single_slice.append(img_crop_zoomout_shift)
                img_aug_total.append(img_crop_zoomout_shift)
                label_aug_total.append(label)
                ID_aug_total.append(idd)

            # Ciclo per ruotare la slice: ogni slice è ruotata in senso antiorario di un angolo campionato in modo
            # randomico tra 10 e 175
            for idx in range(0, N_rotate):
                # ROTATE
                img_rotate = (self.rotate_slice(img, datagen))
                # Aggiunta della slice ruotata alla lista per SLICE
                img_aug_single_slice.append(img_rotate)
                # Aggiunta della slice ruotata alla lista TOTALE
                img_aug_total.append(img_rotate)
                label_aug_total.append(label)
                ID_aug_total.append(idd)

                # Per ogni slice ruotata si applicano zoom, shear, flip, elastic deformation
                # ZOOM
                for idx_zoom in range(0, N_zoom):
                    img_zoom = self.zoom_slice(img_rotate, datagen)
                    img_aug_single_slice.append(img_zoom)
                    img_aug_total.append(img_zoom)
                    label_aug_total.append(label)
                    ID_aug_total.append(idd)

                # SHEAR
                for idx_shear in range(0, N_shear):
                    img_shear = self.shear_slice(img_rotate, datagen)
                    img_aug_single_slice.append(img_shear)
                    img_aug_total.append(img_shear)
                    label_aug_total.append(label)
                    ID_aug_total.append(idd)

                # FLIP
                for idx_shear in range(0, N_flip):

                    img_flip = self.flip_slice(img_rotate, datagen)
                    img_aug_single_slice.append(img_flip)
                    img_aug_total.append(img_flip)
                    label_aug_total.append(label)
                    ID_aug_total.append(idd)

                # ELASTIC DEFORMATION
                for idx_shear in range(0, N_elastic):
                    img_elastic = self.elastic_transform(img, 60, 5, random_state=None)
                    img_aug_single_slice.append(img_elastic)
                    img_aug_total.append(img_elastic)
                    label_aug_total.append(label)
                    ID_aug_total.append(idd)

            # Plot delle trasformazioni per la slice n-esima
            if new_paziente == 1:
                #self.plot_aug(img, img_aug_single_slice, total, n_patient)
                new_paziente = 0

            img_aug_single_slice = []

        # Trasformazione della lista ad array numpy
        img_aug_total = np.array(img_aug_total)
        label_aug_total = np.array(label_aug_total)
        ID_aug_total = np.array(ID_aug_total)
        print("[INFO] -- Numero totale trasformazioni: {}, numero totale label: {}, numero totale id: {}".format(img_aug_total.shape[0], label_aug_total.shape[0], ID_aug_total.shape[0]))

        # Selezione randomica delle slice
        X_aug, Y_aug, idd_aug = self.shuffle_in_unison(number_of_augmented, img_aug_total, label_aug_total, ID_aug_total)

        #self.write_excel(Y_aug, idd_aug, n_patient)

        return X_aug, Y_aug, idd_aug

    def rotate_slice(self, img, datagen):
        # theta: angolo (in senso antiorario), campionato in modo randomico da una distribuzione uniforme con
        # minimi pari a min_rotation_range e massimo pari a maxi_rotation_range
        theta = np.random.uniform(10,175)
        #print("Angolo: {}".format(theta))
        return datagen.apply_transform(x = img, transform_parameters={'theta':theta})

    def flip_slice(self, img, datagen):
        return datagen.apply_transform(x = img, transform_parameters={'flip_horizontal':'true'})

    def crop_zoom_shift_slice(self, img, datagen):
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
        #print("Shift: {}, {}".format(tx, ty))
        # zoom out del 60%
        img_zoom_out = datagen.apply_transform(x=img, transform_parameters={'zx': 1.6, 'zy': 1.6})
        img_shift = datagen.apply_transform(x=img_zoom_out, transform_parameters={'tx': tx, 'ty': ty})
        return img_shift

    def crop_slices(self, idd, n, n_clone):

        idx = 0
        find = 'not_find'
        while (idx <= self.slices.shape[0] and find != 'find'):
            # Paziente trovato, ora bisogno ricercare la slice corretta per quel paziente
            # idx -- indice sui pazienti
            if self.ID_paziente_slice[idx] == idd:
                find = 'find'

                # Se n = n_clone si deve selezionare la prima slice del nuovo ID paziente
                if n == n_clone:
                    img_crop = self.slices[idx]
                    idd_crop = self.ID_paziente_slice[idx]
                # Se l'if non è verificato la slice da selezionare non è la prima ma è quella in posizione
                # idx + n - n_clone
                else:
                    img_crop = self.slices[idx + (n - n_clone)]
                    idd_crop =self.ID_paziente_slice[idx + (n - n_clone)]

            idx += 1

        #print("idd:{}".format(idd))
        #print("idd_crop:{}".format(idd_crop))

        return img_crop

    def zoom_slice(self, img, datagen):
        zoomxy = np.random.uniform(0.65, 1.35)
        #print("Zoom: {}".format(zoomxy))
        # zx e zy: zoom rispettivamente lungo le colonne e lungo le righe. Se valgono meno
        # di 1 l'immagine viene ingrandita altrimenti rimpicciolita
        return datagen.apply_transform(x = img, transform_parameters={'zx':zoomxy, 'zy':zoomxy})

    def shear_slice(self, img, datagen):
        shear = np.random.uniform(25, 40)
        #print("Shear: {}".format(shear))
        return datagen.apply_transform(x = img, transform_parameters={'shear':shear})

    # Funzione per applicare una distorsione all'immagine
    def elastic_transform(self,image, alpha_range, sigma, random_state=None):
        """
           Argomenti funzione:
              image: array numpy con dimensioni (altezza, larghezza, canali)

              alpha range: float per un valore fisso o [minimo, massimo] per
              campionare un valore random da una distribuzione uniforme

              Parametri per il controllo della deformazione
              sigma: float, deviazione standard del filtro Gaussiano
              che modifica la griglia di spostamenti
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        # Verifica se alpha range è un valore float o un intervallo
        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low = alpha_range[0], high = alpha_range[1])

        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=3, mode='reflect').reshape(shape)

    def plot_aug(self, img, img_aug_total, total, n_patient):

        fig = plt.figure(figsize=(12, 12))
        plt.subplot(6, 6, 1)
        plt.axis('off')
        plt.imshow(array_to_img(img), cmap='gray')
        for idx in range(total):
            plt.subplot(6, 6, idx +2)
            plt.axis('off')
            plt.imshow(array_to_img(img_aug_total[idx]), cmap='gray')

        fig.savefig('augmented_{}.png'.format(n_patient))

    def shuffle_in_unison(self,n_total_aug, X, Y, idd):
        indeces = np.random.choice(X.shape[0], size=n_total_aug, replace=False)
        return X[indeces], Y[indeces], idd[indeces]

    def write_excel(self, lab_aug, idd_aug, n_patient):

        # Creazione di un nuovo file Excel e aggiunta di un foglio di lavoro
        workbook_aug = xlsxwriter.Workbook("aug_{}.xlsx".format(n_patient))
        worksheet_aug = workbook_aug.add_worksheet()

        # Allargamento della prima e seconda colonna
        worksheet_aug.set_column('A:B', 20)

        # Aggiunta del formato grassetto
        bold_val = workbook_aug.add_format({'bold': True})

        worksheet_aug.write('A1', 'Augmented Label', bold_val)
        worksheet_aug.write('B1', 'Augmented ID Paziente', bold_val)

        for i in range(lab_aug.shape[0]):
            worksheet_aug.write(i + 1, 0, lab_aug[i, 0])
            worksheet_aug.write(i + 1, 1, idd_aug[i, 0])

        workbook_aug.close()
