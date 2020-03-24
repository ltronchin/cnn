import scipy.io as sio
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

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

    def augment_data(self, X_train, Y_train, ID_paziente_slice_train):

        N_rotate = 1
        N_shear = 1
        N_zoom = 1
        img_rotate_list = []
        img_aug_total = []
        label_aug_total = []

        datagen = ImageDataGenerator(fill_mode='constant', cval=0)

        for n in range(0, 1):
        #for n in range(X_train.shape[0]):
            # Selezione di una slice
            img = X_train[n]
            label = Y_train[n]
            # Ciclo per ruotare la slice: creo una lista di lunghezza N_rotate in cui ogni elemento contiene la slice
            # ruotata in senso antiorario di un angolo campionato in modo randomico tra 10 e 175
            for idx in range(0, N_rotate):
                img_rotate_list.append(self.rotate_slice(img, datagen))
                img_aug_total.append(img_rotate_list[idx])
                label_aug_total.append(Y_train[n])
                # Per ogni slice ruotata si applicano flip, shear e zoom
                for idx_rot in range(0, N_zoom):
                    img_aug_total.append(self.zoom_slice(img_rotate_list[idx], datagen))
                    label_aug_total.append(Y_train[n])
                for idx_rot in range(0, N_shear):
                    img_aug_total.append(self.shear_slice(img_rotate_list[idx], datagen))
                    label_aug_total.append(Y_train[n])

                img_aug_total.append(self.flip_slice(img_rotate_list[idx], datagen))
                label_aug_total.append(Y_train[n])

            img_aug_total.append(self.elastic_transform(img, 70, 5, random_state=None))
            label_aug_total.append(Y_train[n])
            img_aug_total.append(self.crop_slices(ID_paziente_slice_train, n))
            label_aug_total.append(Y_train[n])

            img_aug_total = np.array(img_aug_total)
            label_aug_total = np.array(label_aug_total)

            plt.subplot(2, 3, 1)
            plt.imshow(array_to_img(img), cmap='gray')

            plt.subplot(2, 3, 2)
            plt.imshow(array_to_img(img_aug_total[0]), cmap='gray')

            plt.subplot(2, 3, 2)
            plt.imshow(array_to_img(img_aug_total[1]), cmap='gray')

            plt.subplot(2, 3, 3)
            plt.imshow(array_to_img(img_aug_total[2]), cmap='gray')

            plt.subplot(2, 3, 4)
            plt.imshow(array_to_img(img_aug_total[3]), cmap='gray')

            plt.subplot(2, 3, 5)
            plt.imshow(array_to_img(img_aug_total[4]), cmap='gray')

            plt.subplot(2, 3, 6)
            plt.imshow(array_to_img(img_aug_total[5]), cmap='gray')

    def rotate_slice(self, img, datagen):
        # theta: angolo (in senso antiorario), campionato in modo randomico da una distribuzione uniforme con
        # minimi pari a min_rotation_range e massimo pari a maxi_rotation_range
        theta = np.random.uniform(45)
        return datagen.apply_transform(x = img, transform_parameters={'theta':theta})

    def zoom_slice(self, img, datagen):
        zoomxy = np.random.uniform(0.75, 1.25)
        # zx e zy: zoom rispettivamente lungo le colonne e lungo le righe. Se valgono meno
        # di 1 l'immagine viene ingrandita altrimenti rimpicciolita
        return datagen.apply_transform(x = img, transform_parameters={'zx':zoomxy, 'zy':zoomxy})

    def shear_slice(self, img, datagen):
        shear = np.random.uniform(20, 25)
        return datagen.apply_transform(x = img, transform_parameters={'shear':shear})

    def flip_slice(self, img, datagen):
        return datagen.apply_transform(x = img, transform_parameters={'flip_horizontal':'true'})

    def crop_slices(self, ID_paziente_slice_train, n):
        X_crop = []
        Y_crop = []

        for idx in range(ID_paziente_slice_train.shape[0]):
            for i in range(self.slices.shape[0]):
                if self.ID_paziente_slice[i] == ID_paziente_slice_train[idx]:
                    X_crop.append(self.slices[idx])
                    Y_crop.append(self.labels[idx])

        X_crop = np.array(X_crop)
        Y_crop = np.array(Y_crop)

        img_crop = X_crop[n]

        return img_crop

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