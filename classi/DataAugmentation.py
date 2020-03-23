import scipy.io as sio
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

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

    def add_crop_slices(self, paziente_train, X_train, Y_train):
        X_aug = []
        Y_aug = []
        for n in range(paziente_train.shape[0]):
            for idx in range(self.slices.shape[0]):
                if self.ID_paziente_slice[idx] == paziente_train[n]:
                    X_aug.append(self.slices[idx])
                    Y_aug.append(self.labels[idx])

        X_aug = np.array(X_aug)
        Y_aug = np.array(Y_aug)

        Y_aug_categ = to_categorical(Y_aug, 2, dtype='uint8')

        X_train_aug = np.concatenate((X_train, X_aug), axis=0)
        Y_train_aug_categ = np.concatenate((Y_train, Y_aug_categ), axis=0)

        return X_train_aug, Y_train_aug_categ