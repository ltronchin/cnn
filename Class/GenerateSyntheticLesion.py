import numpy as np
import math

from tensorflow.keras import models

class GenerateSyntheticLesion():
    def __init__(self,
                 n_synthetic_slice,
                 balance_class,
                 path_generator_adaptive,
                 path_generator_not_adaptive):

      self.n_synthetic_slice = n_synthetic_slice # numero di slice da sintetizzare ed aggiungere al set di
      # training, se None nessuna campione sintetico viene aggiunto
      self.balance_class = balance_class # se True si vogliono bilanciare le classi del dataset di training

      self.z = 128

      # Caricamento modelli
      self.generator_adaptive = models.load_model(path_generator_adaptive)
      self.generator_not_adaptive = models.load_model(path_generator_not_adaptive)

      self.synthetic_slice_list = []
      self.synthetic_label_list = []

    def adaptive_lesion(self, n_lesion):
        # Metodo per generare lesioni sintetiche di paziente adattivi ed etichettarle
        # adaptive -- label pari a  1
        noise = np.random.normal(0, 1, (n_lesion, self.z))
        synthetic_slice = self.generator_adaptive.predict(noise)
        # Conversione a double delle immagini generate
        synthetic_slice = synthetic_slice.astype('double')
        # Creazione etichette, adaptive -- 1
        synthetic_label = np.ones((n_lesion, 1), dtype=np.uint8)

        self.synthetic_slice_list.append(synthetic_slice)
        self.synthetic_label_list.append(synthetic_label)
        print('Generate %d slice adaptive, etichette: %d' % (synthetic_slice.shape[0], synthetic_label.shape[0]))

    def not_adaptive_lesion(self, n_lesion):
        # Metodo per generare lesioni sintetiche di paziente non adattivi ed etichettarle
        # not adaptive -- label pari a  0
        noise = np.random.normal(0, 1, (n_lesion, self.z))
        synthetic_slice = self.generator_not_adaptive.predict(noise)
        # Conversione a double
        synthetic_slice = synthetic_slice.astype('double')
        # Creazione etichette, non adaptive -- 0
        synthetic_label = np.zeros((n_lesion, 1), dtype=np.uint8)

        self.synthetic_slice_list.append(synthetic_slice)
        self.synthetic_label_list.append(synthetic_label)
        print('Generate %d slice not adaptive, etichette: %d' % (synthetic_slice.shape[0], synthetic_label.shape[0]))

    def count_label_train(self, Y_train):
        # Metodo per contare quanti slice sono adaptive e quante slice non sono adaptive
        self.adaptive = 0
        self.not_adaptive = 0
        for label in Y_train:
            if (label == 1):
                self.adaptive += 1
            else:
                self.not_adaptive += 1

        print('\nSlice adaptive: %d' % (self.adaptive))
        print('Slice not adaptive: %d\n' % (self.not_adaptive))

    def synthetize_lesion(self):
        if self.balance_class == True:
            if self.not_adaptive > self.adaptive:
                unbalanced_value = self.not_adaptive - self.adaptive
                print("Dataset di training sbilanciato: {} non adattivi"
                      " e {} adattivi,"
                      " differenza: {}".format(self.not_adaptive,self.adaptive, self.not_adaptive - self.adaptive))
                self.adaptive_lesion(unbalanced_value)
            elif self.not_adaptive < self.adaptive:
                unbalanced_value = self.adaptive - self.not_adaptive
                print("Dataset di training sbilanciato: {} non adattivi"
                      " e {} adattivi,"
                      " differenza: {}".format(self.not_adaptive, self.adaptive, self.adaptive - self.not_adaptive))
                self.not_adaptive_lesion(unbalanced_value)
            else:
                print("Dataset di training non sbilanciato! Yeahs!")

        if self.n_synthetic_slice is not None:
            print("Aggiunta di {} slice".format(self.n_synthetic_slice))
            self.adaptive_lesion(math.ceil(self.n_synthetic_slice / 2))
            self.not_adaptive_lesion(math.ceil(self.n_synthetic_slice / 2))

        X_train_synthetic = np.concatenate(self.synthetic_slice_list, axis=0)
        Y_train_synthetic = np.concatenate(self.synthetic_label_list, axis=0)

        # Vengono portati i valori dei pixel delle immagini generate tra 0 e 1
        X_train_synthetic = 0.5 * (X_train_synthetic + 1)
        #X_train_synthetic = np.clip(X_train_synthetic, 0, 1)

        print('Immagini generate: {}, label: {}'.format(X_train_synthetic.shape, Y_train_synthetic.shape))

        return X_train_synthetic, Y_train_synthetic