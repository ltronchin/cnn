from tensorflow.keras.initializers import RandomNormal, GlorotNormal, he_uniform
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import os

from tensorflow.keras import regularizers
class Alexnet():

    def __init__(self, input_dim, l1, l2, lr, cnn_activation, cnn_optimiser, cnn_initializer, run_folder, batch_norm):
        self.input_dim = input_dim
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.cnn_activation = cnn_activation
        self.cnn_optimiser = cnn_optimiser
        self.cnn_initializer = cnn_initializer
        self.run_folder = run_folder
        self.batch_norm = batch_norm

    def build_alexnet(self):
        # --------------------- Inizializzazione pesi -------------------------------
        kernel_init = self.get_kernel_initializer(self.cnn_initializer)
        # ----------------------- Definizione modello -----------------------------
        model = models.Sequential()
        # Convoluzione 1
        model.add(layers.Conv2D(32, (3, 3), kernel_initializer = kernel_init, padding='same', input_shape = self.input_dim))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Convoluzione 2
        model.add(layers.Conv2D(32, (3, 3), kernel_initializer = kernel_init, padding='same'))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Pooling 1
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))

        # Convoluzione 3
        model.add(layers.Conv2D(64, (3, 3), kernel_initializer = kernel_init, padding='same'))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Convoluzione 4
        model.add(layers.Conv2D(64, (3, 3), kernel_initializer = kernel_init, padding='same'))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Pooling 2
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))

        # Convoluzione 5
        model.add(layers.Conv2D(128, (3, 3), kernel_initializer = kernel_init, padding='same'))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Convoluzione 6
        model.add(layers.Conv2D(128, (3, 3), kernel_initializer = kernel_init, padding='same'))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        # Pooling 3
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())

        # Fully connected 1
        model.add(layers.Dense(128, kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(2, kernel_initializer = kernel_init, activation='softmax'))

        opt = self.get_opti(self.lr)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def get_activation(self, activation):

        if activation == 'leaky_relu':
            layer = layers.LeakyReLU(alpha=0.2)
            print('[INFO] -- Funzione di attivazione: leaky_relu\n')
        else:
            layer = layers.Activation(activation)
            print('[INFO] -- Funzione di attivazione: relu\n')
        return layer

    def get_kernel_initializer(self, initializer):
        if initializer == 'xavier':
            kernel = GlorotNormal(seed = 42)
            print('[INFO] -- Inizializzazione pesi: xavier\n')
        elif initializer == 'he_uniform':
            kernel = he_uniform(seed = 42)
            print('[INFO] -- Inizializzazione pesi: he_uniform\n')
        else:
            kernel = RandomNormal(mean = 0., stddev=0.02, seed = 42)
            print('[INFO] -- Inizializzazione pesi: random\n')
        return kernel

    def get_opti(self, lr):
        if self.cnn_optimiser == 'adam':
            opti = Adam(lr = lr)
            print('[INFO] -- Optimser: adam\n')
        elif self.cnn_optimiser == 'rmsprop':
            opti = RMSprop(lr = lr)
            print('[INFO] -- Optimiser: rmsprop\n')
        return opti

    def plot_model(self, run_folder, alexnet):
        plot_model(alexnet, to_file = os.path.join(run_folder, "model.png"), show_shapes='True', expand_nested='True',
                   dpi=1200)
