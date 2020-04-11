from tensorflow.keras.initializers import RandomNormal, GlorotNormal, he_uniform
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import os

from tensorflow.keras import regularizers
class Alexnet():

    def __init__(self,
                 input_dim,
                 l1,
                 l2,
                 lr,
                 cnn_activation,
                 cnn_optimiser,
                 cnn_initializer,
                 run_folder,
                 batch_norm,
                 drop,
                 drop_list,
                 filter_list,
                 kernel_size,
                 padding,
                 regularizer):

        self.input_dim = input_dim
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.cnn_activation = cnn_activation
        self.cnn_optimiser = cnn_optimiser
        self.cnn_initializer = cnn_initializer
        self.run_folder = run_folder
        self.batch_norm = batch_norm
        self.drop = drop
        self.drop_list = drop_list
        self.filter_list = filter_list
        self.kernel_size = kernel_size
        self.padding = padding
        self.regularizer = regularizer

    def build_alexnet(self):
        # --------------------- Inizializzazione pesi -------------------------------
        kernel_init = self.get_kernel_initializer(self.cnn_initializer)
        # ----------------------- Definizione modello -----------------------------
        model = models.Sequential()
        # Convoluzione - attivazione - batch normalization - dropout
        model.add(layers.Conv2D(self.filter_list[0],
                                (self.kernel_size, self.kernel_size),
                                kernel_initializer = kernel_init,
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding = self.padding,
                                input_shape = self.input_dim))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[0] == True:
            model.add(layers.Dropout(self.drop_list[0]))
        # Convoluzione - attivazione - batch normalization - dropout - pooling
        model.add(layers.Conv2D(self.filter_list[1],
                                (self.kernel_size, self.kernel_size),
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding=self.padding,
                                kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[1] == True:
            model.add(layers.Dropout(self.drop_list[1]))
        model.add(layers.MaxPooling2D((2, 2)))

        # Convoluzione - attivazione - batch normalization - dropout
        model.add(layers.Conv2D(self.filter_list[2],
                                (self.kernel_size, self.kernel_size),
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding=self.padding,
                                kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[2] == True:
            model.add(layers.Dropout(self.drop_list[2]))
        # Convoluzione - attivazione - batch normalization - dropout - pooling
        model.add(layers.Conv2D(self.filter_list[3],
                                (self.kernel_size, self.kernel_size),
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding=self.padding,
                                kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[3] == True:
            model.add(layers.Dropout(self.drop_list[3]))
        model.add(layers.MaxPooling2D((2, 2)))

        # Convoluzione - attivazione - batch normalization - dropout
        model.add(layers.Conv2D(self.filter_list[4],
                                (self.kernel_size, self.kernel_size),
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding=self.padding,
                                kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[4] == True:
            model.add(layers.Dropout(self.drop_list[4]))
        # Convoluzione - attivazione - batch normalization - dropout - pooling
        model.add(layers.Conv2D(self.filter_list[5],
                                (self.kernel_size, self.kernel_size),
                                kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                                padding=self.padding,
                                kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[5] == True:
            model.add(layers.Dropout(self.drop_list[5]))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        # Fully connected - attivazione - batch normalization - dropout
        model.add(layers.Dense(self.filter_list[6],
                               kernel_regularizer= self.get_regularizers(self.l2, self.l1),
                               kernel_initializer = kernel_init))
        model.add(self.get_activation(self.cnn_activation))
        if self.batch_norm == True:
            model.add(layers.BatchNormalization())
        if self.drop[6] == True:
            model.add(layers.Dropout(self.drop_list[6]))

        model.add(layers.Dense(1, kernel_initializer = kernel_init, activation='sigmoid'))

        opt = self.get_opti(self.lr)

        model.compile(loss ='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
        model.summary()

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
            opti = RMSprop(lr = lr, momentum = 0.9)
            print('[INFO] -- Optimiser: rmsprop\n')
        return opti

    def get_regularizers(self, l2, l1):
        if self.regularizer == 'l2':
            kernel_regularizer = regularizers.l2(l2),
            print('[INFO] -- regularizers: l2\n')
        elif self.regularizer == 'l1':
            kernel_regularizer = regularizers.l1(l1),
            print('[INFO] -- regularizers: l1\n')
        elif self.regularizer == 'l2_l1':
            kernel_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
            print('[INFO] -- regularizers: l1_l2\n')
        else:
            kernel_regularizer = None
            print('[INFO] -- regularizers: None \n')

        return kernel_regularizer

    def plot_model(self, run_folder, alexnet):
        plot_model(alexnet, to_file = os.path.join(run_folder, "model.png"), show_shapes='True', expand_nested='True',
                   dpi=1200)
