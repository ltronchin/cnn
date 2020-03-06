from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

from classi.Alexnet import Alexnet
import matplotlib.pyplot as plt


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize input data in range between 0 and 1
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
# We will represent y_train and y_test with two matrix
# with dimension, respectevely, [50000, 10] and [10000, 10].
# to_categorical convert a class vector to binary class matrix
NUM_CLASSES = 10
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# ------------------------------------- Definizione dei parametri della run ---------------------------------

l1 = 'none'
l2 = 'none'
lr = 0.0002
activation = 'relu' # leaky_relu
optimiser = 'rmsprop' # adam
initializer = 'xavier'
input_dim = (32, 32, 3)
batch_norm = True

# ------------------------------- Creazione istanze classe Alexnet e classe Callbacks ----------------------------------
alexnet = Alexnet(input_dim = input_dim,
                  l1 = l1,
                  l2 = l2,
                  lr = lr,
                  cnn_activation = activation,
                  cnn_optimiser = optimiser,
                  cnn_initializer = initializer,
                  run_folder = '',
                  batch_norm = batch_norm)

train_datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1)
train_generator = train_datagen.flow(x_train, y_train, batch_size=64, shuffle=True, sample_weight=None)
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(x_test, y_test, batch_size=64,shuffle=True, sample_weight=None)

# Costruzione del modello
model = alexnet.build_alexnet()

# Fit del modello
history = model.fit(train_generator,
                    steps_per_epoch=(x_train.shape[0] / 64),
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=(x_test.shape[0] / 64),
                    sample_weight=None)

summarize_diagnostics(history)