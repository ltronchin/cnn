from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classi.Net_cifar10 import Net_cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize input data in range between 0 and 1
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(y_train.shape)
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
activation = 'leaky_relu' # relu
optimiser = 'adam' #rmsprop
initializer = 'xavier'
input_dim = (32, 32, 3)
batch_norm = True

# ------------------------------- Creazione istanze classe Alexnet e classe Callbacks ----------------------------------
net_cifar10 = Net_cifar10(
                  input_dim = input_dim,
                  l1 = l1,
                  l2 = l2,
                  lr = lr,
                  cnn_activation = activation,
                  cnn_optimiser = optimiser,
                  cnn_initializer = initializer,
                  run_folder = '',
                  batch_norm = batch_norm)

train_datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2)
train_generator = train_datagen.flow(x_train, y_train, batch_size=32, shuffle=True, sample_weight=None)

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32,shuffle=True, sample_weight=None)

# Costruzione del modello
model = net_cifar10.build_alexnet()

# Fit del modello
history = model.fit(
                    train_generator,
                    steps_per_epoch=(x_train.shape[0] / 32),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=(x_test.shape[0] / 32),
                    sample_weight=None)

import numpy as np


CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)