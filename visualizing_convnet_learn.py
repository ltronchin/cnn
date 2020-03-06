from tensorflow.keras import models
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

k = 2
path = 'C:/Users/User/PycharmProjects/Local/CNN/run/CNN_Alexnet1/19-02/2020-02-19_22-55-32/model/'
model = models.load_model(path + 'model_end_of_train_{}.h5'.format(k))
print(model.summary())

image = sio.loadmat('C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/ID1/Slices_data/layer/non adaptive/51236_5.mat')
print(image.keys())
img = image['slice_resize']
print(img.shape, type(img))
#plt.imshow(img, cmap = 'gray')
plt.show()

img_array = img_to_array(img)
print(img_array.shape, type(img_array))

img_tensor = img.reshape((1,) + img_array.shape)
print(img_tensor.shape, type(img_tensor))

# model.layers -> lista dei layer del modello
print(len(model.layers))

# Estraggo gli output (feature map) dei primi 24 layer
layer_outputs = [layer.output for layer in model.layers[:24]]
print(len(layer_outputs))

# model.input -> lista dei tensori in input del modello
# model.outputs -> lista dei tensori in output del modello
# Creazione di un modello che restituisce gli output dei primi 24 layer dato un tensore in input (immagine)
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

# Restituisce la lista di 24 array numpy, un array per layer activation
activations = activation_model.predict(img_tensor)

# Attivazione del primo layer di convoluzione
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# Plot dell'attivazione del quarto canale del primo layer di convoluzione
plt.matshow(first_layer_activation[0, :, :, 4], cmap ='gray')
plt.show()

# Salvataggio in una lista del nome dei layer
layer_names = []
for layer in model.layers[:24]:
    layer_names.append(layer.name)
print(layer_names)

# Numero di immagini per riga che si vogliono nel plot finale
images_per_row = 16
# Ciclo sui vari layer del modello, ad ogni iterazione viene selezionato un layer
for layer_names, layer_activation in zip(layer_names, activations):

    # Ogni feature map ha dimensione (1, size, size, n_features)
    n_features = layer_activation.shape[-1] # numero di feature nella feature map
    size = layer_activation.shape[1]

    # Il numero di righe per il corrente layer è determinato dal numero di feature e dal numero di immagini per riga (colonne)
    # Ad esempio se n_features = 32 e images_per_row 16 si avrà bisogno di 2 righe
    n_row = n_features // images_per_row

    # Creazione di una griglia per salvare le feature estratte dall'immagine: per l'attivazione del primo layer si hanno 32
    # feature in uscita, ciascuna di dimensione dell'immagine -> si avranno 32 feature 64x64, quindi la griglia dovrà
    # avere size*n_row righe (128 righe) e size*images_per_row colonne (1024 colonne) in modo da poter contenere globalmente
    # 32 feature
    display_grid = np.zeros((size * n_row, size * images_per_row))

    for row in range(n_row):
        for col in range(images_per_row):
            channel_image = layer_activation[0, :, :, row * images_per_row + col]
            display_grid[row * size : (row + 1) * size, col * size : (col + 1) * size] = channel_image

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

    scale = 1./ size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_names)
    plt.imshow(display_grid, aspect = 'auto', cmap = 'gray')
    plt.show()