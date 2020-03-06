import os
import datetime
import numpy as np

from classi import Alexnet
from classi import Load
from classi import My_callbacks
from classi import KFold
from classi import Bootstrap

# --------------------------- Aggiunta ambiente virtuale tensorflow i path di Graphviz e CUPTI ------------------------
os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/extras/CUPTI/libx64'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/tensorflow/Lib/site-packages/graphviz'

# ------------------------------------- Definizione dei parametri della run ---------------------------------
augmented = 1
# fattore moltiplicativo per incrementare le immagini che passano attraverso la rete nel training (maggiore di
# 1 solo se "augmented" è settato a 1)
factor = 1
load = False
num_epochs = 100
batch = 128
l1 = 'none'
l2 = 'none'
lr = 0.0001
validation_method = 'bootstrap'
k = 8
boot_iter = 5
activation = 'relu'
optimiser = 'rmsprop'
initializer = 'xavier'
input_dim = (50, 50, 1)
batch_norm = True
slice_path = "ID2"

# ----------------------------------------------- Creazione del PATH --------------------------------------------
model = 'CNN_Alexnet'
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_folder = 'run\\{}\\'.format(model)
run_folder += '___'.join([time])
print(run_folder)
if not os.path.exists(run_folder):
    os.makedirs(run_folder)
    os.makedirs(os.path.join(run_folder, 'model'))
    os.makedirs(os.path.join(run_folder, 'plot'))

file = open(os.path.join(run_folder, "Parameters.txt"), "a")
file.write("Dataset: {}\n".format(slice_path))
file.write("Model: Alexnet\n")
file.write("Validation method: {}\n".format(validation_method))
file.write("Layer: 32, 32, 64, 64, 128, 128, 256 \n")
file.write("Kernel: 3, 3, 3, 3, 3, 3\n")
file.write("Regularization: {}, {}\n".format(l1, l2))
file.write("Activation: {}\n".format(activation))
file.write("Initializer: {}\n".format(initializer))
file.write("Dropout: 0.2, 0.3, 0.4, 0.5 \n")
file.write("Batch normalization: {}\n".format(batch_norm))
file.write("Learning rate: {}\n".format(lr))
file.write("Epoche: {}, Batch size: {}\n".format(num_epochs, batch))
file.write("Learning rate decay:{}\n".format(False))
if validation_method == 'bootstrap':
    file.write("Numero Fold: {}\n".format(boot_iter))
    file.write("Campioni di test estratti ad ogni fold: 10\n")
else:
    file.write("Numero Fold: {}\n".format(k))
file.write("Data Augmentation: {}, fattore moltiplicativo: {}\n".format(augmented, factor))
file.write("Data e ora di inizio simulazione: " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
file.close()

# ------------------------------------------ Caricamento dati -----------------------------------------------
# Creazione istanza della classe Load
load = Load(path_pazienti = "D:/Download/data/pazienti.mat")

view = 'layer'
load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path +"/Slices_data/"+ view +"/slices_resize_" + view + ".mat", view)
ID_paziente, label_paziente = load.ID_paziente()
slices_layer, labels_layer, ID_paziente_slice_layer = load.slices()

view = 'row'
load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path +"/Slices_data/"+ view +"/slices_resize_" + view + ".mat", view)
slices_row, labels_row, ID_paziente_slice_row = load.slices()

view = 'column'
load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path +"/Slices_data/"+ view +"/slices_resize_" + view + ".mat", view)
slices_column, labels_column, ID_paziente_slice_column = load.slices()

slices = np.concatenate((slices_layer, slices_row, slices_column), axis = 0)
labels = np.concatenate((labels_layer, labels_row, labels_column), axis = 0)
ID_paziente_slice = np.concatenate((ID_paziente_slice_layer, ID_paziente_slice_row, ID_paziente_slice_column), axis = 0)

# ------------------------------- Creazione istanze classe Alexnet e classe Callbacks ----------------------------------
alexnet = Alexnet(
                  input_dim = input_dim,
                  l1 = l1,
                  l2 = l2,
                  lr = lr,
                  cnn_activation = activation,
                  cnn_optimiser = optimiser,
                  cnn_initializer = initializer,
                  run_folder = run_folder,
                  batch_norm = batch_norm)

my_callbacks = My_callbacks(run_folder)
# ---------------------------------------- Creazione istanza classe Kfold ----------------------------------------------

if validation_method == 'bootstrap':
    bootstrap = Bootstrap(
        ID_paziente=ID_paziente,
        label_paziente=label_paziente,
        slices=slices,
        labels=labels,
        ID_paziente_slice=ID_paziente_slice,
        num_epochs=num_epochs,
        batch=batch,
        factor=factor,
        boot_iter=boot_iter,
        augmented=augmented,
        alexnet=alexnet,
        my_callbacks=my_callbacks,
        run_folder=run_folder,
        load=load)
    bootstrap.bootstrap_method()
else:
    kfold = KFold(
        ID_paziente=ID_paziente,
        label_paziente=label_paziente,
        slices=slices,
        labels=labels,
        ID_paziente_slice=ID_paziente_slice,
        num_epochs=num_epochs,
        batch=batch,
        factor=factor,
        k=k,
        augmented=augmented,
        alexnet=alexnet,
        my_callbacks=my_callbacks,
        run_folder=run_folder)
    kfold.k_fold_cross_validation()

file = open(os.path.join(run_folder, "Parameters.txt"), "a")
file.write("\nData e ora di fine simulazione: "  + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
file.close()