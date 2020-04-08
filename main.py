import os
import datetime
import numpy as np

from classi.Alexnet import Alexnet
from classi.Load import Load
from classi.My_callbacks import My_callbacks
from classi.Run_net import Run_net
from classi.EvaluateConvNet import EvaluateConvNet

# --------------------------- Aggiunta ambiente virtuale tensorflow i path di Graphviz e CUPTI ------------------------
os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/extras/CUPTI/libx64'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/tensorflow/Lib/site-packages/graphviz'

# ------------------------------------- Definizione dei parametri della run ---------------------------------
augmented = 0
#fill_mode_list = ['constant', 'reflect', 'nearest']
fill_mode = 'constant'
load = False
num_epochs = 250
batch = 128
l1 = 'none'
l2 = 'none' #'True: 0.001'
lr = 0.001
validation_method = 'kfold' #bootstrap
# Parametri kfold
k = 10
# Parametri per bootstrap
n_patient_test = 10
boot_iter = 15
activation = 'leaky_relu' #relu
optimiser = 'adam'  #rmsprop
initializer = 'xavier'
input_dim = (80, 80, 1)
batch_norm = True
allview = False
view = 'layer'
lr_decay = False

drop = [True, True, True, True, True, True, True]
drop_list = [0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 0.50]

filter_list = [32, 32, 64, 64, 128, 128, 256]
kernel_size = 3

slice_path_list = ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8"]
#slice_path_list = ["ID5"]

for slice_path_ID in slice_path_list:
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
        os.makedirs(os.path.join(run_folder, 'data'))
        os.makedirs(os.path.join(run_folder, 'data_pazienti'))

    file = open(os.path.join(run_folder, "Parameters.txt"), "a")
    file.write("Dataset: {}\n".format(slice_path_ID))
    if allview == True:
        file.write("Allview: {}\n".format(allview))
    else:
        file.write("View: {}\n".format(view))
    file.write("Model: Alexnet\n")
    file.write("Validation method: {}\n".format(validation_method))
    file.write("Layer: {} \n".format(filter_list))
    file.write("Kernel_size: {}\n".format(kernel_size))
    file.write("L1_L2_regularization: {}, {}\n".format(l1, l2))
    file.write("Activation: {}\n".format(activation))
    file.write("Optimiser: {}\n".format(optimiser))
    file.write("Initializer: {}\n".format(initializer))
    file.write("Dropout: {}\n".format(drop))
    file.write("Dropout_value: {}\n".format(drop_list))
    file.write("Batch normalization: {}\n".format(batch_norm))
    file.write("Learning rate: {}\n".format(lr))
    file.write("Epoche: {}, Batch size: {}\n".format(num_epochs, batch))
    file.write("Learning rate decay:{}\n".format(lr_decay))
    if validation_method == 'bootstrap':
        file.write("Numero iterazioni bootstrap: {}\n".format(boot_iter))
        file.write("Campioni di test estratti ad ogni iterazione: {}\n".format(n_patient_test))
    else:
        file.write("Numero Fold: {}\n".format(k))
    file.write("Data Augmentation: {}\n".format(augmented))
    file.write("Fill mode: {}\n".format(fill_mode))
    file.write("Data e ora di inizio simulazione: " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    file.close()

    # ------------------------------------------ Caricamento dati -----------------------------------------------
    # Creazione istanza della classe Load
    load = Load(path_pazienti = "D:/Download/data/pazienti_new.mat")
    if allview == True:
        view = 'layer'
        load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        ID_paziente, label_paziente = load.ID_paziente()
        slices_layer, labels_layer, ID_paziente_slice_layer = load.slices()

        view = 'row'
        load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        slices_row, labels_row, ID_paziente_slice_row = load.slices()

        view = 'column'
        load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        slices_column, labels_column, ID_paziente_slice_column = load.slices()

        slices = np.concatenate((slices_layer, slices_row, slices_column), axis = 0)
        labels = np.concatenate((labels_layer, labels_row, labels_column), axis = 0)
        ID_paziente_slice = np.concatenate((ID_paziente_slice_layer, ID_paziente_slice_row, ID_paziente_slice_column), axis = 0)
    else:
        load.read_from_path("D:/Download/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        ID_paziente, label_paziente = load.ID_paziente()
        slices, labels, ID_paziente_slice = load.slices()

    # ------------------------------- Creazione istanze classe Alexnet e classe Callbacks ----------------------------------
    alexnet = Alexnet(input_dim = input_dim,
                      l1 = l1,
                      l2 = l2,
                      lr = lr,
                      cnn_activation = activation,
                      cnn_optimiser = optimiser,
                      cnn_initializer = initializer,
                      run_folder = run_folder,
                      batch_norm = batch_norm,
                      drop = drop,
                      drop_list = drop_list,
                      filter_list = filter_list,
                      kernel_size = kernel_size)

    my_callbacks = My_callbacks(run_folder)

    # ---------------------------------------- Creazione istanza classe Kfold ----------------------------------------------
    run_net = Run_net(validation_method = validation_method,
                      ID_paziente=ID_paziente,
                      label_paziente=label_paziente,
                      slices=slices,
                      labels=labels,
                      ID_paziente_slice=ID_paziente_slice,
                      num_epochs=num_epochs,
                      batch=batch,
                      boot_iter=boot_iter,
                      k_iter = k,
                      n_patient_test=n_patient_test,
                      augmented=augmented,
                      fill_mode=fill_mode,
                      alexnet=alexnet,
                      my_callbacks=my_callbacks,
                      run_folder=run_folder,
                      load=load,
                      lr_decay = lr_decay)

    run_net.run()

    validation_method = 'kfold'
    if validation_method == 'kfold':
        n_iter = k
    else:
        n_iter = boot_iter

    evalConv = EvaluateConvNet(run_folder=run_folder)
    evalConv.use_best_model_on_val_set(n_iter)

    file = open(os.path.join(run_folder, "Parameters.txt"), "a")
    file.write("\nData e ora di fine simulazione: "  + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    file.close()
