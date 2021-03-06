import os
import datetime
import numpy as np

from Models.Alexnet import Alexnet
from Utils.Load import Load
from Callbacks.My_callbacks import My_callbacks
from Class.Run_net import Run_net
from Class.EvaluateConvNet import EvaluateConvNet

# --------------------------- Aggiunta ambiente virtuale tensorflow i path di Graphviz e CUPTI ------------------------
os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/extras/CUPTI/libx64'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/tensorflow/Lib/site-packages/graphviz'

# ------------------------------------- Definizione dei parametri della run ---------------------------------

augmented = True # False
augmentation = ['rotazione, shift, flip']
elastic_deformation = False
#fill_mode_list = ['constant', 'reflect', 'nearest']
fill_mode = 'constant'
load = False
num_epochs = 500 #250
batch = 128
regularizer = None #l2 #l1 #l2_l1
l1 = None #0.001
l2 = None #0.001
lr = 0.001 # 0.0001 # 0.0005
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
allview = False # True
view = 'layer'
lr_decay = False

slice_path_ID = "ID8"

# Sezione WGAN
WGAN_lesion = True # False
#n_of_lesion2add = 2500 # 5000, 7500, 10000 none
n_of_lesion2add_list = [None, 2500, 5000, 7500, 10000]
balance_training_data = True # False
path_generator_adaptive = 'D:/Documenti/Tesi/Run/run/gan/WGAN-resnet/Data augmentation/003/adaptive/generator.h5'
path_generator_not_adaptive = 'D:/Documenti/Tesi/Run/run/gan/WGAN-resnet/Data augmentation/003/non_adaptive/generator_not_adaptive.h5'

drop = [True, True, True, True, True, True, True]
drop_list = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5]
filter_list = [32, 32, 64, 64, 128, 128, 256]
padding = 'valid' #same
kernel_size = 3

for n_of_lesion2add in n_of_lesion2add_list:
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
    file.write("Padding: {} \n".format(padding))
    file.write("Kernel_size: {}\n".format(kernel_size))
    file.write("Regularization: {} \n".format(regularizer))
    file.write("Activation: {}\n".format(activation))
    file.write("Optimiser: {}\n".format(optimiser))
    file.write("Initializer: {}\n".format(initializer))
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

    file.write("Data Augmentation: {}, trasformazioni: {}\n".format(augmented, augmentation))
    file.write("Sintesi lesioni con WGAN: {}\n".format(WGAN_lesion))
    file.write("Bilanciamento classi tramite lesione sintetiche: {}\n".format(balance_training_data))
    file.write("Numero di slice aggiunte: {}\n".format(n_of_lesion2add))

    file.write("Fill mode: {}\n".format(fill_mode))
    file.write("Data e ora di inizio simulazione: " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    file.close()

    # ------------------------------------------ Caricamento dati -----------------------------------------------
    # Creazione istanza della classe Load
    load = Load(path_pazienti = "D:/Download/data/pazienti_new.mat")

    if allview == True:
        view = 'layer'
        load.read_from_path("C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        ID_paziente, label_paziente = load.ID_paziente()
        slices_layer, labels_layer, ID_paziente_slice_layer = load.slices()

        view = 'row'
        load.read_from_path("C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
        slices_row, labels_row, ID_paziente_slice_row = load.slices()

        view = 'column'
        load.read_from_path("C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/"+ slice_path_ID +"/Slices_data/"+ view +"/slices_padding_" + view + ".mat", view)
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
                      kernel_size = kernel_size,
                      padding = padding,
                      regularizer = regularizer)

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
                      elastic_deformation = elastic_deformation,
                      fill_mode=fill_mode,
                      alexnet=alexnet,
                      my_callbacks=my_callbacks,
                      run_folder=run_folder,
                      lr_decay = lr_decay,
                      WGAN_lesion = WGAN_lesion,
                      path_generator_adaptive = path_generator_adaptive,
                      path_generator_not_adaptive = path_generator_not_adaptive,
                      n_of_lesion2add = n_of_lesion2add,
                      balance_training_data = balance_training_data)

    run_net.run()

    validation_method = 'kfold'
    if validation_method == 'kfold':
        n_iter = k
    else:
        n_iter = boot_iter

    evalConv = EvaluateConvNet(run_folder=run_folder,
                               id = slice_path_ID)
    evalConv.use_best_model_on_val_set(n_iter)

    file = open(os.path.join(run_folder, "Parameters.txt"), "a")
    file.write("\nData e ora di fine simulazione: "  + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    file.close()
