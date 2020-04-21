from Class.EvaluateConvNet import EvaluateConvNet

k = 10
boot_iter = 1

run_folder = 'D:/Documenti/Tesi/Run/Aprile/run_CNN_versioneB/padding = valid_RUN2/2020-04-17_15-17-04'
#run_folder = 'C:/Users/GeNeSiS/PycharmProjects/cnn/run/CNN_Alexnet/2020-04-14_15-46-46'

validation_method = 'kfold'
if validation_method == 'kfold':
    n_iter = k
else:
    n_iter = boot_iter

evalConv = EvaluateConvNet(run_folder=run_folder)
evalConv.use_best_model_on_val_set(n_iter)