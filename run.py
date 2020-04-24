from Class.EvaluateConvNet import EvaluateConvNet

k = 10
boot_iter = 1

run_folder = 'C:/Users/User/PycharmProjects/Local/cnn/run/CNN_Alexnet/2020-04-22_03-59-06'

validation_method = 'kfold'
if validation_method == 'kfold':
    n_iter = k
else:
    n_iter = boot_iter

evalConv = EvaluateConvNet(run_folder=run_folder)
evalConv.use_best_model_on_val_set(n_iter)