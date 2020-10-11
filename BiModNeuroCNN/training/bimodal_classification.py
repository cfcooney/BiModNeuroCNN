"""
Description: Class for training CNNs using a nested cross-validation method. Train on the inner_fold to obtain
optimized hyperparameters. Train outer_fold to obtain classification performance.
"""
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.util import set_random_seeds, np_to_var, var_to_np
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.functions import square, safe_log
import torch as th
from BiModNeuroCNN.models.bimodal_cnn_pooling import BiModalNet
from sklearn.model_selection import train_test_split
from BiModNeuroCNN.training.training_utils import current_acc, current_loss
from BiModNeuroCNN.data_loader.data_utils import smote_augmentation, multi_SignalAndTarget
from BiModNeuroCNN.results.results import Results as res
from torch.nn.functional import nll_loss, cross_entropy
from BiModNeuroCNN.training.bimodal_training import Experiment
import numpy as np
import itertools as it
import torch
from torch import optim
import logging
from ast import literal_eval
from BiModNeuroCNN.results.metrics import cross_entropy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)
torch.backends.cudnn.deterministic = True

class Classification:

    def __init__(self, modelname, subnet1_params, subnet2_params, hyp_params, parameters, data_params, model_save_path, tag):
        self.modelname = modelname
        self.subnet1_params = subnet1_params
        self.subnet2_params = subnet2_params
        self.model_save_path = model_save_path
        self.tag = tag
        self.best_loss = parameters["best_loss"]
        self.batch_size = parameters["batch_size"]
        self.monitors = parameters["monitors"]
        self.cuda = parameters["cuda"]
        self.model_constraint = parameters["model_constraint"]
        self.max_increase_epochs = parameters['max_increase_epochs']
        self.lr_scheduler = parameters['learning_rate_scheduler']
        self.lr_step = parameters['lr_step']
        self.lr_gamma = parameters['lr_gamma']
        self.n_classes = data_params["n_classes"]
        self.n_chans_eeg = data_params["n_chans_eeg"]
        self.input_time_length_eeg = data_params["input_time_length_eeg"]
        self.n_chans_fnirs = data_params["n_chans_fnirs"]
        self.input_time_length_fnirs = data_params["input_time_length_fnirs"]
        self.hyp_params = hyp_params
        self.activation = "elu"
        self.learning_rate = 0.001
        self.dropout = 0.1
        self.epochs = parameters['epochs']
        self.window = None
        self.structure = 'deep'
        self.n_filts = 10 #n_filts in n-1 filters
        self.first_pool = False
        self.loss = nll_loss
        for key in hyp_params:
            setattr(self, key, hyp_params[key])
        self.iterator = BalancedBatchSizeIterator(batch_size=self.batch_size)
        self.best_params = None
        self.model_number = 1
        self.y_pred = np.array([])
        self.y_true = np.array([])
        self.probabilities = np.array([])

    def call_model(self):

        self.subnet1_params['structure'] = self.structure
        self.subnet2_params['structure'] = self.structure

        if self.modelname == 'bimodal_cnn':
            model = BiModalNet(n_classes=4, in_chans_1=56, input_time_1=200,
                               SubNet_1_params=self.subnet1_params, in_chans_2=16,
                               input_time_2=200, SubNet_2_params=self.subnet2_params,
                               linear_dims=100, drop_prob=.2, nonlin=torch.nn.functional.leaky_relu,
                               fc1_out_features=500, fc2_out_features=500, gru_hidden_size=250, gru_n_layers=1)
            th.nn.init.kaiming_uniform_(model.fused_linear.weight)
            th.nn.init.constant_(model.fused_linear.bias, 0)
        return model
        
    def train_model(self, train_set_1, val_set_1, test_set_1, train_set_2, val_set_2, test_set_2, save_model):
        """
        :param train_set_1: (np.array) n_trials*n_channels*n_samples
        :param val_set_1: (np.array) n_trials*n_channels*n_samples
        :param test_set_1: (np.array) n_trials*n_channels*n_samples - can be None when training on inner-fold
        :param train_set_2: (np.array) n_trials*n_channels*n_samples
        :param val_set_2: (np.array) n_trials*n_channels*n_samples
        :param test_set_2:  (np.array) n_trials*n_channels*n_samples - can be None when training on inner-fold
        :param save_model: (Bool) True if trained model is to be saved
        :return: Accuracy and loss scores for the model trained with a given set of hyper-parameters
        """
        model = self.call_model()
        predictions = None

        set_random_seeds(seed=20190629, cuda=self.cuda)

        if self.cuda:
            model.cuda()
            torch.backends.cudnn.deterministic = True
            model = torch.nn.DataParallel(model)
            log.info(f"Cuda in use")

        log.info("%s model: ".format(str(model)))
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.01, eps=1e-8, amsgrad=False)

        stop_criterion = Or([MaxEpochs(self.epochs),
                             NoDecrease('valid_loss', self.max_increase_epochs)])
        model_loss_function = None

        #####Setup to run the selected model#####
        model_test = Experiment(model, train_set_1, val_set_1, train_set_2, val_set_2, test_set_1=test_set_1, test_set_2=test_set_2,
                                iterator=self.iterator, loss_function=self.loss, optimizer=optimizer,
                                lr_scheduler=self.lr_scheduler(optimizer, step_size=self.lr_step, gamma=self.lr_gamma),
                                model_constraint=self.model_constraint, monitors=self.monitors, stop_criterion=stop_criterion,
                                remember_best_column='valid_misclass', run_after_early_stop=True, model_loss_function=model_loss_function,
                                cuda=self.cuda, save_file=self.model_save_path, tag=self.tag, save_model=save_model)
        model_test.run()

        model_acc = model_test.epochs_df['valid_misclass'].astype('float')
        model_loss = model_test.epochs_df['valid_loss'].astype('float')
        current_val_acc = 1 - current_acc(model_acc)
        current_val_loss = current_loss(model_loss)

        test_accuracy = None
        if train_set_1 is not None and test_set_2 is not None:
            val_metric_index = self.get_model_index(model_test.epochs_df)
            test_accuracy = round((1 - model_test.epochs_df['test_misclass'].iloc[val_metric_index]) * 100, 3)
            predictions = model_test.model_predictions
        probabilities = model_test.model_probabilities

        return current_val_acc, current_val_loss, test_accuracy, model_test, predictions, probabilities

    
    def train_inner(self, train_set_1, val_set_1, train_set_2, val_set_2, test_set_1=None, test_set_2=None, augment=False, save_model=False):
        """
        :param train_set_1: (np.array) n_trials*n_channels*n_samples
        :param val_set_1: (np.array) n_trials*n_channels*n_samples
        :param test_set_1: (np.array) n_trials*n_channels*n_samples - can be None when performing HP optimization
        :param train_set_2: (np.array) n_trials*n_channels*n_samples
        :param val_set_2: (np.array) n_trials*n_channels*n_samples
        :param test_set_2:  (np.array) n_trials*n_channels*n_samples - can be None when performing HP optimization
        :param augment: (Bool) True if data augmentation to be applied - currently only configured for SMOTE augmentation
        :param save_model: (Bool) True if trained model is to be saved
        :return: Accuracy, loss and cross entropy scores for the model trained with a given set of hyper-parameters
        """
        val_acc, val_loss, val_cross_entropy = [], [], []
        
        if augment:
            # Only augment training data - never test or validation sets
            train_set_1_os, train_labels_1_os = smote_augmentation(train_set_1.X, train_set_1.y, 2)
            train_set_2_os, train_labels_2_os = smote_augmentation(train_set_1.X, train_set_1.y, 2)
            train_set_1, train_set_2 = multi_SignalAndTarget((train_set_1_os, train_labels_1_os), (train_set_2_os, train_labels_2_os))

        names = list(self.hyp_params.keys())
        hyp_param_combs = it.product(*(self.hyp_params[Name] for Name in names))
        for hyp_combination in hyp_param_combs:

            assert len(hyp_combination) == len(self.hyp_params), f"HP combination must be of equal length to original set."

            for i in range(len(self.hyp_params)):
                setattr(self, list(self.hyp_params.keys())[i], hyp_combination[i])


            if 'window' in self.hyp_params.keys():
                # when using classification window as a hyperparameter - currently data would have to be of same number of samples
                train_set_1 = SignalAndTarget(train_set_1.X[:, :, self.window[0]:self.window[1]], train_set_1.y)
                val_set_1 = SignalAndTarget(val_set_1.X[:, :, self.window[0]:self.window[1]], val_set_1.y)
                train_set_2 = SignalAndTarget(train_set_2.X[:, :, self.window[0]:self.window[1]], train_set_2.y)
                val_set_2 = SignalAndTarget(val_set_2.X[:, :, self.window[0]:self.window[1]], val_set_2.y)

            
            current_val_acc, current_val_loss, _, _, _, probabilities = self.train_model(train_set_1, val_set_1, test_set_1, train_set_2,
                                                                                         val_set_2, test_set_2, save_model)
            val_acc.append(current_val_acc)
            val_loss.append(current_val_loss)

            probabilities = np.array(probabilities).reshape((val_set_1.y.shape[0],4))

            val_cross_entropy.append(cross_entropy(val_set_1.y, probabilities)) #1 CE value per-HP, repeat for n_folds

        return val_acc, val_loss, val_cross_entropy


    def train_outer(self, trainsetlist, testsetlist, augment=False, save_model=True, epochs_save_path=None, print_details=False):
        """
        :param trainsetlist: (list) data as split by k-folds n_folds*(n_trials*n_channels*n_samples)
        :param testsetlist: (list) data as split by k-folds n_folds*(n_trials*n_channels*n_samples)
        :param augment: (Bool) True if data augmentation to be applied - currently only configured for SMOTE augmentation
        :param save_model: (Bool) True if trained model is to be saved
        """
        scores, all_preds, probabilities_list, outer_cross_entropy, fold_models = [],[],[],[],[]
        
        fold_number = 1
        for train_set, test_set in zip(trainsetlist, testsetlist):

            train_set_1, train_set_2 = train_set[0], train_set[1]
            test_set_1, test_set_2   = test_set[0], test_set[1]

            train_set_1_X, val_set_1_X, train_set_1_y, val_set_1_y = train_test_split(train_set_1.X, train_set_1.y, test_size=0.2,
                                                                                      shuffle=True, random_state=42, stratify= train_set_1.y)
            train_set_2_X, val_set_2_X, train_set_2_y, val_set_2_y = train_test_split(train_set_2.X, train_set_2.y, test_size=0.2,
                                                                                      shuffle=True, random_state=42, stratify= train_set_2.y)

            train_set_1, val_set_1, train_set_2, val_set_2 = multi_SignalAndTarget((train_set_1_X, train_set_1_y), (val_set_1_X, val_set_1_y),
                                                                                   (train_set_2_X, train_set_2_y), (val_set_2_X, val_set_2_y))

            if augment:
                # Only augment training data - never test or validation sets
                train_set_1_os, train_labels_1_os = smote_augmentation(train_set_1.X, train_set_1.y, 2)
                train_set_2_os, train_labels_2_os = smote_augmentation(train_set_2.X, train_set_2.y, 2)
                train_set_1 = SignalAndTarget(train_set_1_os, train_labels_1_os)
                train_set_2 = SignalAndTarget(train_set_2_os, train_labels_2_os)


            if 'window' in self.hyp_params.keys():
                # when using classification window as a hyperparameter - currently data would have to be of same number of samples
                self.window = literal_eval(self.window)  # extract tuple of indices
                train_set_1 = SignalAndTarget(train_set_1.X[:,:,self.window[0]:self.window[1]], train_set_1.y)
                val_set_1 = SignalAndTarget(val_set_1.X[:,:,self.window[0]:self.window[1]], val_set_1.y)
                test_set_1 = SignalAndTarget(test_set_1.X[:,:,self.window[0]:self.window[1]], test_set_1.y)
                train_set_2 = SignalAndTarget(train_set_2.X[:,:,self.window[0]:self.window[1]], train_set_2.y)
                val_set_2 = SignalAndTarget(val_set_2.X[:,:,self.window[0]:self.window[1]], val_set_2.y)
                test_set_2 = SignalAndTarget(test_set_2.X[:, :, self.window[0]:self.window[1]], test_set_2.y)

            if print_details:
                print(f"Data 1 train set: {train_set_1.y.shape} | Data 1 val_set: {val_set_1.y.shape} | Data 1 test_set: {test_set_1.y.shape}")
                print(f"Data 2 train set: {train_set_2.y.shape} | Data 2 val_set: {val_set_2.y.shape} | Data 2 test_set: {test_set_2.y.shape}")

            _, _, test_accuracy, optimised_model, predictions, probabilities = self.train_model(train_set_1, val_set_1, test_set_1,
                                                                                                train_set_2, val_set_2, test_set_2, save_model)
            if epochs_save_path != None:
                try:
                    optimised_model.epochs_df.to_excel(f"{epochs_save_path}/epochs{fold_number}.xlsx")
                except FileNotFoundError:
                    optimised_model.epochs_df.to_excel(f"{epochs_save_path}/epochs{fold_number}.xlsx", engine='xlsxwriter')
            
            fold_models.append(optimised_model)
            
            probs_array = []
            for lst in probabilities:
                for trial in lst:
                    probs_array.append(trial) # all probabilities for this test-set
            probabilities_list.append(probs_array) #outer probabilities to be used for cross-entropy


            print(f"/"*20)
            scores.append(test_accuracy)
            self.concat_y_pred(predictions)
            self.concat_y_true(test_set_1.y)
            
            fold_number += 1
        for y_true, y_probs in zip(testsetlist, probabilities_list):
            outer_cross_entropy.append(cross_entropy(y_true[0].y, y_probs))

        return scores, fold_models, self.y_pred, probabilities_list, outer_cross_entropy, self.y_true

    def set_best_params(self):
        """
        Set optimal hyperparameter values selected from optimization - Best parameter values can be
        accessed with BiModNeuroCNN.results.Results.get_best_params() and the list assigned to self.best_params.
        """
        assert type(self.best_params) is list, "list of selected parameters required"
        for i in range(len(self.hyp_params)):
            setattr(self, list(self.hyp_params.keys())[i], self.best_params[i+2])

    @staticmethod
    def get_model_index(df):
        """
        Returns the row index of a pandas dataframe used for storing epoch-by-epoch results.
        :param df: pandas.DataFrame
        :return: int index of the selected epoch based on validation metric
        """
        valid_metric_index = df['valid_misclass'].idxmin()
        best_val_acc = df.index[df['valid_misclass'] == df['valid_misclass'].iloc[valid_metric_index]]
        previous_best = 1.0
        i = 0
        for n, index in enumerate(best_val_acc):
            value = df['test_misclass'][index]
            if value < previous_best:
                previous_best = value
                i = n
        return best_val_acc[i]

    def concat_y_pred(self, y_pred_fold):
        """
        Method for combining all outer-fold ground-truth values.
        :param y_pred_fold: array of single-fold true values.
        :return: all outer fold true values in single arrau
        """
        self.y_pred = np.concatenate((self.y_pred, np.array(y_pred_fold)))

    def concat_y_true(self, y_true_fold):
        """
        Method for combining all outer-fold ground-truth values.
        :param y_true_fold: array of single-fold true values.
        :return: all outer fold true values in single arrau
        """
        self.y_true = np.concatenate((self.y_true, np.array(y_true_fold)))

    def concat_probabilities(self, probabilities_fold):
        """
        Method for combining all outer-fold ground-truth values.
        :param y_pred_fold: array of single-fold true values.
        :return: all outer fold true values in single arrau
        """
        self.probabilities = np.concatenate((self.probabilities, probabilities_fold))

