"""
Description: Script adapted from: https://github.com/robintibor/braindecode/tree/master/braindecode/experiments
Modifications primarily to enable bimodal training to implement model saving. Includes probabilites for use with
cross entropy metric.
"""
import logging
from collections import OrderedDict
from copy import deepcopy
import time
import os
import numpy as np
import pandas as pd
import torch as th

from braindecode.datautil.splitters import concatenate_sets
from braindecode.experiments.loggers import Printer
from braindecode.experiments.stopcriteria import MaxEpochs, ColumnBelow, Or
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.monitors import compute_pred_labels_from_trial_preds

from BiModNeuroCNN.training.training_utils import combine_batches



log = logging.getLogger(__name__)
th.backends.cudnn.deterministic = True

class RememberBest(object):
    """
    Class to remember and restore 
    the parameters of the model and the parameters of the
    optimizer at the epoch with the best performance.

    Parameters
    ----------
    column_name: str
        The lowest value in this column should indicate the epoch with the
        best performance (e.g. misclass might make sense).
        
    Attributes
    ----------
    best_epoch: int Index of best epoch
    """
    def __init__(self, column_name, predictions, probabilities):
        self.column_name = column_name
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.lowest_test = float('inf')
        self.lowest_val_misclass = float('inf')
        self.model_predictions = None
        self.model_probabilities = None


    def remember_epoch(self, epochs_df, model, optimizer, save_path, tag, class_acc, save_model, predictions, probabilities):
        """
        Remember this epoch: Remember parameter values in case this epoch
        has the best performance.
        
        Parameters
        ----------
        :param epochs_df: (pandas.Dataframe) Dataframe containing the column `column_name` with which performance is evaluated.
        :param model: (torch.nn.Module)
        :param optimizer: (torch.optim.Optimizer)
        :param subject_id: (str) identifier
        :param tag: (str) label to give the saved CNN e.g. "BmCNN"
        :param directory: (str) directory for saving models
        :param save_model: boolean True or False
        :param probabilities: softmax probabilities to be used for cross entropy metric
        :param predictions:  classifier prediction values for epoch
        """
        self.class_acc = class_acc
        self.optimizer = optimizer
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1]) #validation misclass
        if "test_misclass" in list(epochs_df.columns.values):
            current_test_misclass = float(epochs_df['test_misclass'].iloc[-1]) #test misclass
        else:
            current_test_misclass = 0

        #####Storing of the models enabled depending on current of loss and validation accuracy#####
        if (current_val < self.lowest_val) or (
                current_val == self.lowest_val and current_test_misclass <= self.lowest_test):
            
            self.lowest_test = current_test_misclass
            self.class_acc.append(current_test_misclass)

            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_predictions = predictions
            self.model_probabilities = probabilities
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))
            log.info("")

            if save_model:
                log.info("Saving current best model for validation accuracy...")
                log.info("")

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                try:
                    th.save(model.state_dict(), f"{save_path}/{tag}_{self.best_epoch}.pt")
                except PermissionError:
                    # redundancy for storing of models
                    log.info("Permission denied for this path!")
                    th.save(model.state_dict(), f"{save_path}/{tag}_{self.best_epoch}_a.pt")
                finally:
                    log.info("model not saved! Continuing with training")
                self.model_predictions = predictions
                self.model_probabilities = probabilities
                
        return self.model_predictions, self.model_probabilities

    def reset_to_best_model(self, epochs_df, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows 
        after best epoch from epochs dataframe.
        
        Modifies parameters of model and optimizer, changes epochs_df in-place.
        
        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch+1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class Experiment(object):
    """
    Class that performs one experiment on training, validation and test set.

    It trains as follows:
    
    1. Train on training set until a given stop criterion is fulfilled
    2. Reset to the best epoch, i.e. reset parameters of the model and the 
       optimizer to the state at the best epoch ("best" according to a given
       criterion)
    3. Continue training on the combined training + validation set until the
       loss on the validation set is as low as it was on the best epoch for the
       training set. (or until the ConvNet was trained twice as many epochs as
       the best epoch to prevent infinite training)

    Parameters
    ----------
    Parameters
    ----------
    :param epochs_df: (pandas.Dataframe) Dataframe containing the column `column_name` with which performance is evaluated.
    :model: (torch.nn.Module)
    :train_set_1: (braindecode.SignalAndTarget)
    :valid_set_1: (braindecode.SignalAndTarget)
    :train_set_2: (braindecode.SignalAndTarget)
    :valid_set_2: (braindecode.SignalAndTarget)
    :test_set_1: (braindecode.SignalAndTarget)
    :test_set_2: (braindecode.SignalAndTarget)
    :iterator: (iterator object)
    :loss_function: function 
        Function mapping predictions and targets to a loss: 
        (predictions: `torch.autograd.Variable`, 
        targets:`torch.autograd.Variable`)
        -> loss: `torch.autograd.Variable`
    :optimizer: (torch.optim.Optimizer)
    :model_constraint: object
        Object with apply function that takes model and constraints its 
        parameters. `None` for no constraint.
    :monitors: list of objects
        List of objects with monitor_epoch and monitor_set method, should
        monitor the traning progress.
    :stop_criterion: object
        Object with `should_stop` method, that takes in monitoring dataframe
        and returns if training should stop:
    :remember_best_column: str
        Name of column to use for storing parameters of best model. Lowest value
        should indicate best performance in this column.
    :run_after_early_stop: bool
        Whether to continue running after early stop
    :model_loss_function: function, optional
        Function (model -> loss) to add a model loss like L2 regularization.
        Note that this loss is not accounted for in monitoring at the moment.
    :save_file: (str) path to save model
    :tag: (str) name to attach to saved model
    :save_model: (bool) whetjer to save model or not
    :batch_modifier: object, optional
        Object with modify method, that can change the batch, e.g. for data
        augmentation
    :cuda: bool, optional
        Whether to use cuda.
    :pin_memory: bool, optional
        Whether to pin memory of inputs and targets of batch.
    :do_early_stop: bool
        Whether to do an early stop at all. If true, reset to best model
        even in case experiment does not run after early stop.
    :reset_after_second_run: bool
        If true, reset to best model when second run did not find a valid loss
        below or equal to the best train loss of first run.
    :log_0_epoch: bool
        Whether to compute monitor values and log them before the
        start of training.
    :loggers: list of :class:`.Logger`
        How to show computed metrics.
        
    Attributes
    ----------
    epochs_df: `pandas.DataFrame`
        Monitoring values for all epochs.
    """
    def __init__(self, model, train_set_1, valid_set_1, train_set_2, valid_set_2, test_set_1, test_set_2, iterator,
                 loss_function, optimizer, lr_scheduler, model_constraint, monitors, stop_criterion, remember_best_column, run_after_early_stop,
                 model_loss_function, save_file, tag, save_model, batch_modifier=None, cuda=True, pin_memory=False,
                 do_early_stop=True, reset_after_second_run=False, log_0_epoch=True, loggers=('print',)):

        if run_after_early_stop or reset_after_second_run:
            assert do_early_stop == True, ("Can only run after early stop or "
            "reset after second run if doing an early stop")
        if do_early_stop:
            assert valid_set_1 is not None and valid_set_2 is not None 
            assert remember_best_column is not None
        self.model = model
        self.datasets = OrderedDict((('train_1', train_set_1), ('train_2', train_set_2),
                                     ('valid_1', valid_set_1), ('valid_2', valid_set_2),
                                     ('test_1', test_set_1), ('test_2', test_set_2))) 

        if valid_set_1 is None or valid_set_2 is None:
            self.datasets.pop('valid_1')
            self.datasets.pop('valid_2')
            assert run_after_early_stop == False
            assert do_early_stop == False
        if test_set_1 is None or test_set_2 is None:
            self.datasets.pop('test_1')
            self.datasets.pop('test_2')

        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_constraint = model_constraint
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.remember_best_column = remember_best_column
        self.run_after_early_stop = run_after_early_stop
        self.model_loss_function = model_loss_function
        self.batch_modifier = batch_modifier
        self.cuda = cuda
        self.epochs_df = pd.DataFrame()
        self.before_stop_df = None
        self.rememberer = None
        self.pin_memory = pin_memory
        self.do_early_stop = do_early_stop
        self.reset_after_second_run = reset_after_second_run
        self.log_0_epoch = log_0_epoch
        self.loggers = loggers
        self.save_file = save_file
        self.tag = tag
        self.class_acc = [] 
        self.save_model = save_model
        self.predictions = None
        self.probabilites = None
        self.lr_scheduler = lr_scheduler


    def run(self):
        """
        Run complete training.
        """
        self.setup_training()
        log.info("Run until first stop...")
        self.run_until_first_stop()
        if self.do_early_stop:
            # always setup for second stop, in order to get best model
            # even if not running after early stop...
            log.info("Setup for second stop...")
            self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            self.run_until_second_stop()
            if self.reset_after_second_run:
                # if no valid loss was found below the best train loss on 1st
                # run, reset model to the epoch with lowest valid_misclass
                if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach:
                    log.info("Resetting to best epoch {:d}".format(
                        self.rememberer.best_epoch))
                    self.rememberer.reset_to_best_model(self.epochs_df,
                                                        self.model,
                                                        self.optimizer)

    def setup_training(self):
        """
        Setup training, i.e. transform model to cuda,
        initialize monitoring.
        """
        # reset remember best extension in case you rerun some experiment
        if self.do_early_stop:
            self.rememberer = RememberBest(self.remember_best_column, self.predictions, self.probabilites)
        if self.loggers == ('print',):
            self.loggers = [Printer()]
        self.epochs_df = pd.DataFrame()
        if self.cuda:
            assert th.cuda.is_available(), "Cuda not available"
            self.model.cuda()

    def run_until_first_stop(self):
        """
        Run training and evaluation using only training set for training
        until stop criterion is fulfilled.
        """

        self.run_until_stop(self.datasets, remember_best=self.do_early_stop)

    def run_until_second_stop(self):
        """
        Run training and evaluation using combined training + validation sets 
        for training on both datasets. 
        
        Runs until loss on validation set decreases below loss on training set 
        of best epoch or  until as many epochs trained after as before 
        first stop.
        """
        datasets = self.datasets
        datasets['train_1'] = concatenate_sets([datasets['train_1'],
                                                datasets['valid_1']]) 
        datasets['train_2'] = concatenate_sets([datasets['train_2'],
                                                datasets['valid_2']]) 

        self.run_until_stop(datasets, remember_best=True)

    def run_until_stop(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets until stop criterion is
        fulfilled. Return predictions and probabilites associated with best epochs.
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters at best epoch.
        """
        if self.log_0_epoch:
            self.monitor_epoch(datasets)
            self.log_epoch()
            if remember_best:
                self.model_predictions, self.model_probabilities = self.rememberer.remember_epoch(self.epochs_df, self.model, self.optimizer,
                                                                                                  self.save_file, self.tag, self.class_acc,
                                                                                                  self.save_model, self.predictions, self.probabilites) 
        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets for one epoch. Batches for 
        two data types are combined.
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters if this epoch is best epoch.
        """
        batch_generator_1 = self.iterator.get_batches(datasets['train_1'],
                                                    shuffle=True) 
        batch_generator_2 = self.iterator.get_batches(datasets['train_2'],
                                                    shuffle=True) 
        combined_batches = combine_batches(batch_generator_1, batch_generator_2)
        start_train_epoch_time = time.time()
        for inputs_1, targets_1, inputs_2, targets_2 in combined_batches:
            if self.batch_modifier is not None:
                inputs_1, targets_1 = self.batch_modifier.process(inputs_1, targets_1) 
                inputs_2, targets_2 = self.batch_modifier.process(inputs_2, targets_2) 

            if len(inputs_1) > 0 and len(inputs_2) > 0:

                self.train_batch(inputs_1, targets_1, inputs_2, targets_2)
                if self.lr_scheduler != None:
                    self.lr_scheduler.step()

        end_train_epoch_time = time.time()

        log.info("Time only for training updates: {:.2f}s".format(
            end_train_epoch_time - start_train_epoch_time))

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self. model_predictions, self.model_probabilities = self.rememberer.remember_epoch(self.epochs_df, self.model, self.optimizer,
                                                                                               self.save_file, self.tag, self.class_acc,
                                                                                               self.save_model, self.predictions, self.probabilites) 

    def train_batch(self, inputs_1, targets_1, inputs_2, targets_2):
        """
        Train on given inputs and targets.
        
        Parameters
        ----------
        :inputs_1: (torch.autograd.Variable)
        :targets_1: (torch.autograd.Variable)
        :inputs_2: (torch.autograd.Variable)
        :targets_2: (torch.autograd.Variable)
        """
        
        self.model.train()
        input_vars_1 = np_to_var(inputs_1, pin_memory=self.pin_memory)
        target_vars_1 = np_to_var(targets_1, pin_memory=self.pin_memory)
        input_vars_2 = np_to_var(inputs_2, pin_memory=self.pin_memory) 
        target_vars_2 = np_to_var(targets_2, pin_memory=self.pin_memory) 

        if self.cuda:
            input_vars_1 = input_vars_1.cuda()
            target_vars_1 = target_vars_1.cuda()
            input_vars_2 = input_vars_2.cuda() 
            target_vars_2 = target_vars_2.cuda() 
        self.optimizer.zero_grad()
        th.autograd.set_detect_anomaly(True)

        outputs = self.model(input_vars_1, input_vars_2) 
        loss = self.loss_function(outputs, target_vars_1) 
        if self.model_loss_function is not None:
            loss = loss + self.model_loss_function(self.model)
        
        loss.backward()
       
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)


    def eval_on_batch(self, inputs_1, targets_1, inputs_2, targets_2):
        """
        Evaluate given inputs and targets.
        
        Parameters
        ----------
        :inputs_1: (torch.autograd.Variable)
        :targets_1: (torch.autograd.Variable)
        :inputs_2: (torch.autograd.Variable)
        :targets_2: (torch.autograd.Variable)

        Returns
        -------
        predictions: `torch.autograd.Variable`
        loss: `torch.autograd.Variable`

        """
        self.model.eval()
        with th.no_grad():

            input_vars_1 = np_to_var(inputs_1, pin_memory=self.pin_memory)
            target_vars_1 = np_to_var(targets_1, pin_memory=self.pin_memory) # only 1 target array required
            input_vars_2 = np_to_var(inputs_2, pin_memory=self.pin_memory) 

            if self.cuda:
                input_vars_1 = input_vars_1.cuda()
                target_vars_1 = target_vars_1.cuda()
                input_vars_2 = input_vars_2.cuda()

            outputs = self.model(input_vars_1, input_vars_2)

            probabilities = th.exp(outputs.cpu()).numpy() # calculated probabilities

            loss = self.loss_function(outputs, target_vars_1)
            if hasattr(outputs, 'cpu'):
                outputs = outputs.cpu().data.numpy()
            else:

                outputs = [o.cpu().data.numpy() for o in outputs]
                
            loss = loss.cpu().data.numpy()

        return outputs, loss, probabilities

    def monitor_epoch(self, datasets):
        """
        Evaluate one epoch for given datasets.
        
        Stores results in `epochs_df`
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.

        """
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)

        set_1, set_2, set_list = [], [], []
        for i, j in self.datasets.items():
            set_list.append(i)
        for i in range(0, len(set_list), 2):
            set_1.append(set_list[i])
            set_2.append(set_list[i + 1])


        for name_1, name_2 in zip(set_1, set_2):
            setname = name_1.split('_')[0]

            batch_gen_1 = self.iterator.get_batches(datasets[name_1], shuffle=False) 
            batch_gen_2 = self.iterator.get_batches(datasets[name_2], shuffle=False)  
            combined_batches = combine_batches(batch_gen_1, batch_gen_2)  

            all_preds = []
            all_losses = []
            all_probs = []
            all_batch_sizes = []
            all_targets = []

            for inputs_1, targets_1, inputs_2, targets_2 in combined_batches:
                preds, loss, probabilities = self.eval_on_batch(inputs_1, targets_1,
                                                                inputs_2, targets_2)  
                all_preds.append(preds)
                all_losses.append(loss)
                all_probs.append(probabilities)
                all_batch_sizes.append(len(inputs_1))
                all_targets.append(targets_1)

            for m in self.monitors:
                result_dict = m.monitor_set(setname, all_preds, all_losses,
                                            all_batch_sizes, all_targets,
                                            combined_batches) 
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys()), f"Columns of dataframe: {str(set(self.epochs_df.columns))}\n and keys of dict {str(set(row_dict.keys()))} not same"
        
        self.epochs_df = self.epochs_df[list(row_dict.keys())]
        self.predictions = compute_pred_labels_from_trial_preds(all_preds, None)
        self.probabilites = all_probs


    def log_epoch(self):
        """
        Print monitoring values for this epoch.
        """
        for logger in self.loggers:
            logger.log_epoch(self.epochs_df)

    def setup_after_stop_training(self):
        """
        Setup training after first stop. 

        Resets parameters to best parameters and updates stop criterion.
        """
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.before_stop_df = deepcopy(self.epochs_df)
        self.rememberer.reset_to_best_model(self.epochs_df, self.model,
                                            self.optimizer)
        loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
        self.stop_criterion = Or(stop_criteria=[
            MaxEpochs(max_epochs=self.rememberer.best_epoch * 2),
            ColumnBelow(column_name='valid_loss', target_value=loss_to_reach)])
        log.info(f"Train loss to reach {loss_to_reach}")


