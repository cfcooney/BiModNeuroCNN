import numpy as np
import pandas as pd
from BiModNeuroCNN.results.results import Results
from BiModNeuroCNN.training.bimodal_classification import Classification
from BiModNeuroCNN.utils import load_subject, format_data, timer, get_model_loss_and_acc, windows_index
from BiModNeuroCNN.data_loader.data_loader import Loader
from BiModNeuroCNN.data_loader.data_utils import get_class_index_tuples, combine_removed_trials
from BiModNeuroCNN.data_loader.utils1 import subject_data_loader 
from BiModNeuroCNN.models.bimodal_cnn import BiModalNet
from BiModNeuroCNN.models.bimodal_cnn_pooling import BiModalNet_w_Pool
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import StratifiedKFold
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.functions import safe_log
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
import sys
import torch as th
import os

th.backends.cudnn.deterministic = True
log = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


WINDOW_LEN = 200
OVERLAP = 150
windows = windows_index(500,WINDOW_LEN,OVERLAP,250)

hyp_params = dict(window=windows[:2],
				  activation=["leaky_relu"],
                  structure= ["shallow"])


parameters = dict(best_loss = 100.0,
                  batch_size = 32,
                  monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()],
                  model_constraint = MaxNormDefaultConstraint(),
                  max_increase_epochs = 0,
                  cuda = True,
                  epochs=1,
                  learning_rate_scheduler=StepLR,
                  lr_step=20, lr_gamma=0.9)



EEGSubNet_params = dict(n_filters_time=40, filter_time_length=5, n_filters_spat=40, n_filters_2=20, filter_length_2=20,
                        pool_time_length_1=5, pool_time_stride_1=2, pool_length_2=5, pool_stride_2=3, final_conv_length='auto',
                        conv_nonlin=th.nn.functional.leaky_relu, pool_mode='mean', pool_nonlin=safe_log,
                        split_first_layer=True, batch_norm=True, batch_norm_alpha=0.2,
                        drop_prob=0.1)

fNIRSSubNet_params = dict(n_filters_time=40, filter_time_length=5, n_filters_spat=40, n_filters_2=20, filter_length_2=20,
                        pool_time_length_1=5, pool_time_stride_1=2, pool_length_2=5, pool_stride_2=3, final_conv_length='auto',
                        conv_nonlin=th.nn.functional.leaky_relu, pool_mode='mean', pool_nonlin=safe_log,
                        split_first_layer=True, batch_norm=True, batch_norm_alpha=0.2,
                        drop_prob=0.1)


@timer
def train_nested_cv(data1, labels1, data2, labels2, model, rm1_file, rm2_file, subnet1_params,
					subnet2_params, directory, hyps, params, labels_dict):


	unique = np.unique(labels1, return_counts=False)

	num_folds = 5
	skf = StratifiedKFold(n_splits=num_folds, shuffle=False,
	                      random_state=10)  # don't randomize trials to preserce structure


	subj_results = Results(directory, num_folds, 'test') # results structure
	subj_results.get_acc_loss_df(hyps, 'Fold')  # empty dataframe headed with each HP set


	##### Match Removed Trials #####
	d1Rem = get_class_index_tuples(rm1_file)
	d2Rem = get_class_index_tuples(rm2_file)
	names = 'data_1,data_2'
	removed_all = combine_removed_trials(d1Rem, d2Rem, names)

	# Remove bad trials from both datasets and align 
	data1_matched, labels1_matched, data2_matched, labels2_matched = Loader.match_removed_trials(data1, labels1, data2,
	                                                                                             labels2, total_labels,
	                                                                                             removed_all, print_result=False)

	subtr_ceoff = np.min(labels1_matched) # required to set labels from zero
	fcn = lambda l: l - subtr_ceoff
	labels1_matched = fcn(labels1_matched)
	labels2_matched = fcn(labels2_matched)
	
	assert labels1_matched.all() == labels2_matched.all(), f"Order of trial labels must be identical!"


	data_params = dict(n_classes=len(unique),
	                   n_chans_d1=data1_matched.shape[1],
	                   input_time_length_d1=WINDOW_LEN,
	                   n_chans_d2=data2_matched.shape[1],
	                   input_time_length_d2=WINDOW_LEN)

	clf = Classification(model, subnet1_params, subnet2_params, hyps, params, data_params, path3, "package_test")

	subj_results.y_true = np.array([])
	trainsetlist, testsetlist, inner_fold_acc, inner_fold_loss, inner_fold_CE = ([] for i in range(5))

	print(f"Inner-fold training for Subject {subject} in progress...")

	for inner_ind, outer_index in skf.split(data1_matched, labels1_matched):

	    data1_matched_if, data1_matched_of, data2_matched_if, data2_matched_of = data1_matched[inner_ind], data1_matched[outer_index], \
	                                                                             data2_matched[inner_ind], data2_matched[outer_index]
	    inner_labels, outer_labels = labels1_matched[inner_ind], labels1_matched[outer_index]
	    subj_results.concat_y_true(outer_labels)

	    print(data1_matched_if.shape, data2_matched_if.shape)

	    trainsetlist.append((SignalAndTarget(data1_matched_if, inner_labels), SignalAndTarget(data2_matched_if, inner_labels)))  # used for outer-fold train/test
	    testsetlist.append((SignalAndTarget(data1_matched_of, outer_labels), SignalAndTarget(data2_matched_of, outer_labels)))

	    for train_idx, valid_idx in skf.split(data1_matched_if, inner_labels):

	        d1_train, d1_val, d2_train, d2_val = data1_matched_if[train_idx], data1_matched_if[valid_idx], \
	        									 data2_matched_if[train_idx], data2_matched_if[valid_idx]
	        y_train, y_val = inner_labels[train_idx], inner_labels[valid_idx]

	        train_set_1 = SignalAndTarget(d1_train, y_train)
	        val_set_1 = SignalAndTarget(d1_val, y_val)
	        train_set_2 = SignalAndTarget(d2_train, y_train)
	        val_set_2 = SignalAndTarget(d2_val, y_val)

	        hyp_param_acc, hyp_param_loss, hyp_param_CE = clf.train_inner(train_set_1, val_set_1, train_set_2, val_set_2,
	                                                                      test_set_1=None, test_set_2=None, save_model=False)

	        
	        inner_fold_loss.append(hyp_param_loss) #5 outer folds * 5 inner folds * number of HPs
	        inner_fold_acc.append(hyp_param_acc)
	        inner_fold_CE.append(hyp_param_CE)

	subj_results.fill_acc_loss_df(inner_fold_loss,  inner_fold_acc, inner_fold_CE)

	subj_results.get_best_params("accuracy")
	print(f"best params: {subj_results.best_params}")
	clf.best_params = subj_results.best_params 
	clf.set_best_params()

	# accuracy score for each fold, combined predictions for each fold
	scores, fold_models, predictions, probabilities, outer_cross_entropy, y_true = clf.train_outer(trainsetlist,
	                                                                                               testsetlist,
	                                                                                               False, print_details=True)

	print(f"Accuracy: {round((accuracy_score(y_true, predictions) * 100), 3)}")

	subj_results.outer_fold_accuracies = scores
	subj_results.y_pred = np.array(predictions)
	subj_results.y_probs = np.array(probabilities)
	subj_results.outer_fold_cross_entropies = outer_cross_entropy

	subj_results.train_loss, subj_results.valid_loss, subj_results.test_loss, subj_results.train_acc, subj_results.valid_acc, subj_results.test_acc = get_model_loss_and_acc(
	    fold_models)
	try:
	    subj_results.save_result()
	except BaseException:
	    print(f"Unable to save results for Subject: {subj} / Session: {session} - {category}")
	try:
	    subj_results.subject_stats()
	    print("")
	    print(subj_results.subject_stats_df.head())

	    subj_results.get_accuracy()
	    #print(f"Mean Accuracy: {subj_results.accuracy}")
	except ValueError:
	    print(f"Unable to store subject stats as excel file - see the subject results pickle.")
	    pass  


if __name__ == '__main__':

	directory = 'BiModNeuroCNN/data/'

	subjects = ['11']
	sessions = [1]

	save_dir = directory # chance to suitable storage directory

	labels_dict = dict(actionText=[1, 2, 3,4], combsText=[5, 6, 7, 8],
	                   actionImage=[9, 10, 11, 12], combsImage=[13, 14, 15,16],
	                   actionAudio=[17, 18, 19, 20], combsAudio=[21, 22, 23, 24])

	categories = ["actionImage"]

	model = BiModalNet

	for subject in subjects:
		path1 = f"{save_dir}/S{subject}"

		if not os.path.exists(path1):
		    os.makedirs(path1)

		for session in sessions:
		    path2 = f"{path1}/Session_{session}"

		    total_labels = pd.read_csv(f"{directory}/S{subject}/Session_{session}/total_labels.txt", header=None).values[0]
		    removed_labels_1 = "{directory}/S{subj}/Session_{session}/removedEEG.txt"
		    removed_labels_2 = "{directory}/S{subj}/Session_{session}/removedEEG.txt"

		    if not os.path.exists(path2):
		        os.makedirs(path2)

		    for category in categories:
		        path3 = f"{path2}/{category}"

		        if not os.path.exists(path3):
		            os.makedirs(path3)

		        f_name_1 = f"{directory}/S{subject}/Session_{session}/classifierData/{category}_EEG_CLF"
		        f_name_2 = f"{directory}/S{subject}/Session_{session}/classifierData/{category}_fNIRS_CLF_0"
		        data_1, labels_1 = subject_data_loader(f_name_1)
		        data_2, labels_2 = subject_data_loader(f_name_2)
		       
		        train_nested_cv(data_1, labels_1, data_2, labels_2, model, removed_labels_1, removed_labels_2, EEGSubNet_params,
		        				fNIRSSubNet_params, directory, hyp_params, parameters, labels_dict)


