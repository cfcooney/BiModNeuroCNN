import pickle
import numpy as np
import pandas as pd
import collections
from braindecode.datautil.signal_target import SignalAndTarget
from imblearn.over_sampling import SMOTE

def multi_SignalAndTarget(*args):
        """
        Returns muliple SignalAndTarget objects from multiple (X,y) data tuples

        :param: *args (tuple) any number of tuples containing data and labels
        """
        return_list= []
        for arg in args:
            return_list.append(SignalAndTarget(arg[0], arg[1]))
        return tuple(return_list)

def load_pickle(filename):
   
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file

def get_class_index_tuples(filename):
	"""
	Load removed trials from .txt file and reformat into list of tuples (index, class).
	Index is the trial number and class is the corresponding class label

	Inputs: filename (str): .txt file containing removed trials. E.g. f"{path}/removedEEG.txt"
	Returns: list of tuples (index, class)
	"""
	class_l, index_l, return_l = [], [], []
	try:
		removed_trials= pd.read_csv(filename, header=None).values[0]

		for d in removed_trials:
			if type(d) == str:
				values = d.replace("(",",").replace(")","")
				class_l.append(int(values.split(",")[0]))
				index_l.append(int(values.split(",")[1]))
		[return_l.append((x,y)) for x,y in zip(index_l,class_l) if (x,y) not in return_l]
		return return_l
	except:
		print("Either no file available or no trials removed: [] returned.")
		return []

def combine_removed_trials(Rm1, Rm2, names):
	removed = collections.namedtuple("removed_samples", names)
	return removed(Rm1, Rm2)

def get_classifier_window(data, start, end, prestim=0.5, sfreq=100):
	"""
	Similar to <BiModNeuroCNN.subjects.subjects.Subject.get_classifier_window> in 
	that it extracts epoched time windows within a trial period.
	
	:param data: (np.array) n_trials * n_chans * n_samples
	:param start: (float) time to begin classification window
	:param end: (float) time to end classification window
	:param prestim: (float) length of pre-stimulus period in the data
	:return: (np.array): n_trials*n_channels*len(classification_window)
	"""
	
	fcn = lambda x : x * sfreq
	
	start_samples = int(fcn(start)) + int(fcn(prestim))
	end_samples = int(fcn(end)) + int(fcn(prestim))
	classifier_data = data[:,:,start_samples:end_samples]
	epoch = classifier_data.shape[2]

	return classifier_data, epoch


def smote_augmentation(data, labels, mixing_ratio=2, print_shape=False):
    """
    Method for oversampling the number of trials to augment
    training data. Shoulf only be used on training data
    :input: data (3d array): training data
            labels (np.array OR list): class labels
            mixing_ratio (int): ratio to oversample - e.g. 2 means
            ratio of synthetic data to real data is 2:1
            print_shape (bool): command to print oversampled data shape
    :return: data_os (ndarray): array with a balanced set of trials
             labels_os (np.array): array with a balanced set of labels
    """
    unique, counts = np.unique(labels, return_counts=True)
    os_value = np.ceil(np.max(counts) * mixing_ratio).astype(np.int32)

    s = SMOTE(sampling_strategy={np.unique(labels)[0]: os_value, np.unique(labels)[1]: os_value,
                                 np.unique(labels)[2]: os_value, np.unique(labels)[3]: os_value},
              random_state=10, k_neighbors=3)

    data_os_2d, labels_os = s.fit_resample(data.reshape((data.shape[0], data.shape[1] * data.shape[2])), labels)
    data_os = data_os_2d.reshape((data_os_2d.shape[0], data.shape[1], data.shape[2]))

    if print_shape:
        print(f"Oversampled data shape: {data_os.shape}")

    return data_os, labels_os