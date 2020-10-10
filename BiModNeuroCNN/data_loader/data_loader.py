import numpy as np
import scipy.io as spio
import pickle
from BiModNeuroCNN.utils import timer
from BiModNeuroCNN.data_loader.data_utils import load_pickle


class Loader:

	def __init__(self):
		self.data1 = np.array([])
		self.labels1 = []
		self.data2 = np.array([])
		self.labels2 = []
		self.combined_data = None
		self.datatype1 = None
		self.datatype2 = None

	def __repr__(self):
		return f"<class 'BiModNeuroCNN.data_loader.Loader'>"

	def __str__(self):
		return f"Class for loading two different data types"

	def __getattr__(self, attr):
		if attr == "state":
			return f"Data 1:{not self.data1.size==0} : Labels 1:{not self.labels1==[]} :\
			Data 2:{not self.data2.size==0} : Labels 2:{not self.labels2==[]}"
		if attr == "processed":
			pass

	def __setattr__(self, name, value):
		if name == "datatypes":
			self.datatype1 = value[0]
			self.datatype2 = value[1]
		else:
			super().__setattr__(name, value)

	@timer
	def loadmat(self, datafile, labelsfile=None):
		"""
	    Load previously-validated EEG data and labels in the form of a .mat file.

	    params: datafile (str): location and name of file containing data
	    params: labelsfile (str): location and name of separate file containing labels

	    Returns: n_trial * n_chans * n_samples Numpy array contianing EEG data.
	             list containing labels for all trials.
	    """
		data = spio.loadmat(f"{datafile}.mat")
		data = data[list(data.keys())[3]]
		
		if labelsfile != None:  
			labels = spio.loadmat(f"{labelsfile}.mat")
			labels = labels[list(labels.keys())[3]] 

		if self.data1.size == 0:
			self.data1 = data
			self.labels1 = labels[0]
		elif self.data2.size == 0:
			self.data2 = data
			self.labels2 = labels[0]
		else:
			raise AttributeError("Maximum 2 data types already loaded")

	@timer
	def loadMNE(self, filename, data_tag='EEG', label_tag='labels', load_labels=True):
		"""
        Load previously-validated EEG data and labels in the form of an MNE Raw Array.

        Returns: n_chans * n_samples Numpy array contianing EEG data.
                 data3D: n_trials * n_chans * n_samples reshaped EEG data.
                 Numpy array containing labels for all trials.
        """
		mnePickle = load_pickle(filename)
		data = mnePickle[data_tag].get_data()[:,:-1,:] #remove trigger channel from data
		if load_labels:
			labels = mnePickle[label_tag]

		if self.data1.size == 0:
			self.data1 = data
			self.labels1 = labels
		elif self.data2.size == 0:
			self.data2 = data
			self.labels2 = labels
		else:
			raise AttributeError("Maximum 2 data types already loaded")

	@timer
	def combine_data(self):
		"""
        Combine two data types into single np.array. Useful option for combined classification.
		Number of trials, channels and samples must be equal.

        Returns: n_trials * n_chans * n_samples Numpy array contianing data.
        """
		assert self.data1 is not None, "No data loaded for set 1!"
		assert self.data2 is not None, "No data loaded for set 2!"

		assert self.data1.shape[0] == self.data2.shape[0], "Number of trials must be identical!"
		assert self.data1.shape[2] == self.data2.shape[2], "NUmber of samples must be identical!"

		self.combined_data = np.concatenate((self.data1, self.data2), axis=1)
		assert self.combined_data.shape[1] == self.data1.shape[1] + self.data1.shape[1], "Axis 1 should be sum of EEG and fNIRS Axis 1"

	
	@staticmethod	
	def match_removed_trials(data1, labels1, data2, labels2, total_labels, removed_all, print_result=True):
		"""
		Ensure that samples in two data types are correctly aligned by removed rejected trials from both.

	    Inputs: data1 (np.ndarray): one of the two multimodal data types
	            data2 (np.ndarray): one of the two multimodal data types
	            total_labels (np.array || list): all class labels from entire dataset
	            removed_trials_df (pd.DataFrame): 2 rows containing class and index of removed trials
	            labels: (np.array || list): labels associated with the specific classes of data1 and data2
        Returns: data1 (np.array): data1 == in dimensions to data2
                 labels1 (np.array || list): data1 == data2
                 data2 (np.array): data2 == in dimensions to data1
                 labels2 (np.array || list): data2 == data1
	    """

		placeholder_data1 = np.zeros((data1.shape[1],data1.shape[2]))
		placeholder_data2 = np.zeros((data2.shape[1],data1.shape[2]))

		for tup1 in removed_all.data_1:
			labels1 = np.insert(labels1,tup1[0],tup1[1])
			data1 = np.insert(data1, tup1[0], placeholder_data1, axis=0)

		for tup2 in removed_all.data_2:
			labels2 = np.insert(labels2,tup2[0],tup2[1])
			data2 = np.insert(data2, tup2[0], placeholder_data2, axis=0)
		
		combined_tups = removed_all.data_1
		for t in removed_all.data_2:
		    if t not in combined_tups:
		        combined_tups.append(t)
		removal_index = []
		for i in combined_tups:
		    removal_index.append(i[0])
		removal_index = list(reversed(np.sort(removal_index)))

		for idx in removal_index:
		    total_labels = np.delete(total_labels, idx)
		    data1 = np.delete(data1, idx, axis=0)
		    labels1 = np.delete(labels1, idx)
		    data2 = np.delete(data2, idx, axis=0)
		    labels2 = np.delete(labels2, idx)

		if print_result:
			_, counts = np.unique(total_labels, return_counts=True)
			print(f"Total: {counts}")
			_, counts = np.unique(labels1, return_counts=True)
			print(f"EEG: {counts}")
			_, counts = np.unique(labels2, return_counts=True)
			print(f"fNIRS: {counts}")

		return data1, labels1, data2, labels2