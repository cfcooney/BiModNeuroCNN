from BiModNeuroCNN.utils import timer, labels_dict_and_list
from BiModNeuroCNN.data_loader.data_loader import Loader
from scipy.signal import decimate as dec
from tensorflow.keras.utils import normalize
from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
import scipy.io as spio
import pickle
import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)


class Subject(Loader):
	
	direct = 'C:/Users/cfcoo/OneDrive - Ulster University/Study_3/Subject_Data'

	def __init__(self, id):

		super().__init__()

		self.id = id
		
		self.channels_validated = False
		self.trials_validated = False
		self.description = None
		self.data_loaded = False

		self.data1 = np.array([])
		self.data2 = np.array([])
		self.data_combined = None
		self.labels1 = []
		self.labels2 = []
		self.labels_combined = None

		self.epoched_data1 = None
		self.epoched_data2 = None
		self.classifier_start = 0
		self.classifier_end = 0

		self.classnames = []

		self.sfreq1 = 0
		self.sfreq2 = 0
		self.lowcut = 0
		self.highcut = 0
		self.downsample_rate1 = 2
		self.downsample_rate2 = 2
		self.downsampled = [False, False]
		self.normalized = [False, False]
		self.filtered = [False, False]

	def __repr__(self):
		return f"<class 'BiModNeuroCNN.subjects.Subject'>"

	def __str__(self):
		return f"Class for creating subject-specific objects for multi-subject experiments."

	# def __getattr__(self, attr):
	# 	pass

	# def __setattr__(self, name, value):
	# 	pass

	def set_description(self, description):
		self.description = description 

	def get_description(self):
		return self.description

	def change_directory(self, new_direct):
		self.direct = new_direct

	def set_channel_validation(self, validated):
		assert type(validated) == bool
		self.channels_validated = validated 

	def get_channel_validation(self):
		return self.channels_validated

	def set_trial_validation(self, validated):
		assert type(validated) == bool
		self.trials_validated = validated 

	def get_trial_validation(self):
		return self.trials_validated

	def get_classifier_window(self, start, end, data1=True, data2=True, prestim=0.5, sfreq1=100, sfreq2=100):
		"""
		Epoch the time-period within each trial to extract a specfic window for analysis.

		:param start: (float) time to begin classification window
		:param end: (float) time to end classification window
		:param data1 (bool) whether to apply method to self.data1
		:param data2 (bool) whether to apply method to self.data2
		:param prestim: (float) length of pre-stimulus period in the data
		:param sfreq1: (int) sampling frequency of self.data1
		:param sfreq2: (int) sampling frequency of self.data2
		:return: (np.array): n_trials*n_channels*len(classification_window)
		"""
		if data1 == False and data2 == False:
			raise ValueError(f"Require at least one data type to be True: data1:{data1}, data2:{data2}")
		else:
			self.classifier_start = start
			self.classifier_end = end
			if data1:
				fcn = lambda x : x * sfreq1
			
				start_samples = int(fcn(start)) + int(fcn(prestim))
				end_samples = int(fcn(end)) + int(fcn(prestim))

				self.epoched_data1 = self.data1[:,:,start_samples:end_samples]
			if data2:
				fcn = lambda x : x * sfreq2

				start_samples = int(fcn(start)) + int(fcn(prestim))
				end_samples = int(fcn(end)) + int(fcn(prestim))
				
				self.epoched_data2 = self.data2[:,:,start_samples:end_samples]

	def bandpass(self, lowcut, highcut, order, data1=True, data2=False, sfreq1=100, sfreq2=100):
		"""
		Bandpass filter the data with butterworth filter. Use for EEG data

		:params: lowcut (float): low-pass cutoff frequency
	    :params: highcut (float): high-pass cutoff frequency
		:params: order (int): Butterworth filter order number - see scipy docs.
		:params: data1 (bool): filter data1 or not
		:params: data2 (bool): filter data2 or not
		:params: sfreq1: (int) sampling frequency of self.data1
		:params: sfreq2: (int) sampling frequency of self.data2
	    Returns: n_trial * n_chans * n_samples Numpy array contianing filtered data.
		"""
		if data1 == False and data2 == False:
			raise ValueError(f"Require at least one data type to be True: data1:{data1}, data2:{data2}")
		else:
			self.lowcut = lowcut
			self.highcut = highcut
			if data1:
				self.sfreq1 = sfreq1
				nyq = 0.5 * sfreq1
				low = lowcut / nyq
				high = highcut / nyq
				b, a = butter(order, [low, high], btype='band')
				self.data1 = lfilter(b, a, self.data1)
				self.filtered[0] = True
			if data2:
				self.sfreq2 = sfreq2
				nyq = 0.5 * sfreq2
				low = lowcut / nyq
				high = highcut / nyq
				b, a = butter(order, [low, high], btype='band')
				self.data2 = lfilter(b, a, self.data2)
				self.filtered[1] = True
	
	def down_and_normal(self, data1=True, data2=False, downsample_rate1=2, downsample_rate2=2, norm=True):
		"""
		Downsample and normalize the data.

		:params: data1 (bool): apply to data1 or not
		:params: data2 (bool): apply to data2 or not
		:params: downsample_rate1 (int): downsample rate.
		:params: downsample_rate2 (int): downsample rate.
		:params: norm: (bool) to normalize or not to normalize.
	    Returns: n_trial * n_chans * n_samples Numpy array containing downsampled and/or normalized data.
		"""
		if data1 == False and data2 == False:
			raise ValueError(f"Require at least one data type to be True: data1:{data1}, data2:{data2}")
		else:
			fnc = lambda a: a * 1e6 # improves numerical stability
			if data1:
				self.downsample_rate1 = downsample_rate1
				if self.downsample_rate1 > 1:
					self.data1 = dec(self.data1, downsample_rate1) 
					self.downsampled[0] = True
				
				self.data1 = fnc(self.data1)
				if norm:
					self.data1 = normalize(self.data1)
					self.normalized[0] = True

			if data2:
				self.downsample_rate2 = downsample_rate2
				if self.downsample_rate2 > 1:
					self.data2 = dec(self.data2, downsample_rate2) 
					self.downsampled[1] = True
				
				self.data2 = fnc(self.data2)
				if norm:
					self.data2 = normalize(self.data2)
					self.normalized[1] = True
		

	def get_classnames(self, classes):
		"""
		Returns sub-group of classnames from a global list of class names. List of
		class names passed as a pd.DataFrame with column names == class names
		labels corresponding to trials are associated with values in a dict.
		:return: list of class names to object
		"""
		labels_dict, _ = labels_dict_and_list(classes)
		for i in np.unique(self.labels1):
			self.classnames.append(labels_dict[str(i)])

	def clear_data(self):
		"""
		Reset to empty data structures.
		"""
		self.data1 = np.array([])
		self.data2 = np.array([])
		self.data_combined = None
		self.labels1 = []
		self.labels2 = []
		self.labels_combined = None

		self.epoched_data1 = None
		self.epoched_data2 = None

	def save_subject(self, path, filename):
		"""
		Save the subject object as a pickle.

		:param path: (str) path to saving directory
		:param filename: (str) name to save object as
		"""
		if not os.path.exists(path):
			print("Creating new subject file...")
			os.makedirs(path)
		filename = f"{path}/{filename}.pickle"
		filehandler = open(filename, 'wb')
		pickle.dump(self.__dict__, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
		print(f"Data object saved to: '{filename}'\n")

	def update(self,newdata):
	    for key,value in newdata.items():
	        setattr(self,key,value)

	@classmethod
	def load_subject(self, f_name):
		 with open(f_name, 'rb') as f:
		 	tmp_dict = pickle.load(f)
	 		f.close()
	 		self.update(self, tmp_dict)
 			return self

	def get_details(self):
		print(f"Subject: {self.id}")
		print("-"*15)
		print(self.description)
		print("-"*15)
		if self.data1.size != 0:
			print(f"Data 1 shape: {self.data1.shape}")
			print(f"Labels 1 shape: {len(self.labels1)}")
			print(f"Class names: {self.classnames}")
			print(f"Number of  valid channels: {self.data1.shape[1]}")
			print(f"Sampling Frequency: {self.sfreq1} Hz")
			print(f"Data downsampled: {self.downsampled[0]}")
			if self.downsampled[0]:
				print(f"Downsample Rate: {self.downsample_rate1}")
			if self.normalized[0]:
				print(f"Data normalized: {self.normalized[0]}")
			if self.filtered[0]:
				print(f"Data bandpass filtered between {self.lowcut} and {self.highcut} Hz")
			if self.epoched_data1 is not None:
				print(f"Classifier Window Size: {self.epoched_data1.shape}")
				print(f"Classifier Start Time: {self.classifier_start} seconds")
				print(f"Classifier End Time: {self.classifier_end} seconds\n")
		if self.data1.size != 0:
			print(f"Data 2 shape: {self.data2.shape}")
			print(f"Labels 2 shape: {len(self.labels2)}")
			print(f"Class names: {self.classnames}")
			print(f"Number of  valid channels: {self.data2.shape[1]}")
			print(f"Sampling Frequency: {self.sfreq2} Hz")
			print(f"Data downsampled: {self.downsampled[1]}")
			if self.downsampled[1]:
				print(f"Downsample Rate: {self.downsample_rate1}")
			print(f"Data normalized: {self.normalized[1]}")
			if self.filtered[1]:
				print(f"Data bandpass filtered between {self.lowcut} and {self.highcut} Hz")
			if self.epoched_data2 is not None:
				print(f"Classifier Window Size: {self.epoched_data1.shape}")