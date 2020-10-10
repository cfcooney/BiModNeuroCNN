from BiModNeuroCNN.results.dataframe_utils import results_df, get_col_list, param_scores_df
from BiModNeuroCNN.utils import load_pickle
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import h5py

class Results():

	direct = 'C:/Users/sb00745777/OneDrive - Ulster University/Study_3/Subject_Data'

	def __init__(self, save_path, folds=5, tag='', name="A"):
		self.save_path = save_path
		self.n_folds = folds
		self.y_true_list = []
		self.y_true = np.array([])
		self.y_pred_list = []
		self.y_pred = np.array([])
		self.y_probs = None
		self.results_path = None
		self.lossdf = None
		self.accdf = None
		self.cross_entropydf = None
		self.subject_stats_df = None
		self.best_params = None
		self.hyp_param_means = []
		self.outer_fold_accuracies = []  # list of scores - 1 per fold
		self.outer_fold_cross_entropies = []
		self.of_mean = None
		self.of_std = None
		self.accuracy = None
		self.precision = None
		self.f1_score = None
		self.recall = None
		self.kappa = None
		self.precision_list = []
		self.f1_score_list = []
		self.recall_list = []
		self.kappa_list = []
		self.cm = None
		self.train_loss = None
		self.test_loss = None
		self.valid_loss = None
		self.train_acc = None
		self.test_acc = None
		self.valid_acc = None
		self.tag = tag
		self.id = name

	def __repr__(self):
		return f"<class 'BiModNeuroCNN.results.results.Results'>"

	def __str__(self):
		return f"Class for creating a Results object containing several metrics"

	def __getattr__(self, name):
		if name == "fold_accuracies":
			for i, j in enumerate(self.outer_fold_accuracies):
				print (f"Fold {i+1}: {j}%")
			folds_mean = np.mean(self.outer_fold_accuracies)
			return f"Mean: {folds_mean}%"
		elif name == "classes":
			unique, counts = np.unique(self.y_true, return_counts=True)
			return [f"Class {u}:{c}" for u,c in zip(unique, counts)]
		elif name == "predictions":
			unique, counts = np.unique(self.y_pred, return_counts=True)
			return [f"Class {u}:{c}" for u,c in zip(unique, counts)]
		elif name in dir(self):
			return name
		else:
			print(dir(self))
			raise AttributeError(f"'{name}' does not exist in this context")

	def __setattr__(self, name, value):
		if name == "fold_accuracies":
			self.outer_fold_accuracies = value
		else:
			super().__setattr__(name, value)

	def change_directory(self, direct):
		self.direct = direct

	def set_results_path(self, results_path):
		self.results_path = results_path

	def concat_y_true(self, y_true_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_true_fold: array of single-fold true values.
		:return: all outer fold true values in single arrau
		"""
		self.y_true = np.concatenate((self.y_true, np.array(y_true_fold)))

	def concat_y_pred(self, y_pred_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_pred_fold: array of single-fold true values.
		:return: all outer fold true values in single arrau
		"""
		self.y_pred = np.concatenate((self.y_pred, np.array(y_pred_fold)))

	def append_y_true(self, y_true_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_true_fold: array of single-fold true values.
		:return: list of outer fold true values. Each element contains one fold
		"""
		self.y_true_list.append((np.array(y_true_fold)))

	def append_y_pred(self, y_pred_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_pred_fold: array of single-fold true values.
		:return: list of outer fold true values. Each element contains one fold
		"""
		self.y_pred_list.append((np.array(y_pred_fold)))

	def get_acc_loss_df(self, hyp_params, index_name, nested=True):
		"""
		Instantiates pd.DataFrames for storing accuracy or loss metrics for each fold
		and hyperparameter set.
		:param hyp_params (dict) keys: names of hyp_params, values: lists of HP values 
		:param index_name (str) index name for dataframe
		"""
		if nested:
			index = list(n+1 for n in range(self.n_folds*self.n_folds))
		else:
			index = list(n+1 for n in range(self.n_folds))
		index.append("Mean")
		index.append("Std.")
		columns_list = get_col_list(hyp_params)

		names = list(hyp_params.keys())

		self.lossdf = results_df(index,index_name,columns_list,names)
		self.accdf  = results_df(index,index_name,columns_list,names)
		self.cross_entropydf = results_df(index,index_name,columns_list,names)


	def fill_acc_loss_df(self, inner_fold_accs=None, inner_fold_loss=None, inner_fold_CE=None, save=True):
		"""
		Method for inserting all inner-fold accuracies and losses associated with each hyper-parameter
		combination in a dataframe. Mean and Std. computed. The dataframes can be used to select optimal
		hyper-parameters.
		:param inner_fold_accs: list containing all inner-fold accuracy scores
		:param inner_fold_loss: list containing all inner-fold loss values
		:param inner_fold_CE: list containing all inner-fold CE values
		:param save: Boolean
		:return: Dataframes in which each column represents a particular hyper-parameter set.
		"""
		if inner_fold_accs is not None:
			for n, acc in enumerate(inner_fold_accs):
				self.accdf.iloc[n] = acc
			self.accdf.loc["Mean"].iloc[0] = self.accdf.iloc[1:n].mean(axis=0).values
			self.accdf.loc["Std."].iloc[0] = self.accdf.iloc[1:n].std(axis=0).values
			if save:
				try:
					self.accdf.to_excel(f"{self.save_path}/HP_acc{self.tag}.xlsx")
				except:
					self.accdf.to_excel(f"{self.save_path}/HP_acc{self.tag}.xlsx", engine='xlsxwriter')

		if inner_fold_loss is not None:
			for n, loss in enumerate(inner_fold_loss):
				self.lossdf.iloc[n] = loss
			self.lossdf.loc["Mean"].iloc[0] = self.lossdf.iloc[1:n].mean(axis=0).values
			self.lossdf.loc["Std."].iloc[0] = self.lossdf.iloc[1:n].std(axis=0).values
			if save:
				try:
					self.lossdf.to_excel(f"{self.save_path}/HP_loss{self.tag}.xlsx")
				except:
					self.lossdf.to_excel(f"{self.save_path}/HP_loss{self.tag}.xlsx", engine='xlsxwriter')

		if inner_fold_CE is not None:
			for n, ce in enumerate(inner_fold_CE):
				self.cross_entropydf.iloc[n] = ce
			self.cross_entropydf.loc["Mean"].iloc[0] = self.cross_entropydf.iloc[1:n].mean(axis=0).values
			self.cross_entropydf.loc["Std."].iloc[0] = self.cross_entropydf.iloc[1:n].std(axis=0).values
			if save:
				try:
					self.cross_entropydf.to_excel(f"{self.save_path}/HP_CE{self.tag}.xlsx")
				except:
					self.cross_entropydf.to_excel(f"{self.save_path}/HP_CE{self.tag}.xlsx", engine='xlsxwriter')


	def get_best_params(self, selection_method, save_path=None):
		"""
		Method for returning best hyper-parameter combination from inner fold accuracy or loss.
		:param selection_method: str: "accuracy" Or "loss".
		:return: list of optimal hyper-parameters.
		"""
		if save_path == None: # can overwrite object save_path with argument if required
			save_path = self.save_path

		if selection_method == "accuracy":
			self.best_params = list(self.accdf.columns[self.accdf.loc["Mean"].values.argmax()])
		else:
			self.best_params = list(self.lossdf.columns[self.lossdf.loc["Mean"].values.argmin()])
		best_params = pd.DataFrame(dict(best_params=self.best_params))
		try:
			best_params.to_excel(f"{save_path}/BestParameters{self.tag}.xlsx")
		except:
			best_params.to_excel(f"{save_path}/BestParameters{self.tag}.xlsx", engine='xlsxwriter') # occasional problems with writing

	
	def get_hp_means(self, hyp_params, selection_method, save=False, save_path=None):
		"""
		Extracts mean accuracies for specific HP values (as opposed to HP sets)

		:param hyp_params (dict) keys: names of hyp_params, values: lists of HP values 
		:param selection_method (str) 'accuracy' OR 'loss'
		"""
		if save_path == None: # can overwrite object save_path with argument if required
			save_path = self.save_path

		columns_list = get_col_list(hyp_params)
		for HP in columns_list:
			
			for value in HP:
				if selection_method == 'accuracy':
					sub_df = self.accdf[[i for i in self.accdf.columns if i[0] == value or i[1] == value or i[2] == value or i[3] == value]] 
					self.hyp_param_means.append((value, sub_df.loc["Mean"].values.mean()))
				else:
					sub_df = self.lossdf[[i for i in self.lossdf.columns if i[0] == value or i[1] == value or i[2] == value or i[3] == value]] 
					self.hyp_param_means.append((value, sub_df.loc["Mean"].values.mean()))
		if save:
			hp_val_list, hp_mean_list = [], []
			for tup in self.hyp_param_means:
				hp_val_list.append(tup[0])
				hp_mean_list.append(tup[1])
			hp_means_df = pd.DataFrame(dict(HP_value=hp_val_list, HP_mean=hp_mean_list))
			try:
				hp_means_df.to_excel(f"{save_path}/HP_means{self.tag}.xlsx")
			except:
				hp_means_df.to_excel(f"{save_path}/HP_means{self.tag}.xlsx", engine='xlsxwriter') # occasional problems with writing


	def set_outer_fold_accuracies(self, outer_fold_accuracies):
		self.outer_fold_accuracies = outer_fold_accuracies
		self.of_mean = np.mean(outer_fold_accuracies)
		self.of_std = np.std(outer_fold_accuracies)

	def get_accuracy(self):
		"""
		Method for calculating accuracy from all true and predicted values.
		:return: accuracy value (%) rounded to 3 decimal places.
		"""
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.accuracy = round((accuracy_score(self.y_true, self.y_pred) * 100), 3)

	def get_precision(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.precision = round((precision_score(self.y_true, self.y_pred, average="macro") * 100), 3)

	def get_recall(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.recall = round((recall_score(self.y_true, self.y_pred, average='macro') * 100), 3)

	def get_f_score(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.f1_score = round((f1_score(self.y_true, self.y_pred, average='macro') * 100), 3)

	def get_kappa_value(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.kappa = round(cohen_kappa_score(self.y_true, self.y_pred),3)

	def precision_recall_f_score(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		precision_recall_fscore_support(self.y_true, self.y_pred)

	def confusion_matrix(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.cm = confusion_matrix(self.y_true, self.y_pred)

	def subject_stats(self):
		"""
		Method for constructing and saving a Pandas Dataframe with Accuracy and
		statistical scores as below:
			fold 1  fold 2    Mean   Std.  Precision  Recall  F1 Score
		01  18.065  16.779  17.422  0.643     16.447  16.447    16.447
		"""

		folds = []
		for i in range(1, self.n_folds+1):
			folds.append(f'fold {i}')

		if np.array(self.outer_fold_accuracies).ndim == 1:
			self.subject_stats_df = pd.DataFrame(index=[self.id], columns=folds)
			self.subject_stats_df.iloc[0] = self.outer_fold_accuracies
			self.get_accuracy()
			self.subject_stats_df['Subj Mean'] = self.subject_stats_df.mean(axis=1, skipna=True)
			self.subject_stats_df['Subj Std.'] = self.subject_stats_df.drop('Subj Mean',axis=1).std(axis=1, skipna=True)
			self.get_precision()
			self.get_recall()
			self.get_f_score()
			self.subject_stats_df['Precision'] = self.precision
			self.subject_stats_df['Recall'] = self.recall
			self.subject_stats_df['F1 Score'] = self.f1_score
			for n,ce in enumerate(self.outer_fold_cross_entropies):
				self.subject_stats_df[f"CE - fold {n+1}"] = ce
			self.subject_stats_df["CE mean"] = np.mean(self.outer_fold_cross_entropies)
			self.subject_stats_df["CE std."] = np.std(self.outer_fold_cross_entropies)

			handle = f"{self.save_path}/statistics{self.tag}.xlsx"
			
		else:
			self.subject_stats_df = pd.DataFrame(index=[self.ids], columns=folds)
			for n,score in enumerate(self.outer_fold_accuracies):
				self.subject_stats_df.iloc[n] = score
			self.subject_stats_df['Subj Mean'] = self.subject_stats_df.mean(axis=1, skipna=True)
			self.subject_stats_df['Subj Std.'] = self.subject_stats_df.drop('Subj Mean',axis=1).std(axis=1, skipna=True)
			self.subject_stats_df['Precision'] = self.precision_list
			self.subject_stats_df['Recall'] = self.recall_list
			self.subject_stats_df['F1 Score'] = self.f1_score_list

			# adding cross-entropy values for each fold
			for n,_ in enumerate(folds):
				self.subject_stats_df[f"CE - fold {n+1}"] = ""
			for n,ce_list in enumerate(self.outer_fold_cross_entropies):
				for m,ce in enumerate(ce_list):
					self.subject_stats_df[f"CE - fold {m+1}"].iloc[n] = ce
			self.subject_stats_df["CE mean"] = self.outer_fold_ce_means
			self.subject_stats_df["CE std."] = self.outer_fold_ce_std

			self.subject_stats_df.loc["Mean"] = self.subject_stats_df.iloc[0:len(self.ids)].mean(axis=0).values
			self.subject_stats_df.loc["Std."] = self.subject_stats_df.iloc[0:len(self.ids)].std(axis=0).values

			handle = f"{self.results_path}/combined_stats{self.tag}.xlsx"
		try:
			self.subject_stats_df.to_excel(handle)
		except:
			self.subject_stats_df.to_excel(handle, engine='xlsxwriter')


	def save_result(self):
		filename = f"{self.save_path}/results_object{self.tag}.pickle"
		filehandler = open(filename, 'wb')
		try:
			pickle.dump(self.__dict__, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
		except ValueError:
			file = f"{self.save_path}/results_object_alt_{self.tag}"
			self.save_as_pickled_object(file)
	
	def update(self, newdata):
	    for key,value in newdata.items():
	    	setattr(self,key,value)

	@classmethod
	def load_result(self, f_name):
		 with open(f_name, 'rb') as f:
		 	tmp_dict = pickle.load(f)
	 		f.close()
	 		self.update(self, tmp_dict)
 			return self

	def save_as_pickled_object(self, filepath):
		"""
        This is a defensive way to write pickle.write, allowing for very large files on all platforms
        """
		subject = dict(subject=self)
		max_bytes = 2 ** 31 - 1
		bytes_out = pickle.dumps(subject)
		n_bytes = sys.getsizeof(bytes_out)
		with open(filepath, 'wb') as f_out:
			for idx in range(0, n_bytes, max_bytes):
				f_out.write(bytes_out[idx:idx + max_bytes])

	def try_to_load_as_pickled_object_or_None(filepath):
		"""
        This is a defensive way to write pickle.load, allowing for very large files on all platforms
        """
		max_bytes = 2 ** 31 - 1
		try:
			input_size = os.path.getsize(filepath)
			bytes_in = bytearray(0)
			with open(filepath, 'rb') as f_in:
				for _ in range(0, input_size, max_bytes):
					bytes_in += f_in.read(max_bytes)
			obj = pickle.loads(bytes_in)
		except:
			return None
		return obj


class CombinedResults(Results):
	"""
	Written for combining the results of multiple subject/experiments.
	"""

	def __init__(self, save_path, load_path, f_names, folds, ids, tag):

		super().__init__(save_path, folds, tag)

		self.load_path = load_path
		self.f_names = f_names
		self.ids = ids
		self.total_cross_val_df = None
		self.total_best_hps = [] #list of best HPs for each subject
		self.BestParams = None
		self.hp_results_df = None
		self.outer_fold_ce_means = []
		self.outer_fold_ce_std = []
		self.combined_train_loss = []
		self.combined_test_loss = []
		self.combined_valid_loss = []
		self.combined_train_acc = []
		self.combined_test_acc = []
		self.combined_valid_acc = []
		self.HP_acc = pd.DataFrame(columns=self.ids)
		self.HP_loss = pd.DataFrame(columns=self.ids)
		self.HP_ce = pd.DataFrame(columns=self.ids)
		self.total_number = 0

	def __repr__(self):
		return f"<class 'BiModNeuroCNN.results.results.CombinedResults'>"

	def __str__(self):
		return f"Class for combining the results from multiple subjects/experiments"

	def __len__(self):
		return len(self.f_names)

	def __getattr__(self, name):
		if name == "all_ids":
			return [(n+1, i) for n, i in enumerate(self.ids[:-2])]

	def cross_val_results_df(self, accuracy=True, cross_entropy=False, save=True):
		"""
		Combine all results into single pd.DataFrame, calculate mean and stdev. and store
		in Excel format
		
		:param: accuracy (bool) True if accuracy scores to be considered
		:param: cross entropy (bool) True if cross entropy scores to be considered
		:param: save (bool) True if results are to be stored as xlsx
		"""
		folds = []
		for i in range(1, self.n_folds+1):
			folds.append(f'fold {i}')

		if accuracy:
			assert len(self.outer_fold_accuracies) != [], "Scores must be loaded to CombinedResults.outer_fold_accuracies" 
			assert len(self.outer_fold_accuracies) == len(self.ids), "Number of subjects and results are not equal"
			assert len(self.outer_fold_accuracies[0]) == self.n_folds, "Number of scores and folds are not equal"

			self.total_acc_df = pd.DataFrame(index=self.ids, columns=folds)

			for n,score in enumerate(self.outer_fold_accuracies):
				self.total_acc_df.iloc[n] = score

			self.total_acc_df['Mean'] = self.total_acc_df.mean(axis=1,skipna=True)
			self.total_acc_df['Std.'] = self.total_acc_df.drop('Mean',axis=1).std(axis=1,skipna=True)

		if cross_entropy:
			assert len(self.outer_fold_cross_entropies) != [], "Scores must be loaded to CombinedResults.outer_fold_accuracies" 
			assert len(self.outer_fold_cross_entropies) == len(self.ids), "Number of subjects and results are not equal"
			assert len(self.outer_fold_cross_entropies[0]) == self.n_folds, "Number of scores and folds are not equal"

			self.total_ce_df = pd.DataFrame(index=self.ids, columns=folds)

			for n,score in enumerate(self.outer_fold_cross_entropies):
				self.total_ce_df.iloc[n] = score

			self.total_ce_df['Mean'] = self.total_ce_df.mean(axis=1,skipna=True)
			self.total_ce_df['Std.'] = self.total_ce_df.drop('Mean',axis=1).std(axis=1,skipna=True)
	
		if save:
			if accuracy and cross_entropy:
				with pd.ExcelWriter(f'{self.save_path}/combined_scores.xlsx') as writer:
					self.total_acc_df.to_excel(writer, sheet_name='accuracy')
					self.total_ce_df.to_excel(writer, sheet_name='cross_entropy')
			elif not cross_entropy:
				with pd.ExcelWriter(f'{self.save_path}/combined_scores.xlsx') as writer:
					self.total_acc_df.to_excel(writer, sheet_name='accuracy')
			elif not accuracy:
				with pd.ExcelWriter(f'{self.save_path}/combined_scores.xlsx') as writer:
					self.total_ce_df.to_excel(writer, sheet_name='cross_entropy')

	def get_subject_results(self):
		"""
		Read in multiple results.Results objects and extract the required metrics into containers
		for further processing.
		"""

		for f_name in self.f_names:

			results_object = self.load_result(f"{self.load_path}/{f_name}.pickle")

			self.y_true = np.concatenate((self.y_true, results_object.y_true))
			self.y_pred = np.concatenate((self.y_pred, results_object.y_pred)) # all true and prediction values

			self.outer_fold_accuracies.append(results_object.outer_fold_accuracies)
			self.outer_fold_cross_entropies.append(results_object.outer_fold_cross_entropies)
			self.outer_fold_ce_means.append(np.mean(results_object.outer_fold_cross_entropies))
			self.outer_fold_ce_std.append(np.std(results_object.outer_fold_cross_entropies))

			results_object.precision = self.get_res_obj_precision(results_object)
			results_object.f_score = self.get_res_obj_recall(results_object)
			results_object.recall = self.get_res_obj_f_score(results_object)
			results_object.kappa = self.get_res_obj_kappa_value(results_object)
			self.precision_list.append(results_object.precision)
			self.f1_score_list.append(results_object.f1_score)
			self.recall_list.append(results_object.recall)
			self.kappa_list.append(results_object.kappa)

			self.total_best_hps.append(results_object.best_params)
			self.hyp_param_means.append(results_object.hyp_param_means)

			self.combined_train_loss.append(results_object.train_loss)
			self.combined_test_loss.append(results_object.test_loss)
			self.combined_valid_loss.append(results_object.valid_loss)
			self.combined_train_acc.append(results_object.train_acc)
			self.combined_test_acc.append(results_object.test_acc)
			self.combined_valid_acc.append(results_object.valid_acc)
	
		# Save combined predictions and ground truth values to csv
		# np.savetxt(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/y_true.csv", [self.y_true],
		# 		   delimiter=',', fmt='%d')
		# np.savetxt(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/y_pred.csv", [self.y_pred],
		# 		   delimiter=',', fmt='%d')

	
	def param_scores(self, hyp_params):
		"""
		Saves a Pandas DataFrame as an Excel file which contains average inner-fold accuracy (or loss)
		for each independent hyperparameter value, and for all subjects
		:param hyp_params: dict containing all hyperparameter keys and values.
		"""
		paramscores_df = param_scores_df(self.ids, hyp_params)
	
		for i, j in enumerate(self.hyp_param_means):
			paramscores_df.iloc[i] = [score[1] for score in j]

		paramscores_df.loc["Mean"] = paramscores_df[0:len(self.ids)].mean(axis=0, skipna=True)
		paramscores_df.loc["Std."] = paramscores_df[0:len(self.ids)].std(axis=0, skipna=True)
		paramscores_df.to_excel(f"{self.save_path}/param_scores.xlsx")

	
	def inter_subject_hps(self, hyp_params, index_name, selection_method):
		"""
		Saves a Pandas DataFrame as an Excel file which contains average inner-fold accuracy (or loss)
		for each independent hyperparameter value, and for all subjects
		:param hyp_params: dict containing all hyperparameter keys and values.
		:param index_name: str name to give index column.
		:param selection_method: str "accuracy" OR "loss".
		"""
		index = self.ids
		columns_list = get_col_list(hyp_params)
		names = list(hyp_params.keys())
		
		self.hp_results_df = results_df(index, index_name, columns_list, names)
		
		combined_hp = []
		for f_name in self.f_names:

			results_object = self.load_result(f"{self.load_path}/{f_name}.pickle")
		
			acc = results_object.accdf.loc['Mean'].values
			combined_hp.append(acc)
		
		for i, j in enumerate(combined_hp):
			self.hp_results_df.iloc[i] = j
		self.hp_results_df.loc["Mean"].iloc[0] = self.hp_results_df.iloc[0:len(self.ids)].mean(axis=0, skipna=True)
		self.hp_results_df.loc["Std."].iloc[0] = self.hp_results_df.iloc[0:len(self.ids)].std(axis=0, skipna=True)
		self.hp_results_df.to_excel(f"{self.save_path}/total_hp_scores.xlsx")

		self.BestParams = self.hp_results_df.columns[self.hp_results_df.loc["Mean"].values.argmax()]
		self.BestParams = pd.DataFrame(dict(BestParams=self.BestParams))
		self.BestParams.to_excel(f"{self.save_path}/BestParams.xlsx")

	
	def get_combined_inner_scores(self):
		"""
		Create pd.DataFrames to contain inner-fold validation accuracies/loss/cross entropy
		for all subjects or experiments and compute a mean - can be used for selecting inter-subject
		hyperparameters. 
		"""

		for i, f_name in enumerate(self.f_names):

			results_object = self.load_result(f"{self.load_path}/{f_name}.pickle")

			self.HP_acc[self.ids[i]]  = results_object.accdf.loc['Mean'].apply(lambda x : x * 100).values.ravel()
			self.HP_loss[self.ids[i]] = results_object.lossdf.loc['Mean'].values.ravel()
			self.HP_ce[self.ids[i]]   = results_object.cross_entropydf.loc['Mean'].values.ravel()

		self.HP_acc.fillna(0, inplace=True) # zero-filling -- mean-filling may be a better option
		self.HP_loss.fillna(0, inplace=True)
		self.HP_ce.fillna(0, inplace=True)
		self.HP_acc['Mean'] = self.HP_acc.mean(axis=1, skipna=True)
		self.HP_loss['Mean'] = self.HP_loss.mean(axis=1, skipna=True)
		self.HP_ce['Mean'] = self.HP_ce.mean(axis=1, skipna=True)

	@staticmethod
	def get_res_obj_precision(res_obj):
		assert len(res_obj.y_true) == len(res_obj.y_pred), "data must be of equal length"
		return round((precision_score(res_obj.y_true, res_obj.y_pred, average="macro") * 100), 3)

	@staticmethod
	def get_res_obj_recall(res_obj):
		assert len(res_obj.y_true) == len(res_obj.y_pred), "data must be of equal length"
		return round((recall_score(res_obj.y_true, res_obj.y_pred, average='macro') * 100), 3)

	@staticmethod
	def get_res_obj_f_score(res_obj):
		assert len(res_obj.y_true) == len(res_obj.y_pred), "data must be of equal length"
		return round((f1_score(res_obj.y_true, res_obj.y_pred, average='macro') * 100), 3)

	@staticmethod
	def get_res_obj_kappa_value(res_obj):
		assert len(res_obj.y_true) == len(res_obj.y_pred), "data must be of equal length"
		return round(cohen_kappa_score(res_obj.y_true, res_obj.y_pred),3)