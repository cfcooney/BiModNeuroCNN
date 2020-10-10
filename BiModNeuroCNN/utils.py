"""
Name: Ciaran Cooney
Date: 12/01/2019
Description: Functions required for data processing and training of 
CNNs on imagined speech EEG data.
"""

import pickle
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import wraps

def eeg_to_3d(data, epoch_size, n_events,n_chan):
    """
    function to return a 3D EEG data format from a 2D input.
    Parameters:
      data: 2D np.array of EEG
      epoch_size: number of samples per trial, int
      n_events: number of trials, int
      n_chan: number of channels, int
        
    Output:
      np.array of shape n_events * n_chans * n_samples
    """
    idx, a, x = ([] for i in range(3))
    [idx.append(i) for i in range(0,data.shape[1],epoch_size)]
    for j in data:
        [a.append([j[idx[k]:idx[k]+epoch_size]]) for k in range(len(idx))]
   
    
    return np.reshape(np.array(a),(n_events,n_chan,epoch_size))

def load_pickle(direct, folder, filename):
    
    for file in os.listdir(direct + folder):
        if file.endswith(filename):
            pickle_file = (direct + folder + '/' + file)
            with open(pickle_file, 'rb') as f:
                file = pickle.load(f)

            return file, pickle_file

def create_events(data, labels):
    events = []
    x = np.zeros((data.shape[0], 3))
    for i in range(data.shape[0]):
        x[i][0] = i 
        x[i][2] = labels[i]
    [events.append(list(map(int, x[i]))) for i in range(data.shape[0])]
    return np.array(events)

def reverse_coeffs(coeffs, N):
    """ Reverse order of coefficients in an array."""
    idx = np.array([i for i in reversed(range(N))])
    coeffs = coeffs[idx]
    coeffs = coeffs.reshape((N,1))
    z = np.zeros((N,1))
    return np.append(coeffs, z, axis=1) , coeffs

def class_ratios(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_weight = dict()
    for i in range(len(unique)):
       class_weight[unique[i]] = len(labels) / (len(unique)*counts[i])
    return class_weight

def classification_report_csv(report, output_file):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(output_file + '.csv', index = False)

def load_features(direct, dict_key1, dict_key2=None):
    with open(direct, 'rb') as f:
        file = pickle.load(f)
    if dict_key2 == None:
        return np.array(file[dict_key1])
    else:
        return np.array(file[dict_key1]), np.array(file[dict_key2])

def short_vs_long(features, labels, split, event_id):
    """Function for multilabel data into binary-class sets i.e.,
       short words and long words
    """
    short, long, s_idx, l_idx, s_features, l_features = ([] for i in range(6))
    
    [short.append(event_id[i]) for i in event_id if len(i) <= split]
    [long.append(event_id[i]) for i in event_id if len(i) > split]
    
    [s_idx.append(i) for i, e in enumerate(labels) if e in short]
    [l_idx.append(i) for i, e in enumerate(labels) if e in long]
    
    [s_features.append(e) for i, e in enumerate(features) if i in s_idx]
    [l_features.append(e) for i, e in enumerate(features) if i in l_idx]
    
    s_labels = np.zeros(np.array(s_features).shape[0])
    l_labels = np.ones(np.array(l_features).shape[0])

    features = np.concatenate((s_features, l_features))
    labels = np.concatenate((s_labels,l_labels))
    
    return s_features, l_features, s_labels, l_labels, features, labels 

def return_indices(event_id, labels):
    indices = []
    for _, k in enumerate(event_id):
        idx = []
        for d, j in enumerate(labels):
            if event_id[k] == j:
                idx.append(d)
        indices.append(idx)
    return indices

def load_subject_eeg(subject_id, vowels):
    """ returns eeg data corresponding to words and vowels 
        given a subject identifier.
    """

    data_folder = 'C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    data_folder1 = 'C:\\Users\\cfcoo\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    words_file = 'raw_array_ica.pickle'
    vowels_file = 'raw_array_vowels_ica.pickle'
    
    try:
        with open(data_folder + words_file, 'rb') as f:
            file = pickle.load(f)
    except:
        print("Not on PC! Attempting to load from laptop.")
        with open(data_folder1 + words_file, 'rb') as f:
            file = pickle.load(f)
            
    w_data = file['raw_array'][:][0]
    w_labels = file['labels']
    if vowels == False:
        return w_data, w_labels

    elif vowels:
        try:
            with open(data_folder + vowels_file, 'rb') as f:
                file = pickle.load(f)
        except:
            with open(data_folder1 + vowels_file, 'rb') as f:
                file = pickle.load(f)
        v_data = file['raw_array'][:][0]
        v_labels = file['labels']
    return w_data, v_data, w_labels, v_labels

def balanced_subsample(features, targets, random_state=12):
    """
    function for balancing datasets by randomly-sampling data
    according to length of smallest class set.
    """
    from sklearn.utils import resample
    unique, counts = np.unique(targets, return_counts=True)
    unique_classes = dict(zip(unique, counts))
    mnm = len(targets)
    for i in unique_classes:
        if unique_classes[i] < mnm:
            mnm = unique_classes[i]

    X_list, y_list = [],[]
    for unique in np.unique(targets):
        idx = np.where(targets == unique)
        X = features[idx]
        y = targets[idx]
        
        #X1, y1 = resample(X,y,n_samples=mnm, random_state=random_state)
        X_list.append(X[:mnm])
        y_list.append(y[:mnm])
    
    balanced_X = X_list[0]
    balanced_y = y_list[0]
    
    for i in range(1, len(X_list)):
        balanced_X = np.concatenate((balanced_X, X_list[i]))
        balanced_y = np.concatenate((balanced_y, y_list[i]))

    return balanced_X, balanced_y

def predict(model, X_test, batch_size, iterator, threshold_for_binary_case=None):
    """
    Load torch model and make predictions on new data.
    """
    all_preds = []
    with th.no_grad():
        for b_X, _ in iterator.get_batches(SignalAndTarget(X_test, X_test), False):
            b_X_var = np_to_var(b_X)
            all_preds.append(var_to_np(model(b_X_var)))

        pred_labels = compute_pred_labels_from_trial_preds(
                    all_preds, threshold_for_binary_case)
    return pred_labels

def plot_confusion_matrix(cm, classes,filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    Code for confusion matrix extracted from here:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(1, figsize=(9, 6))
    #ax = plt.add_subplot(111)
    plt.tick_params(labelsize='large')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize='large', fontname='sans-serif')
    plt.xlabel('Predicted label', fontsize='large', fontname='sans-serif')
    fig.savefig(filename + '.jpg', bbox_inches='tight')
    return(fig)

def print_confusion_matrix(confusion_matrix, class_names, filename, normalize = True, figsize = (5,5), fontsize=16):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    fmt = '.2f' if normalize else 'd'
    #####set heatmap customization#####
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap='GnBu', linewidths=.5, cbar=False, annot_kws={"size": 16})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=16, fontname='sans-serif')
    plt.xlabel('Predicted label', fontsize=16, fontname='sans-serif')
    
    if filename != None:
        fig.savefig(filename + '.png', bbox_inches='tight') #store image as .png
    
    return fig

def data_wrangler(data_type, subject_id):
    """
    Function to return EEG data in format #trials*#channels*#samples.
    Also returns labels in the range 0 to n-1.
    """
    epoch = 4096
    if data_type == 'words':
        data, labels = load_subject_eeg(subject_id, vowels=False)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
    elif data_type == 'vowels':
        _, data, _, labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
    elif data_type == 'all_classes':
        w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(w_data)
        words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
        vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
        data = np.concatenate((words, vowels), axis=0)
        labels = np.concatenate((w_labels, v_labels), axis=0).astype(np.int64)
    
    x = lambda a: a * 1e6
    data = x(data)
    
    if data_type == 'words': # zero-index the labels
        labels[:] = [x - 6 for x in labels]
    elif (data_type == 'vowels' or data_type == 'all_classes'):
        labels[:] = [x - 1 for x in labels]

    return data, labels


def format_data(data_type, subject_id, epoch):
    """
    Returns data into format required for inputting to the CNNs.

    Parameters:
        data_type: str()
        subject_id: str()
        epoch: length of single trials, int
    """

    if data_type == 'words':
        data, labels = load_subject_eeg(subject_id, vowels=False)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 6 for x in labels] # zero-index the labels
    elif data_type == 'vowels':
        _, data, _, labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 1 for x in labels]
    elif data_type == 'all_classes':
        w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(w_data)
        words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
        vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
        data = np.concatenate((words, vowels), axis=0)
        labels = np.concatenate((w_labels, v_labels)).astype(np.int64)
        labels[:] = [x - 1 for x in labels]

    return data, labels

def current_loss(model_loss):
    """
    Returns the minimum validation loss from the 
    trained model
    """
    losses_list = []
    [losses_list.append(x) for x in model_loss]
    return np.min(np.array(losses_list))

def current_acc(model_acc):
    """
    Returns the maximum validation accuracy from the 
    trained model
    """
    accs_list = []
    [accs_list.append(x) for x in model_acc]
    return np.min(np.array(accs_list))

def balance_classes(data1,data2):

    if data1.shape[0] > data2.shape[0]:
        data1 = data1[:data2.shape[0],:,:]
    elif data1.shape[0] < data2.shape[0]:
        data2 = data2[:data1.shape[0],:,:]
        
    return data1, data2

def timer(orig_func):
    """
    decorator for logging time of function.
    """
    import time
    
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, *kwargs)
        t2 = time.time() - t1
        print(f"{orig_func.__name__} ran in: {round(t2,3)} seconds")
        return result
    
    return wrapper

def windows(trial_data, sub, window_size, overlap, fs):
    """
    Functon for obtaining classification windows for training.

    :param trial_data: EEG data - n_trials * n_chans * n_samples
    :param sub: subject object
    :param window_size: n number of samples
    :param overlap: n number of samples for overlap
    :param fs: sampling frequency
    :return: list containing data from each window
    """
    windows_list, index_list = [],[]
    n_windows = int(sub.epoch / window_size + np.floor((sub.epoch - overlap) / window_size))
    if n_windows == 0:
        n_windows = 1
    low_index = 0
    high_index = window_size
    for w in range(n_windows):
        data = trial_data[:, :, low_index:high_index]
        windows_list.append(data)
        index_list.append([low_index,high_index])
        low_index += overlap
        high_index += overlap

    return np.array(windows_list), index_list

def windows_index(epoch, window_size, overlap, fs):
    """
    Functon for obtaining classification windows for training.

    :param epoch: length of overal trial
    :param window_size: n number of samples
    :param overlap: n number of samples for overlap
    :param fs: sampling frequency
    :return: list containing data from each window
    """
    index_list = []
    n_windows = int(epoch / window_size + np.floor((epoch - overlap) / window_size))
    if n_windows == 0:
        n_windows = 1
    low_index = 0
    high_index = window_size
    for w in range(n_windows):
        index_list.append((low_index,high_index))
        low_index += overlap
        high_index += overlap

    return index_list

def get_class_labels(paradigm):
    """
    Function for obtaining class labels from paradigm description
    :param paradigm: string format: 'EEG_semantics_text'
    :return:
    """
    paradigm = paradigm.split('_')[1]
    if paradigm == 'semantics':
        class_labels = ['pig', 'dog', 'car', 'bus']
    elif paradigm == 'action':
        class_labels = ['kick', 'jump', 'chew', 'blink']
    elif paradigm == 'twoword':
        class_labels = ['red ball', 'blue hat', 'red blue', 'ball hat']
    elif paradigm == 'concrete':
        class_labels = ['apple', 'tiger', 'fruit', 'animal']
    return class_labels

def misclass_to_class(column):
    return 1 - column

def get_model_loss_and_acc(fold_models):
     """
     Function for extracting epoch-by-epoch model loss and accuracy scores from
     models associated with multiple cross-validation folds
     :param fold_models: list of Braindecode (PyTorch) sequential models
     :return: train_loss: (pandas.series) main training loss per epoch across folds
              valid_loss: (pandas.series) main tvalidation loss per epoch across folds
              test_loss: (pandas.series) main testing loss per epoch across folds
              train_acc: (pandas.series) main training acc per epoch across folds
              valid_acc: (pandas.series) main tvalidation acc per epoch across folds
              test_acc: (pandas.series) main testing acc per epoch across folds
     """
     train_loss = dict()
     valid_loss = dict()
     test_loss = dict()
     train_acc = dict()
     valid_acc = dict()
     test_acc = dict()

     for i, model in enumerate(fold_models):
        train_loss[i] = model.epochs_df['train_loss']
        valid_loss[i] = model.epochs_df['valid_loss']
        test_loss[i] = model.epochs_df['test_loss']
        train_acc[i] =  model.epochs_df['train_misclass']
        valid_acc[i] = model.epochs_df['valid_misclass']
        test_acc[i] = model.epochs_df['test_misclass']

     train_loss = pd.DataFrame(train_loss)
     valid_loss = pd.DataFrame(valid_loss)
     test_loss = pd.DataFrame(test_loss)
     train_loss = train_loss.mean(axis=1, skipna=True)
     valid_loss = valid_loss.mean(axis=1, skipna=True)
     test_loss = test_loss.mean(axis=1, skipna=True)

     train_acc = pd.DataFrame(train_acc).apply(lambda x : misclass_to_class(x)) # function converts misclass to classification accuracy
     valid_acc = pd.DataFrame(valid_acc).apply(lambda x : misclass_to_class(x))
     test_acc = pd.DataFrame(test_acc).apply(lambda x : misclass_to_class(x))
     train_acc = train_acc.mean(axis=1, skipna=True)
     valid_acc = valid_acc.mean(axis=1, skipna=True)
     test_acc = test_acc.mean(axis=1, skipna=True)

     return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc


def labels_dict_and_list(classes):
    """
    input: empty pandas DataFrame with column headings
           corresponding to class labels
    output: labels_dict (dict): key=number, value=string
            key_list (list): list of classes
    """

    labels_dict = dict()
    key_list = []
    for n, label in enumerate(classes.columns):
        labels_dict[str(n + 1)] = label

    for key in labels_dict:
        key_list.append(key)
    return labels_dict, key_list

# def data_loader(directory, subj, session, category, *args):
#     """

#     :param directory: (str) directory of stored data
#     :param subj: (str) Subject Identity, e.g. '01'
#     :param session: (int) Session Identity
#     :param category: (str) Experimental paradigm, e.g. "actionText"
#     :param args: (str) modalities of data
#     :return: list of tuples containing data and labels
#     """
#     data = []
#     for arg in args:
#         filename = f"classifierData/{category}_{arg}_CLF"
#         subj_object = load_subject(directory, subj, session, filename)["subject"]
#         data.append((subj_object.classifier_data.astype(np.float32), subj_object.labels.astype(np.int64) ))
#     return data

def get_ordered_lists(*args):
    flatten = lambda fl: [item for sublist in fl for item in sublist]  # flatten nested lists
    op_list = []
    for arg in zip(*args):
        arg_list = flatten(arg)
        op_list.append(arg_list)
    return op_list

def ordered_lists(*args):
    op_list = []
    for arg in args:
        op_list.append(get_ordered_lists(*arg))
    return get_ordered_lists(*op_list)


"""
Name: Ciaran Cooney
Date: 12/01/2019
Description: Functions required for data processing and training of 
CNNs on imagined speech EEG data.
"""

import pickle
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from functools import wraps

def eeg_to_3d(data, epoch_size, n_events,n_chan):
    """
    function to return a 3D EEG data format from a 2D input.
    Parameters:
      data: 2D np.array of EEG
      epoch_size: number of samples per trial, int
      n_events: number of trials, int
      n_chan: number of channels, int
        
    Output:
      np.array of shape n_events * n_chans * n_samples
    """
    idx, a, x = ([] for i in range(3))
    [idx.append(i) for i in range(0,data.shape[1],epoch_size)]
    for j in data:
        [a.append([j[idx[k]:idx[k]+epoch_size]]) for k in range(len(idx))]
   
    
    return np.reshape(np.array(a),(n_events,n_chan,epoch_size))

def load_subject(direct, subject, session, filename):
    f_name = f"{direct}/S{subject}/Session_{session}/{filename}.pickle"
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def load_pickle(direct, folder, filename):
	
    for file in os.listdir(direct + folder):
        if file.endswith(filename):
            pickle_file = (direct + folder + '/' + file)
            with open(pickle_file, 'rb') as f:
                file = pickle.load(f)

            return file, pickle_file

def create_events(data, labels):
	events = []
	x = np.zeros((data.shape[0], 3))
	for i in range(data.shape[0]):
		x[i][0] = i 
		x[i][2] = labels[i]
	[events.append(list(map(int, x[i]))) for i in range(data.shape[0])]
	return np.array(events)

def reverse_coeffs(coeffs, N):
	""" Reverse order of coefficients in an array."""
	idx = np.array([i for i in reversed(range(N))])
	coeffs = coeffs[idx]
	coeffs = coeffs.reshape((N,1))
	z = np.zeros((N,1))
	return np.append(coeffs, z, axis=1) , coeffs

def class_ratios(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_weight = dict()
    for i in range(len(unique)):
       class_weight[unique[i]] = len(labels) / (len(unique)*counts[i])
    return class_weight

def classification_report_csv(report, output_file):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(output_file + '.csv', index = False)

def load_features(direct, dict_key1, dict_key2=None):
    with open(direct, 'rb') as f:
        file = pickle.load(f)
    if dict_key2 == None:
        return np.array(file[dict_key1])
    else:
        return np.array(file[dict_key1]), np.array(file[dict_key2])

def short_vs_long(features, labels, split, event_id):
    """Function for multilabel data into binary-class sets i.e.,
       short words and long words
    """
    short, long, s_idx, l_idx, s_features, l_features = ([] for i in range(6))
    
    [short.append(event_id[i]) for i in event_id if len(i) <= split]
    [long.append(event_id[i]) for i in event_id if len(i) > split]
    
    [s_idx.append(i) for i, e in enumerate(labels) if e in short]
    [l_idx.append(i) for i, e in enumerate(labels) if e in long]
    
    [s_features.append(e) for i, e in enumerate(features) if i in s_idx]
    [l_features.append(e) for i, e in enumerate(features) if i in l_idx]
    
    s_labels = np.zeros(np.array(s_features).shape[0])
    l_labels = np.ones(np.array(l_features).shape[0])

    features = np.concatenate((s_features, l_features))
    labels = np.concatenate((s_labels,l_labels))
    
    return s_features, l_features, s_labels, l_labels, features, labels 

def return_indices(event_id, labels):
    indices = []
    for _, k in enumerate(event_id):
        idx = []
        for d, j in enumerate(labels):
            if event_id[k] == j:
                idx.append(d)
        indices.append(idx)
    return indices

def load_subject_eeg(subject_id, vowels):
    """ returns eeg data corresponding to words and vowels 
        given a subject identifier.
    """

    data_folder = 'C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    data_folder1 = 'C:\\Users\\cfcoo\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    words_file = 'raw_array_ica.pickle'
    vowels_file = 'raw_array_vowels_ica.pickle'
    
    try:
        with open(data_folder + words_file, 'rb') as f:
            file = pickle.load(f)
    except:
        print("Not on PC! Attempting to load from laptop.")
        with open(data_folder1 + words_file, 'rb') as f:
            file = pickle.load(f)
            
    w_data = file['raw_array'][:][0]
    w_labels = file['labels']
    if vowels == False:
        return w_data, w_labels

    elif vowels:
        try:
            with open(data_folder + vowels_file, 'rb') as f:
                file = pickle.load(f)
        except:
            with open(data_folder1 + vowels_file, 'rb') as f:
                file = pickle.load(f)
        v_data = file['raw_array'][:][0]
        v_labels = file['labels']
    return w_data, v_data, w_labels, v_labels

def balanced_subsample(features, targets, random_state=12):
    """
    function for balancing datasets by randomly-sampling data
    according to length of smallest class set.
    """
    from sklearn.utils import resample
    unique, counts = np.unique(targets, return_counts=True)
    unique_classes = dict(zip(unique, counts))
    mnm = len(targets)
    for i in unique_classes:
        if unique_classes[i] < mnm:
            mnm = unique_classes[i]

    X_list, y_list = [],[]
    for unique in np.unique(targets):
        idx = np.where(targets == unique)
        X = features[idx]
        y = targets[idx]
        
        #X1, y1 = resample(X,y,n_samples=mnm, random_state=random_state)
        X_list.append(X[:mnm])
        y_list.append(y[:mnm])
    
    balanced_X = X_list[0]
    balanced_y = y_list[0]
    
    for i in range(1, len(X_list)):
        balanced_X = np.concatenate((balanced_X, X_list[i]))
        balanced_y = np.concatenate((balanced_y, y_list[i]))

    return balanced_X, balanced_y

def predict(model, X_test, batch_size, iterator, threshold_for_binary_case=None):
    """
    Load torch model and make predictions on new data.
    """
    all_preds = []
    with th.no_grad():
        for b_X, _ in iterator.get_batches(SignalAndTarget(X_test, X_test), False):
            b_X_var = np_to_var(b_X)
            all_preds.append(var_to_np(model(b_X_var)))

        pred_labels = compute_pred_labels_from_trial_preds(
                    all_preds, threshold_for_binary_case)
    return pred_labels

def plot_confusion_matrix(cm, classes,filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    Code for confusion matrix extracted from here:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(1, figsize=(9, 6))
    #ax = plt.add_subplot(111)
    plt.tick_params(labelsize='large')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize='large', fontname='sans-serif')
    plt.xlabel('Predicted label', fontsize='large', fontname='sans-serif')
    fig.savefig(filename + '.jpg', bbox_inches='tight')
    return(fig)

def print_confusion_matrix(confusion_matrix, class_names, filename, normalize = True, figsize = (5,5), fontsize=16):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    fmt = '.2f' if normalize else 'd'
    #####set heatmap customization#####
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap='GnBu', linewidths=.5, cbar=False, annot_kws={"size": 16})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=16, fontname='sans-serif')
    plt.xlabel('Predicted label', fontsize=16, fontname='sans-serif')
    
    if filename != None:
        fig.savefig(filename + '.png', bbox_inches='tight') #store image as .png
    
    return fig

def data_wrangler(data_type, subject_id):
    """
    Function to return EEG data in format #trials*#channels*#samples.
    Also returns labels in the range 0 to n-1.
    """
    epoch = 4096
    if data_type == 'words':
        data, labels = load_subject_eeg(subject_id, vowels=False)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
    elif data_type == 'vowels':
        _, data, _, labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
    elif data_type == 'all_classes':
        w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(w_data)
        words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
        vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
        data = np.concatenate((words, vowels), axis=0)
        labels = np.concatenate((w_labels, v_labels), axis=0).astype(np.int64)
    
    x = lambda a: a * 1e6
    data = x(data)
    
    if data_type == 'words': # zero-index the labels
        labels[:] = [x - 6 for x in labels]
    elif (data_type == 'vowels' or data_type == 'all_classes'):
        labels[:] = [x - 1 for x in labels]

    return data, labels


def format_data(data_type, subject_id, epoch):
    """
    Returns data into format required for inputting to the CNNs.

    Parameters:
        data_type: str()
        subject_id: str()
        epoch: length of single trials, int
    """

    if data_type == 'words':
        data, labels = load_subject_eeg(subject_id, vowels=False)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 6 for x in labels] # zero-index the labels
    elif data_type == 'vowels':
        _, data, _, labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 1 for x in labels]
    elif data_type == 'all_classes':
        w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(w_data)
        words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
        vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
        data = np.concatenate((words, vowels), axis=0)
        labels = np.concatenate((w_labels, v_labels)).astype(np.int64)
        labels[:] = [x - 1 for x in labels]

    return data, labels

def current_loss(model_loss):
    """
    Returns the minimum validation loss from the 
    trained model
    """
    losses_list = []
    [losses_list.append(x) for x in model_loss]
    return np.min(np.array(losses_list))

def current_acc(model_acc):
    """
    Returns the maximum validation accuracy from the 
    trained model
    """
    accs_list = []
    [accs_list.append(x) for x in model_acc]
    return np.min(np.array(accs_list))

def balance_classes(data1,data2):

    if data1.shape[0] > data2.shape[0]:
        data1 = data1[:data2.shape[0],:,:]
    elif data1.shape[0] < data2.shape[0]:
        data2 = data2[:data1.shape[0],:,:]
        
    return data1, data2

def timer(orig_func):
    """
    decorator for logging time of function.
    """
    import time
    
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, *kwargs)
        t2 = time.time() - t1
        print(f"{orig_func.__name__} ran in: {round(t2,3)} seconds")
        return result
    
    return wrapper

def windows(trial_data, sub, window_size, overlap, fs):
    """
    Functon for obtaining classification windows for training.

    :param trial_data: EEG data - n_trials * n_chans * n_samples
    :param sub: subject object
    :param window_size: n number of samples
    :param overlap: n number of samples for overlap
    :param fs: sampling frequency
    :return: list containing data from each window
    """
    windows_list, index_list = [],[]
    n_windows = int(sub.epoch / window_size + np.floor((sub.epoch - overlap) / window_size))
    if n_windows == 0:
        n_windows = 1
    low_index = 0
    high_index = window_size
    for w in range(n_windows):
        data = trial_data[:, :, low_index:high_index]
        windows_list.append(data)
        index_list.append([low_index,high_index])
        low_index += overlap
        high_index += overlap

    return np.array(windows_list), index_list

def windows_index(epoch, window_size, overlap, fs):
    """
    Functon for obtaining classification windows for training.

    :param epoch: length of overal trial
    :param window_size: n number of samples
    :param overlap: n number of samples for overlap
    :param fs: sampling frequency
    :return: list containing data from each window
    """
    index_list = []
    n_windows = int(epoch / window_size + np.floor((epoch - overlap) / window_size))
    if n_windows == 0:
        n_windows = 1
    low_index = 0
    high_index = window_size
    for w in range(n_windows):
        index_list.append((low_index,high_index))
        low_index += overlap
        high_index += overlap

    return index_list

def get_class_labels(paradigm):
    """
    Function for obtaining class labels from paradigm description
    :param paradigm: string format: 'EEG_semantics_text'
    :return:
    """
    paradigm = paradigm.split('_')[1]
    if paradigm == 'semantics':
        class_labels = ['pig', 'dog', 'car', 'bus']
    elif paradigm == 'action':
        class_labels = ['kick', 'jump', 'chew', 'blink']
    elif paradigm == 'twoword':
        class_labels = ['red ball', 'blue hat', 'red blue', 'ball hat']
    elif paradigm == 'concrete':
        class_labels = ['apple', 'tiger', 'fruit', 'animal']
    return class_labels

def misclass_to_class(column):
    return 1 - column

def get_model_loss_and_acc(fold_models):
     """
     Function for extracting epoch-by-epoch model loss and accuracy scores from
     models associated with multiple cross-validation folds
     :param fold_models: list of Braindecode (PyTorch) sequential models
     :return: train_loss: (pandas.series) main training loss per epoch across folds
              valid_loss: (pandas.series) main tvalidation loss per epoch across folds
              test_loss: (pandas.series) main testing loss per epoch across folds
              train_acc: (pandas.series) main training acc per epoch across folds
              valid_acc: (pandas.series) main tvalidation acc per epoch across folds
              test_acc: (pandas.series) main testing acc per epoch across folds
     """
     train_loss = dict()
     valid_loss = dict()
     test_loss = dict()
     train_acc = dict()
     valid_acc = dict()
     test_acc = dict()

     for i, model in enumerate(fold_models):
        train_loss[i] = model.epochs_df['train_loss']
        valid_loss[i] = model.epochs_df['valid_loss']
        test_loss[i] = model.epochs_df['test_loss']
        train_acc[i] =  model.epochs_df['train_misclass']
        valid_acc[i] = model.epochs_df['valid_misclass']
        test_acc[i] = model.epochs_df['test_misclass']

     train_loss = pd.DataFrame(train_loss)
     valid_loss = pd.DataFrame(valid_loss)
     test_loss = pd.DataFrame(test_loss)
     train_loss = train_loss.mean(axis=1, skipna=True)
     valid_loss = valid_loss.mean(axis=1, skipna=True)
     test_loss = test_loss.mean(axis=1, skipna=True)

     train_acc = pd.DataFrame(train_acc).apply(lambda x : misclass_to_class(x)) # function converts misclass to classification accuracy
     valid_acc = pd.DataFrame(valid_acc).apply(lambda x : misclass_to_class(x))
     test_acc = pd.DataFrame(test_acc).apply(lambda x : misclass_to_class(x))
     train_acc = train_acc.mean(axis=1, skipna=True)
     valid_acc = valid_acc.mean(axis=1, skipna=True)
     test_acc = test_acc.mean(axis=1, skipna=True)

     return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc


def labels_dict_and_list(classes):
    """
    input: empty pandas DataFrame with column headings
           corresponding to class labels
    output: labels_dict (dict): key=number, value=string
            key_list (list): list of classes
    """

    labels_dict = dict()
    key_list = []
    for n, label in enumerate(classes.columns):
        labels_dict[str(n + 1)] = label

    for key in labels_dict:
        key_list.append(key)
    return labels_dict, key_list

def data_loader(directory, subj, session, category, *args):
    """

    :param directory: (str) directory of stored data
    :param subj: (str) Subject Identity, e.g. '01'
    :param session: (int) Session Identity
    :param category: (str) Experimental paradigm, e.g. "actionText"
    :param args: (str) modalities of data
    :return: list of tuples containing data and labels
    """
    data = []
    for arg in args:
        filename = f"classifierData/{category}_{arg}_CLF"
        subj_object = subjects.Subject.load_subject(f"filename.pickle") #load_subject(directory, subj, session, filename)
        #print(subj_object['data1'])
        data.append((subj_object.data1.astype(np.float32), subj_object.labels1.astype(np.int64)))
    return data

def load_subject(direct, subject, session, filename):
    f_name = f"{direct}/S{subject}/Session_{session}/{filename}.pickle"
    with open(f_name, 'rb') as f:
        return pickle.load(f)