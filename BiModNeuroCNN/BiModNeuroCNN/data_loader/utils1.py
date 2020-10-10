import numpy as np
import pickle
from BiModNeuroCNN.subjects import subjects

def subject_data_loader(filename):
    """

    :param filename: (str) directory of stored data
    :return: tuples containing data and labels
    """  
    subj_object = subjects.Subject.load_subject(f"{filename}.pickle") 
    return (subj_object.data1.astype(np.float32), subj_object.labels1.astype(np.int64))
