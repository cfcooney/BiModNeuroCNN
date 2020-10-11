import numpy as np

def eeg_to_3d(data, labels, epoch, n_events, n_chans):
    """
    function to return a 3D EEG data format from a 2D input.
    Parameters:
      data: 2D np.array of EEG
      labels: (np.array ||list) 
      epoch: number of samples per trial, int
      n_events: number of trials, int
      n_chan: number of channels, int
        
    Output:
      np.array of shape n_events * n_chans * n_samples
    """
    idx, a, x = ([] for i in range(3))
    [idx.append(i) for i in range(0,data.shape[1],epoch)]
    for j in data:
        [a.append([j[idx[k]:idx[k]+epoch]]) for k in range(len(idx))]
        
    return np.reshape(np.array(a),(labels.shape[0],n_chans,epoch))