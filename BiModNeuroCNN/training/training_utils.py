import numpy as np

def combine_batches(batch1,batch2):
    """
    Function for combining batches of inputs and targets for
    2 modalities of data.
    :param batch1: (list-type) [0] inputs, [1] targets
    :param batch2: (list-type) [0] inputs, [1] targets
    :return: (list) [inputs1, targets1, inputs2, targets2]
    """
    new_batch = []
    for inputs, target in zip(batch1, batch2):
        a = list(inputs)
        b = list(target)
        a.append(b[0])
        a.append(b[1])
        new_batch.append(a)
    return new_batch

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