import torch as th

def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)

def tensor_size(x):
    print(x.size())
    return x

def reshape_tensor(x):
    x
    return x.view(x.size(0),x.size(1)*x.size(2)*1)

def reshape_output(x):
    return x.view(x.size(0),4, 1 ,1)

def reshape_4_lstm(x):
    return x.view(x.size(0),1,x.size(1))

def dense_input(x):
    return x.size(2)

def tensor_print(x):
    print(x.data.cpu().numpy())
    return x

def linear_input_shape(x):
    print(x.size(1)*x.size(2))
    return x.size(1)*x.size(2)

def mean_inplace(tensor_1, tensor_2):
    """
    function for meaning softmax outputs from two networks,
    Cuurently not able to use as inplace changes to the tensor
    cause problems with backpropagation
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    for i in range(len(tensor_1)):
        for j in range(len(tensor_1[i])):
            tensor_1[i][j] = (tensor_1[i][j] + tensor_2[i][j]) / 2
    return tensor_1
def new_mean(tensor_1, tensor_2):
    avg = []
    for sm1, sm2 in zip(tensor_1, tensor_2):
        avg.append([(a+b) / 2 for a,b in zip(sm1, sm2)])
    avg = th.tensor(avg, dtype=th.float32).cuda()
    return avg

# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    #print(x.shape)
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
