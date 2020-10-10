import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from BiModNeuroCNN.models.network_utils import reshape_tensor, reshape_4_lstm, _transpose_time_to_spat, tensor_size

class SubNet(BaseModel):
    """
    Temporal-Spatial first layer based on [2]

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            in_chans,
            n_classes,
            input_time_length=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            n_filters_2=10,
            filter_length_2=10,
            pool_time_length_1=5,
            pool_time_stride_1=2,
            pool_length_2=5,
            pool_stride_2=2,
            final_conv_length=30,
            conv_nonlin=square,
            pool_mode="max",
            pool_nonlin=safe_log,
            later_nonlin=None,
            later_pool_nonlin=nn.functional.leaky_relu,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.1,
            stride_before_pool=False,
            structure = "shallow",
            fc1_out_features = 500,
            fc2_out_features = 500,
    ):
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
        else:
            conv_stride = 1

        pool_class_dict = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)
        first_pool_class = pool_class_dict[self.pool_mode]
        pooling_reduction = self.pool_time_length_1 * self.pool_time_stride_1 * 10

        model = nn.Sequential()

        if self.split_first_layer:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module("conv_time", nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1),
                                                    stride=1, ), )
            model.add_module("conv_spat", nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                                                    (1, self.in_chans), stride=1, bias=not self.batch_norm, ),)
            n_filters_conv = self.n_filters_spat
            n_filters_op = self.n_filters_spat * (self.input_time_length - (4 + pooling_reduction)) # semi-hardcoded at the moment
        else:
            model.add_module("conv_time", nn.Conv2d(self.in_chans, self.n_filters_time,
                                                    (self.filter_time_length, 1), stride=1,
                                                    bias=not self.batch_norm, ), )
            n_filters_conv = self.n_filters_time
            
            n_filters_op = self.n_filters_time * (self.input_time_length - (4 + pooling_reduction)) # semi-hardcoded at the moment

        if self.batch_norm:
            model.add_module("bnorm", nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha,
                                                     affine=True), )
        model.add_module("conv_nonlin", Expression(self.conv_nonlin))
        model.add_module("drop", nn.Dropout(p=self.drop_prob))

        model.add_module("pool", first_pool_class(kernel_size=(self.pool_time_length_1, 1), stride=(self.pool_time_stride_1, 1)),)
        model.add_module("pool_nonlin", Expression(self.pool_nonlin))


        def add_conv_pool_block(model, n_filters_before,
                                n_filters, filter_length, block_nr):

            model.add_module(f"conv_{block_nr}", nn.Conv2d(n_filters_before, n_filters,
                                                           (filter_length, 1), stride=(conv_stride, 1),
                                                           bias=not self.batch_norm))
            
            if self.batch_norm:
                model.add_module(f"bnorm_{block_nr}", nn.BatchNorm2d(n_filters,
                                                                     momentum=self.batch_norm_alpha,
                                                                     affine=True, eps=1e-5))
            model.add_module(f"nonlin_{block_nr}", Expression(self.conv_nonlin))
            model.add_module(f"drop_{block_nr}", nn.Dropout(p=self.drop_prob))

            model.add_module("pool", first_pool_class(kernel_size=(self.pool_length_2, 1),
                            stride=(self.pool_stride_2, 1)),)
            model.add_module("pool_nonlin", Expression(self.pool_nonlin))

        if self.structure == "deep":

            add_conv_pool_block(model, n_filters_conv, self.n_filters_2,
                                self.filter_length_2, 2)
            model.add_module("tensor shape", Expression(tensor_size))
            pooling_reduction = pooling_reduction + 22
            print(pooling_reduction)
            n_filters_op = self.n_filters_2 * 45 #(self.input_time_length - (23 + pooling_reduction)) # semi-hardcoded at the moment

        model.add_module('reshape', Expression(reshape_tensor))
        
        model.add_module('fc_1', nn.Linear(n_filters_op, self.fc1_out_features, bias=True))
        

        # Initialization is xavier for initial layers
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(model.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(model.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)

        param_dict = dict(list(model.named_parameters()))
        if self.structure == "deep":
            conv_weight = param_dict['conv_2.weight']
            init.kaiming_normal_(conv_weight)  # He initialization
            if not self.batch_norm:
                conv_bias = param_dict['conv_2.bias']
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict['bnorm_2.weight']
                bnorm_bias = param_dict['bnorm_2.bias']
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        fc_weight = param_dict['fc_1.weight']
        init.kaiming_uniform_(fc_weight)
        # model.eval()

        return model


class BiModalNet(nn.Module):

    def __init__(self, n_classes, in_chans_1, input_time_1, SubNet_1_params, in_chans_2, input_time_2,
                 SubNet_2_params, linear_dims, drop_prob, nonlin, fc1_out_features, fc2_out_features,
                 gru_hidden_size, gru_n_layers=1):
        """
        BiModal CNN network receiving 2 different data types corresponding to a single ground truth (e.g. EEG and fNIRS)
        Two SubNets are initialised and the forward pass of both is performed before their outputs are fed into the 
        remainder of the network to be fused and applied to GRU and linear layers before log softmax classification.

        Parameters
        :param: n_classes (int) number of classes in classification task
        :param: in_chans_1 (int) number of channels in data
        :param: input_time_1 (int) number of time samples in data
        :param: SubNet_1_params (dict) parameters for initiating subnet 1
        :param: in_chans_2 (int) number of channels in data
        :param: input_time_2 (int) number of time samples in data
        :param: SubNet_2_params (dict) parameters for initiating subnet 2
        :param: linear_dims (int) dimension of linear layer
        :param: drop_prob (float) dropout probability
        :param: nonlin (th.nn.functional) activation function
        :param: fc1_out_features (int) output dimension of subnet 1 linear layer
        :param: fc2_out_features (int) output dimension of subnet 2 linear layer
        :param: gru_hidden_size (int) size of GRU hidden layer
        :param: gru_n_layers (int) number of GRU hidden layers
        """
        self.n_classes = n_classes
        self.in_chans_1 = in_chans_1
        self.input_time_1 = input_time_1
        for key in SubNet_1_params:
            setattr(self, f"SN1_{key}", SubNet_1_params[key])
        self.in_chans_2 = in_chans_2
        self.input_time_2 = input_time_2
        for key in SubNet_2_params:
            setattr(self, f"SN2_{key}", SubNet_2_params[key])

        self.linear_dims = linear_dims
        self.drop_prob = drop_prob
        self.fc1_out_features = fc1_out_features
        self.fc2_out_features = fc2_out_features
        self.fused_dimension = fc1_out_features + fc2_out_features
        self.gru_hidden_size = gru_hidden_size
        self.gru_n_layers = gru_n_layers

        super(BiModalNet, self).__init__()
        model = nn.Sequential()

        self.subnet_1 = SubNet(in_chans=self.in_chans_1, n_classes=self.n_classes, input_time_length=self.input_time_1,
                                n_filters_time=self.SN1_n_filters_time, filter_time_length=self.SN1_filter_time_length,
                                n_filters_spat=self.SN1_n_filters_spat, n_filters_2=self.SN1_n_filters_2, filter_length_2=self.SN1_filter_length_2,
                                pool_time_length_1=self.SN1_pool_time_length_1, pool_time_stride_1=self.SN1_pool_time_stride_1, pool_length_2=self.SN1_pool_length_2,
                                pool_stride_2=self.SN1_pool_stride_2, final_conv_length='auto',
                                conv_nonlin=self.SN1_conv_nonlin, pool_mode=self.SN1_pool_mode, pool_nonlin=self.SN1_pool_nonlin,
                                split_first_layer=self.SN1_split_first_layer, batch_norm=self.SN1_batch_norm, batch_norm_alpha=self.SN1_batch_norm_alpha,
                                drop_prob=self.SN1_drop_prob, structure=self.SN1_structure, fc1_out_features=self.fc1_out_features).create_network() 

 
        self.subnet_2 = SubNet(in_chans=self.in_chans_2, n_classes=self.n_classes, input_time_length=self.input_time_2,
                                n_filters_time=self.SN2_n_filters_time, filter_time_length=self.SN2_filter_time_length,
                                n_filters_spat=self.SN2_n_filters_spat, n_filters_2=self.SN2_n_filters_2, filter_length_2=self.SN2_filter_length_2,
                                pool_time_length_1=self.SN2_pool_time_length_1, pool_time_stride_1=self.SN2_pool_time_stride_1, pool_length_2=self.SN2_pool_length_2,
                                pool_stride_2=self.SN2_pool_stride_2, final_conv_length='auto',
                                conv_nonlin=self.SN2_conv_nonlin, pool_mode=self.SN2_pool_mode, pool_nonlin=self.SN2_pool_nonlin,
                                split_first_layer=self.SN2_split_first_layer, batch_norm=self.SN2_batch_norm, batch_norm_alpha=self.SN2_batch_norm_alpha,
                                drop_prob=self.SN2_drop_prob, structure=self.SN2_structure, fc2_out_features=self.fc2_out_features).create_network()

        self.reshape_tensor = reshape_4_lstm # works for GRU also
        
        self.gru = nn.GRU(input_size=self.fused_dimension, hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_n_layers, batch_first=True)
 
        self.nonlin  = nonlin
        self.fused_dp = nn.Dropout(p=self.drop_prob)

        self.fused_linear = nn.Linear(self.gru_hidden_size, self.n_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

        self.size = Expression(tensor_size) # useful for debugging tensor/kernel dimension mismatches
        

    def forward(self, data_1, data_2):
        """
        Forward pass of the Bimodal CNN
        
        :param data_1: tensor
        :param data_2: tensor
        """
        data_1_h = self.subnet_1(data_1)
        data_2_h = self.subnet_2(data_2)

        fusion_tensor = th.cat((data_1_h, data_2_h), dim=1)
        
        fusion_tensor_gru = self.reshape_tensor(fusion_tensor)
        gru_inp = fusion_tensor.view(fusion_tensor_gru.size(0), 1, self.fused_dimension)

        gru_op, _ = self.gru(gru_inp)

        gru_op = self.nonlin(gru_op)
        gru_op_dp = self.fused_dp(gru_op)

        fused_linear = self.fused_linear(gru_op_dp.view(gru_op_dp.size(0), gru_op_dp.size(2)))
        fused_linear = self.nonlin(fused_linear)

        softmax = self.softmax(fused_linear)

        return softmax



