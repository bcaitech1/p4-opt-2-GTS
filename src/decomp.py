import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from src.vbmf import *
import warnings
warnings.filterwarnings(action='ignore')
tl.set_backend('pytorch')

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = EVBMF(unfold_0)
    _, diag_1, _, _ = EVBMF(unfold_1)
    if diag_0 is None or diag_1 is None:
        return [0, 0]
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
    if layer.bias is None:
        target_bias = False
    else:
        target_bias = True

    ranks = estimate_ranks(layer)
    if ranks[0] == 0 or ranks[1] == 0:
        print(layer, "Skip")
        return layer
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=target_bias)

    if target_bias:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)