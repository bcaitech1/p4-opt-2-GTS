# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from functools import partial
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def translate_list(list_node, speedup=None):
    """
    Get the list of values from the list construct node.
    Parameters
    ---------
    list_node: Torch.C.Value
        The cpp node of the target list.
    speedup: ModuleSpeed
        The Module speedup module.
    Returns
    -------
    values: list
        The list of values in the target cpp list node.
    """
    # the node that create the list
    create_node = list_node.node()
    assert create_node.kind() == 'prim::ListConstruct'
    inputs = list(create_node.inputs())
    values = []
    for _i in inputs:
        debugName = _i.debugName()
        if speedup is not None and debugName in speedup.internal_result:
            # this value is the result of the other nodes, such as
            # ate::size
            values.append(speedup.internal_result[debugName].item())
        else:
            # if the corresponding value is a constant
            values.append(_i.toIValue())
    return values


def dropout_python(node, speedup):
    return torch.dropout


def flatten_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    start_dim = inputs[1].toIValue()
    end_dim = inputs[2].toIValue()
    new_flatten = partial(torch.flatten, start_dim=start_dim, end_dim=end_dim)
    return new_flatten


def relu_inplace_python(node, speedup):
    return torch.relu_


def relu_python(node, speedup):
    return torch.relu


def mean_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    keep_dim = inputs[2].toIValue()
    print(dim_list)
    print(keep_dim)
    new_mean = partial(torch.mean, dim=tuple(dim_list), keepdim=keep_dim)
    return new_mean


def add_python(node, speedup):
    return torch.add


def slice_python(node, speedup):
    class SliceMoudle(torch.nn.Module):
        def __init__(self, sliceobj):
            super(SliceMoudle, self).__init__()
            self.sliceobj = sliceobj

        def forward(self, x):
            return x[self.sliceobj]

    c_node = node.key_node
    inputs = list(c_node.inputs())
    print(inputs)
    slice_dim = inputs[1].toIValue()
    slice_start = inputs[2].toIValue()
    slice_end = inputs[3].toIValue()
    slice_step = inputs[4].toIValue()
    slice_obj = slice(slice_start, slice_end, slice_step)
    slice_list = []
    for _ in range(slice_dim-1):
        slice_list.append(slice(None, None))
    slice_list.append(slice_obj)
    return SliceMoudle(tuple(slice_list))


def size_python(node, speedup):
    # return None
    class SizeMoudle(torch.nn.Module):
        def __init__(self, sizedim):
            super(SizeMoudle, self).__init__()
            self.sizedim = sizedim

        def forward(self, x):
            return torch.as_tensor([x.size(self.sizedim)], dtype=torch.long)
            # return torch.tensor(x.size(self.sizedim))
    c_node = node.key_node
    inputs = list(c_node.inputs())
    size_dim = inputs[1].toIValue()
    return SizeMoudle(size_dim)


def transpose_python(node, speedup):
    return torch.t


def transpose2_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_1 = inputs[1].toIValue()
    dim_2 = inputs[2].toIValue()
    new_transpose = partial(torch.transpose, dim0=dim_1, dim1=dim_2)
    return new_transpose


def toint_python(node, speedup):
    class ToIntModule(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.int)
    return ToIntModule()


def view_python(node, speedup):
    class ViewModule(torch.nn.Module):
        def __init__(self, shape):
            super(ViewModule, self).__init__()
            self.shape = shape
            logger.info('View Module output size: %s', str(self.shape))

        def forward(self, *args):
            return args[0].view(self.shape)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    shape = translate_list(inputs[1], speedup)
    return ViewModule(shape)


def reshape_python(node, speedup):
    class ReshapeModule(torch.nn.Module):
        def __init__(self, shape):
            super(ReshapeModule, self).__init__()
            self.shape = shape
            logger.info('Reshape Module output size: %s', str(self.shape))

        def forward(self, *args):
            return args[0].view(self.shape)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    shape = translate_list(inputs[1], speedup)
    return ReshapeModule(shape)


def permute_python(node, speedup):
    class PermuteModule(torch.nn.Module):
        def __init__(self, dimlist):
            super(PermuteModule, self).__init__()
            self.dimlist = dimlist

        def forward(self, x):
            return x.permute(self.dimlist)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    return PermuteModule(dim_list)


def matmul_python(node, speedup):
    return torch.matmul


def div_python(node, speedup):
    # The second input parameter of torch.div can be a
    # tensor or a constant, if it is a constant, we need
    # to return
    c_node = node.key_node
    inputs = list(c_node.inputs())
    if inputs[1].debugName() in speedup.internal_result:
        # the second input parameters is the output of the other
        # nodes
        return torch.div
    else:
        other = inputs[1].toIValue()
        new_div = partial(torch.div, other=other)
        # print(other)
        return new_div


def softmax_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = inputs[1].toIValue()
    new_softmax = partial(torch.softmax, dim=dim)
    return new_softmax


def contiguous_python(node, speedup):
    class contiguousModule(torch.nn.Module):
        def forward(self, x):
            return x.contiguous()
    return contiguousModule()


def gelu_python(node, speedup):
    return torch.nn.GELU()


def cat_python(node, speedup):
    class CatModule(torch.nn.Module):
        def __init__(self, cat_dim):
            super(CatModule, self).__init__()
            self.cat_dim = cat_dim

        def forward(self, *args):
            return torch.cat(args, dim=self.cat_dim)

    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = inputs[1].toIValue()
    return CatModule(dim)


def avgpool2d_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    kernel_size = translate_list(inputs[1], speedup)
    stride = translate_list(inputs[2], speedup)
    padding = translate_list(inputs[3], speedup)
    new_avgpool = partial(torch.nn.functional.avg_pool2d,
                          kernel_size=kernel_size, stride=stride, padding=padding)
    return new_avgpool


def adaptive_avgpool_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    output_size = translate_list(inputs[1], speedup)
    new_avgpool = torch.nn.AdaptiveAvgPool2d(output_size)
    return new_avgpool


def tupleunpack_python(node, speedup):
    # Note: tuple unpack should only exists at the
    # the end of the model, and is no need to replace/propagate mask
    return None


trans_from_jit_to_python = {
    'aten::add': add_python,
    'aten::add_': add_python,
    'aten::relu': relu_python,
    # tanh behaives like relu
    'aten::tanh': relu_python,
    'aten::tanh_': relu_python,
    'aten::flatten': flatten_python,
    'aten::mean': mean_python,
    'aten::dropout': dropout_python,
    'aten::relu_': relu_inplace_python,
    'aten::slice': slice_python,
    'aten::size': size_python,
    'aten::t': transpose_python,
    'aten::transpose': transpose2_python,
    'aten::Int': toint_python,
    'aten::view': view_python,
    'aten::reshape': reshape_python,
    'aten::permute': permute_python,
    'aten::matmul': matmul_python,
    'aten::div': div_python,
    'aten::softmax': softmax_python,
    'aten::contiguous': contiguous_python,
    'aten::gelu': gelu_python,
    'aten::cat': cat_python,
    'aten::avg_pool2d': avgpool2d_python,
    'aten::max_pool2d': avgpool2d_python,
    'aten::adaptive_avg_pool2d': adaptive_avgpool_python,
    'prim::TupleUnpack': tupleunpack_python
}


def jit_to_python_function(node, speedup):
    """
    Return a callable object to inference the mask according to the
    node.op_type.
    Parameters
    ---------
    node: NodeGroup
        The target node to inference the mask
    speedup: ModelSpeedup
        The speedup object of the target model.
    Returns
    ------
    func: callable object(nn.Module/function)
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in trans_from_jit_to_python:
        logger.warning(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~')
        # return None to skip the mask inference for this node
        return None
    return trans_from_jit_to_python[node.op_type](node, speedup)