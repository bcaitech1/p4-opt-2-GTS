import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.init as init
from torch import nn as nn

from ptflops import get_model_complexity_info
    
def calc_macs(model, input_shape):
    macs, params = get_model_complexity_info(
        model=model,
        input_res=input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        ignore_modules=[nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6],
    )
    return macs

def model_info(model, verbose=False):
    """Print out model info."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )
    print(
        f"Model Summary: {len(list(model.modules()))} layers, "
        f"{n_p:,d} parameters, {n_g:,d} gradients"
    )
    
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)
                
@torch.no_grad()
def check_runtime(
    model: nn.Module, img_size: List[int], device: torch.device, repeat: int = 100
) -> float:
    repeat = min(repeat, 20)
    img_tensor = torch.rand([1, *img_size]).to(device)
    measure = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.eval()
    for _ in range(repeat):
        start.record()
        _ = model(img_tensor)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        measure.append(start.elapsed_time(end))

    measure.sort()
    n = len(measure)
    k = int(round(n * (0.2) / 2))
    trimmed_measure = measure[k + 1 : n - k]

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        _ = model(img_tensor)
    print(prof)
    print("measured time(ms)", np.mean(trimmed_measure))
    model.train()
    return np.mean(trimmed_measure)


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def autopad(
    kernel_size: Union[int, List[int]], padding: Union[int, None] = None
) -> Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    # Pad to 'same'
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    return padding or [x // 2 for x in kernel_size]


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.type = act_type
        self.args = [1] if self.type == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.type is None:
            return nn.Identity()
        elif hasattr(nn, self.type):
            return getattr(nn, self.type)(*self.args)
        else:
            return getattr(
                __import__("src.modules.activations", fromlist=[""]), self.type
            )()