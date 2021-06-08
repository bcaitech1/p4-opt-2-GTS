"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv3 import (
    InvertedResidualv3,
    InvertedResidualv3Generator,
)
from src.modules.eca_invertedresidualv3 import (
    ECAInvertedResidualv3,
    ECAInvertedResidualv3Generator,
)
from src.modules.invertedresidualv2 import (
    InvertedResidualv2,
    InvertedResidualv2Generator,
)
from src.modules.eca_invertedresidualv2 import (
    ECAInvertedResidualv2,
    ECAInvertedResidualv2Generator,
)
from src.modules.eca_bottleneck import (
    ECABottleneck,
    ECABottleneckGenerator,
)
from src.modules.mbconv import (
    MBConv,
    MBConvGenerator,
)
from src.modules.fire import (
    Fire,
    FireGenerator,
)
from src.modules.ghost_bottleneck import (
    GhostBottleneck,
    GhostBottleneckGenerator,
)
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (
    AvgPoolGenerator,
    GlobalAvgPool,
    GlobalAvgPoolGenerator,
    MaxPoolGenerator,
)

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "ECAInvertedResidualv2",
    "InvertedResidualv3",
    "ECAInvertedResidualv3",
    "ECABottleneck",
    "Fire",
    "MBConv",
    "GhostBottleneck",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "ECAInvertedResidualv3Generator",
    "ECAInvertedResidualv2Generator",
    "ECABottleneckGenerator",
    "InvertedResidualv3Generator",
    "FireGenerator",
    "MBConvGenerator",
    "GhostBottleneckGenerator",
]
