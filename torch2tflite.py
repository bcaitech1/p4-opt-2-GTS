import torch
import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
from onnx_tf.backend import prepare
import tensorflow as tf

import numpy as np
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import os
from src.model import Model
from src.utils.torch_utils import *
from src.utils.train_utils import *
from src.utils.common import *
import argparse
import warnings
import logging

def get_nas_model(cfg):
    model_config = read_yaml(cfg=cfg)
    model_instance = Model(model_config, verbose=False)
    net = model_instance.model

    net.eval()
    return net

class Torch2tflite:
    def __init__(self, model_cfg, model_path, save_path, model_name, image_size = 224):
        self.pytorch_model = get_nas_model(model_cfg)
        self.pytorch_model.state_dict(torch.load(model_path))
        self.save_path = save_path
        self.image_size = image_size
        self.model_name = model_name


    def convert(self):
        self.__torch2onnx()
        self.__onnx2tf()
        self.__tf2tflite()

    def __torch2onnx(self):
        print("[Info] torch2onnx...", end="")
        self.__onnx_path = os.path.join(self.save_path, self.model_name + ".onnx")
        
        example_input = torch.randn(10, 3, self.image_size, self.image_size)
        torch.onnx.export(
            model=self.pytorch_model,
            args=example_input, 
            f=self.onnx_path, # where should it be saved
            verbose=False,
            export_params=True,
            do_constant_folding=False,  # fold constant values for optimization
            input_names=['input'],
            output_names=['output']
        )

        self.onnx_model = onnx.load(self.__onnx_path)
        print("Done...")

    def __onnx2tf(self):
        print("[INFO] onnx2tf.....", end="")
        self.__tf_path = os.path.join(self.save_path, self.model_name + ".pb")

        self.tf_model = prepare(self.onnx_model)
        self.tf_model.export_graph(self.__tf_path)
        print("Done...")

    def __tf2tflite(self):
        print("[INFO] tf2tflite.....", end="")

        self.__tflite_path = os.path.join(self.save_path, self.model_name + ".tflite")

        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_path)

        tflite_model = converter.convert()
        open(self.__tflite_path, "wb").write(tflite_model)
        
        print("Done...")

    @property
    def onnx_path(self):
        return self.__onnx_path


    @property
    def tf_path(self):
        return self.__tf_path
    

    @property
    def tflite_path(self):
        return self.__tflite_path


if __name__=="__main__":
    warnings.filterwarnings(action='ignore')
    tf.get_logger().setLevel(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type=str, default="./configs/model/lastnet.yaml", help="set model config file path")
    parser.add_argument("--model_path", type=str, default="./saved/lastnet_taco.pth", help="set saved model weight file path")

    parser.add_argument("--save_path", type=str, default="./converted_model", help="set save path for converted model") 
    
    parser.add_argument("--image_size", type=int, default=224, help="set image size")
    parser.add_argument("--model_name", type=str, default="convert_model", help="set converted model name ex) [model_name].pd")

    args = parser.parse_args()

    tflite_converter = Torch2tflite(model_cfg= args.model_cfg,
                                    model_path= args.model_path,
                                    save_path= args.save_path,
                                    model_name= args.model_name,
                                    image_size= args.image_size)
    
    tflite_converter.convert()

    print("Process..Done")
    print("onnx model saved path: \t\t", tflite_converter.onnx_path)
    print("tensor flow model saved path: \t\t", tflite_converter.tf_path)
    print("tf lite model saved path: \t\t", tflite_converter.tflite_path)
