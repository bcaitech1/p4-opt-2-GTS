import copy
from torchvision import transforms
from src.model import Model
from src.decomp import *
from src.loss import *
from src.utils.torch_utils import *
from src.utils.train_utils import *
from src.utils.common import *
from src.dataloader import *
from src.trainer import *
from typing import Any, Dict, List, Tuple, Union
import argparse
import math
from datetime import datetime
# Torch
import torch
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(file_yaml = 'lastnet.yaml', CLASSES = 10, CHECKPOINT_PATH = None):
    model_yaml_path = './configs/model/' + file_yaml
    model_config = read_yaml(cfg=model_yaml_path)
    model_instance = Model(model_config, verbose=False)
    net = model_instance.model
    if net[-1].linear.out_features != CLASSES:
        in_features = net[-1].linear.in_features
        net[-1].linear = nn.Linear(in_features, CLASSES)
    if CHECKPOINT_PATH is not None:
        print("Weight Load..")
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint)
    else:
        print("Can't load your pretrained weight")
    net.eval()
    return net

# Find conv layer -> Decomp
def find_conv(model):
    for i, key in enumerate(model._modules):
        if isinstance(model._modules[key], torch.nn.modules.conv.Conv2d):
            if model._modules[key].groups==1:
                model._modules[key] = tucker_decomposition_conv_layer(model._modules[key])
        else:
            find_conv(model._modules[key])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="lastnet.yaml", help="example.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Your model checkpoint path")
    parser.add_argument("--METRIC", type=str, default="F1", help="Select Metric [ACC, F1]")
    parser.add_argument("--num_epochs", type=int, default=50, help="Fine Tuning Epochs")
    parser.add_argument("--image_size", type=int, default=32, help="Select image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Select batch size")
    parser.add_argument("--CLASSES", type=int, default=10, help="Number of classes")
    parser.add_argument("--data_type", type=str, default="CIFAR10", help="Select data type [CIFAR10, CIFAR100, CUSTOM]") # CIFAR10, CIFAR100, IMAGENET, CUSTOM
    parser.add_argument("--data_root", type=str, default="../", help="Set data directory path")
    parser.add_argument("--seed", type=int, default=41, help="Select Random Seed")
    args = parser.parse_args()
    
    
    if args.data_type == "CIFAR10":
        args.CLASSES = 10
        args.image_size = 32
        args.num_epochs = 200
        model = get_model(file_yaml = args.yaml, CLASSES = args.CLASSES, CHECKPOINT_PATH = args.checkpoint_path)
        train_loader, test_loader = get_dataset(data_type=args.data_type, data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size)
    elif args.data_type == "CIFAR100":
        args.CLASSES = 100
        args.image_size = 32
        args.num_epochs = 200
        model = get_model(file_yaml = args.yaml, CLASSES = args.CLASSES, CHECKPOINT_PATH = args.checkpoint_path)
        train_loader, test_loader = get_dataset(data_type=args.data_type, data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size)
    elif args.data_type == "CUSTOM":
        model = get_model(file_yaml = args.yaml, CLASSES = args.CLASSES, CHECKPOINT_PATH = args.checkpoint_path)
        train_loader, test_loader = get_dataset(data_type=args.data_type, data_root=args.data_root, image_size=args.image_size)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scheduler_name = "cosine"
    
    # Test (before Decomposition)
    print("Before Decomposition Model Test...")
    before_acc, before_f1, _ = test_fn(test_loader, model.to(device), args.CLASSES, device)
    print("Done.")
    
    before_macs = calc_macs(model, (3, args.image_size, args.image_size))
    model = find_conv(model.cpu())
    after_macs = calc_macs(model, (3, args.image_size, args.image_size))
    
    # Test (After Decomposition)
    print("After Decomposition Model Test...")
    after_acc, after_f1, _ = test_fn(test_loader, model.to(device), args.CLASSES, device)
    print("Done.")
    
    print("Step 1 : Model Decomposition.")
    print("-"*80)
    print(f'Before MACs {before_macs:.2f} -> After MACs {after_macs:.2f}')
    print(f'Compression ratio : {after_macs/before_macs*100:.2f}%')
    print(f'Model Peformance ACC : {before_acc:.4f} -> {after_acc:.4f}')
    print(f'Model Peformance F1 : {before_f1:.4f} -> {after_f1:.4f}')
    print("-"*80)
    
    print("Step 2 : Model Fine tuning.")
    test_score = train_fn(model, args.METRIC, args.CLASSES, None, args.num_epochs, train_loader, test_loader, loss_fn, optimizer, scheduler, scheduler_name, None, device)
    print(f"Fine tuning Score : {test_score:.4f}")
    
    # TODO
    # pytorch -> TFlite
    
if __name__ == '__main__':
    main()

