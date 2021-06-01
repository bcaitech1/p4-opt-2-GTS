import os
from types import new_class
import pandas as pd 
import numpy as np 
import nni 
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset.customdataset import CustomDataset
from models.utils import freeze, AverageMeter, EarlyStopping, Unfreeze, cutmix_data
from utils.utils import label_smooth_loss_fn, calc_macs, loss_fn
from sklearn.metrics import f1_score
from nni.compression.pytorch import ModelSpeedup
from models.mobilenetv2 import MobileNetV2

def vali_fn(val_data_loader, model, verbose=False):
    preds, gt = [], []
    label_list = [i for i in range(9)]
    model.eval()
    loss_fn = nn.CrossEntropyLoss().cuda()
    vali_loss = AverageMeter()
    with torch.no_grad():
        for images, labels in tqdm(val_data_loader):
            images, labels = images.cuda().float(), labels.cuda()
            current_batch_size = images.shape[0]

            model_pred = model(images)

            losses = loss_fn(model_pred, labels)
            
            _, pred = torch.max(model_pred, 1)
            
            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()

            vali_loss.update(losses.detach().item(), current_batch_size)
    vali_f1=f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)

    return vali_f1

def train_fn(model, num_epochs, cut_mix, train_data_loader, val_data_loader, optimizer, scheduler, save_path, device):
    freeze(model,-1)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    train_loss = AverageMeter()
    loss_fn = label_smooth_loss_fn
    early_stop = EarlyStopping(path=save_path)
    for epoch in range(num_epochs):
        model.train()
        if epoch==3:
          Unfreeze(model)
        for images, labels in tqdm(train_data_loader):
            # Optimizer.zero_grad
            for param in model.parameters():
                param.grad = None
            
            images, labels = images.to(device).float(), labels.to(device)
            current_batch_size = images.shape[0]
            if cut_mix:
              images, labels_a, labels_b, lam = cutmix_data(images, labels)
            
            model_pred = model(images)
            
            if cut_mix:
              loss_out = lam * loss_fn(model_pred, labels_a, device) + (1 - lam) * loss_fn(model_pred, labels_b, device)
            else:
              loss_out = loss_fn(model_pred, labels, device)
            train_loss.update(loss_out.detach().item(), current_batch_size)
            
            scaler.scale(loss_out).backward()
            scaler.step(optimizer)
            scaler.update()
            
        val_f1 = vali_fn(val_data_loader,model,device)
        print(f"\nEpoch #{epoch+1} train loss: [{train_loss.avg}]  validation f1 score : [{val_f1}]\n")
        
        scheduler.step(val_f1)
        early_stop(val_f1,model)
        if early_stop.early_stop:
            print('Stop Training.....')
            break

def get_dataloader(image_size, batch_size):
  data_root = f"/opt/ml/input/data/resize{image_size}"
  train_path = os.path.join(data_root, "train")
  val_path = os.path.join(data_root, "val")

  train_transform =  A.Compose([
                            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.4),     
                            A.GaussNoise(var_limit=(5.0, 30.0),p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p=0.3),
                            A.VerticalFlip(p=0.4),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2(p=1.0)
                        ])

  test_transform = A.Compose([
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2(p=1.0)
                        ])
                  
  train_dataset = CustomDataset(train_path, True, train_transform)
  val_dataset   = CustomDataset(val_path, True, test_transform)
    
  counts = np.bincount(train_dataset.img_label)
  labels_weights = 1. / counts
  weights = labels_weights[train_dataset.img_label]
  weight_sampler = WeightedRandomSampler(weights, len(weights))

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=weight_sampler, shuffle=False, num_workers=0)
  val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 

  return train_dataloader, val_dataloader

def get_model(mode, lr, model_path, mask_path, image_size):
  model = MobileNetV2(n_class=9, input_size=image_size)

  if model_path is not None:
    model.load_state_dict(torch.load(model_path))

  if mode == 2:
    sample_data = torch.randn(2, 3, image_size, image_size)
    ms = ModelSpeedup(model, sample_data, mask_path, torch.device('cpu'))
    ms.speedup_model()

  optimizer = optim.AdamW(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

  return model, optimizer, scheduler      

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--mode", type=int, default=0, help="0. train model\n1. search pruning\n2. Do pruning and fine tunning")
  parser.add_argument("--epoch", type=int, default=100, help="If mode 1, train_episode, else train epoch")
  parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
  parser.add_argument("--flops_ratio", type=float, default=0.5, help="choose which flops ratio to remind")
  parser.add_argument("--image_size", type=int, default=64)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--save_model_path", type=str)
  parser.add_argument("--mask_path", type=str, default=None)
  parser.add_argument("--model_path", type=str, default=None)

  args = parser.parse_args()
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  train_loader, val_loader = get_dataloader(args.image_size, args.batch_size)
  
  if args.mode == 0:
    print("[Training Mode]......")
    model, optimizer, scheduler = get_model(args.mode, args.lr, args.model_path, args.mask_path, args.image_size)
    model = model.to(device)
    train_fn(model, args.epoch, False, train_loader, val_loader, optimizer, scheduler, args.save_model_path, device)
  
  elif args.mode == 1:
    print("[Pruning Search Mode]......")
    from nni.algorithms.compression.pytorch.pruning import AMCPruner
    model, _, _ = get_model(args.mode, args.lr, args.model_path, args.mask_path, args.image_size)
    model = model.to(device)
    
    config_list = [{
        'op_types': ['Conv2d', 'Linear']
      }]
    
    lbound = args.flops_ratio - 0.2 if args.flops_ratio > 0.3 else 0.01
    pruner = AMCPruner(
        model, config_list, vali_fn, val_loader, model_type="mobilenetv2", dataset='cifar10',
        train_episode=args.epoch, flops_ratio=args.flops_ratio, lbound=lbound,
        rbound=1., suffix=None, seed=41)
    pruner.compress()
  elif args.mode == 2:
    print("[Pruning And Fine Tunning Mode]......")
    model, optimizer, scheduler = get_model(args.mode, args.lr, args.model_path, args.mask_path, args.image_size)
    model = model.to(device)
    train_fn(model, args.epoch, False, train_loader, val_loader, optimizer, scheduler, args.save_model_path, device)