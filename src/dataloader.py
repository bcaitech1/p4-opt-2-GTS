import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from src.augmentation.custom import get_train_transform, get_valid_transform
from src.custom_dataset import CustomDataset

# Cifar10 & Cifar100 Standard : https://github.com/kuangliu/pytorch-cifar
CIFAR_TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
CIFAR_TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
        
def get_dataset(data_type="CIFAR10", data_root=None, image_size=224, batch_size=128):
    if data_type=="CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root=data_root+'/input', train=True, download=True, transform=CIFAR_TRANSFORM_TRAIN)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        test_dataset = torchvision.datasets.CIFAR10(root=data_root+'/input', train=False, download=True, transform=CIFAR_TRANSFORM_TEST)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    elif data_type=="CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root=data_root+'/input', train=True, download=True, transform=CIFAR_TRANSFORM_TRAIN)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        test_dataset = torchvision.datasets.CIFAR100(root=data_root+'/input', train=False, download=True, transform=CIFAR_TRANSFORM_TEST)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    elif data_type=="CUSTOM":
        # 배포 시 경로 재설정
        train_dataset = CustomDataset(data_dir=data_root+'/transforms_data/resize128/train', transforms=get_train_transform(data_type, image_size))
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        
        test_dataset = CustomDataset(data_dir=data_root+'/transforms_data/resize128/val', transforms=get_valid_transform(data_type, image_size))
        test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_loader, test_loader
    
    