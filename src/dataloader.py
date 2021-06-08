import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset

def get_dataset(data_type="Cifar10", data_root=None):
    if data_type=="Cifar10":
        # Cifar10 Standard : https://github.com/kuangliu/pytorch-cifar
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_root+'/input', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        test_dataset = torchvision.datasets.CIFAR10(root=data_root+'/input', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    
    return train_loader, test_loader
    
    