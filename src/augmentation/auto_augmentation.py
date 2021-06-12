from torchvision import transforms
from src.augmentation.transforms import *
import optuna

def get_augmentation(trial:optuna.Trial):
    augmentations = []

    aug_func_list = transforms_info()
    aug_use_flag = {name: trial.suggest_categorical(name=name + "_use", choices=[True, False]) for name in aug_func_list.keys()}

    
    for func_name, use_flag in aug_use_flag.items():
        if(use_flag):
            low = min(aug_func_list[func_name][1], aug_func_list[func_name][2])
            high = max(aug_func_list[func_name][1], aug_func_list[func_name][2])
            augmentations.append(aug_func_list[func_name][0](trial.suggest_float(name=func_name, low=low, high=high, step=10)))

    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)))

    return transforms.Compose(augmentations)