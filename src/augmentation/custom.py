import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "CUSTOM": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}

# Add your Augmentation list
def get_train_transform(data_type="CUSTOM", image_size=224):
    return A.Compose([     
        A.Cutout(num_holes=8, max_h_size=4, max_w_size=4, fill_value=0, always_apply=False, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0),p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.VerticalFlip(p=0.4),
        A.Resize(image_size,image_size),
        A.Normalize(mean=NORMALIZE_INFO[data_type]["MEAN"], std=NORMALIZE_INFO[data_type]["STD"], max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2(p=1.0)
    ])

def get_valid_transform(data_type="CUSTOM", image_size=224):
    return A.Compose([
        A.Resize(image_size,image_size),
        A.Normalize(mean=NORMALIZE_INFO[data_type]["MEAN"], std=NORMALIZE_INFO[data_type]["STD"], max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2(p=1.0)
    ])