# :tada: p4-opt-2-GTS :tada:
## About 

만약 새로운 경량화 구조의 모델을 찾아내서 서비스까지 해보고 싶으시다면, 우리 `플랫폼`을 사용해보세요.

## File Struct    

```
$> tree -d
.
├── configs
│   └── model
├── src
│   ├── augmentation
│   ├── modules
│   ├── utils
│   ├── custom_dataset.py
│   ├── dataloader.py
│   ├── decomp.py
│   ├── loss.py
│   ├── model.py
│   ├── network_prune.py
│   ├── trainer.py
│   └── vbmf.py
├── model_decomp.py
├── model_search.py
├── requirement.txt
└── torch2tflite.py   

```

## Requirements 

* Pytorch
* TorchVision
* optuna
* ....

## How to use?   

### 1. model_search.py   

NAS를 통해 경량화 모델을 찾을 수 있습니다. 

#### 경량화 모델을 찾기 위해 사용되는 모듈들   

![module_table](https://user-images.githubusercontent.com/68745983/122677014-61b2f480-d21b-11eb-84a5-bffa0b8c5641.PNG)

#### 필요한 Args  

| Args 요소 | 설명 |
|:--------:|:--------|
| METRIC     | 모델의 성능을 평가하는 지표입니다. [F1, ACC]|        |  
| LIMIT_MACS|찾고자하는 모델의 MACS에 제한을 걸어 줍니다.|   
|image_size|타깃이 되는 데이터의 이미지 사이즈를 정의합니다.|   
|batch_size|dataloader의 배치사이즈를 지정해줍니다.|   
|CLASSES|분류하고자하는 데이터의 클래스 수를 지정합니다.|   
|MAX_DEPTH|찾고자하는 모델의 최대 깊이를 지정해주세요.|   
|data_type|데이터 유형을 지정해줍니다. [CIFAR10, IMAGENET, CUSTOM]|    
|data_root|데이터가 저장된 디렉터리 경로를 지정합니다.|   
|Trial|목표 아키텍쳐를 찾기 위해 몇번의 Trial을 할 것인지 지정합니다.|  
|prun_type|아키텍쳐를 찾을 때 어떤식으로 모델을 선별하여 가지고 올것인지 정해줍니다. [0.사용안함, 1.optuna에 내장된 pruner를 사용, 2. custom]|   
|auto_augment|오토 어그멘테이션을 사용할지에 대한 여부를 알려줍니다.|    

### 2. model_decomp.py   

NAS를 통해 찾은 모델을 좀 더 가볍게 해주기 위해 decomposition단계를 거칩니다.   

### 3. torch2tflite.py 

Pytorch모델을 다양한 플랫폼에서 서비스 하기위해 여러 프랫폼으로 변환해주는 작업을 해줍니다.   
![convert_model](https://user-images.githubusercontent.com/68745983/122677005-595ab980-d21b-11eb-820b-a76eb50f9af3.PNG)

## Project with this platform  

해당 프로젝트를 통해 GTSNet이라는 모델을 찾아 학습시킨후 여러 프로젝트에 적용을 해보았습니다.   

* Android Project: https://github.com/gihop/MaskCheckIn-boostcamp-p4-GTS-side-project   
* Raspberry Pi Project: https://github.com/happyBeagle/rpi_image


## About Other Branches
### 1. Pruning 
#### * [AMC Pruning](https://github.com/bcaitech1/p4-opt-2-GTS/tree/AMCpruning_MinJung)

