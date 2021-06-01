# p4-opt-2-GTS

## AMC Pruning

used [nni](https://github.com/microsoft/nni)

### Required

* torch >= 1.6.0
* Do not install nni 
    * If you want to prune mobilenetV2, current version of nni can not prune the model.
    * I think it will be updated next release but now, It can not be (2021.06.01)
    * I referenced speedup_v2 branch at nni github

### update note   

* [2021.06.01]   
    * AMC pruning : mobilenetV2   

### How to use

you can use train.sh and this is expected file struct

```
$> tree -d
.
├── data
|     ├── resize{image_size}
|     |      ├── category_1
|     |      |    ├── files....            
|     |      └── category_2...... 
|     ├── resize64
|     |      ├── category_1
|     |      |    ├── files....            
|     |      └── category_2...... 
│     └── ...
└── git_repo
      └── ...
```

#### training 

```   
python train.py --mode 0 --save_model_path <save_model_path> 
```

#### AMC pruning

```   
python train.py --mode 1 --save_model_path <save_model_path> --flops_ratio <flops_ratio>
```

#### Pruning and fine tunning

```   
python train.py --mode 2 --save_model_path <save_model_path> --mask_path <best_mask.pth> --model_path <best_model.pth>
```
