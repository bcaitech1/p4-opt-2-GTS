from src.model import Model
from src.loss import *
from src.utils.torch_utils import *
from src.utils.train_utils import *
from src.utils.common import *
from src.dataloader import *
from src.network_prune import *
from src.trainer import *
from typing import Any, Dict, List, Tuple, Union
import argparse
import copy
import math
from datetime import datetime
# Torch
import torch
import torch.optim as optim
#YAML
import yaml
import ruamel.yaml as yaml
from ruamel.yaml.comments import CommentedSeq, CommentedMap
#optuna
import optuna
from optuna.structs import TrialPruned
import warnings
warnings.filterwarnings(action='ignore') 
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_CONFIG = read_yaml(cfg="configs/model/example.yaml")
PRUNED_BACKBONE_SET = set()
BEFORE_PRUNED = False
BEST_MODEL_SCORE = 0

DICT_POOLING = {
        224:4, 
        128:4,
        64:4,
        32:3,
    }

MAX_NUM_POOLING = 3
MAX_DEPTH = 6

def add_module(trial, depth, n_pooling):
    m_name = 'm'+str(depth)
    normal_module_list = ["InvertedResidualv2","InvertedResidualv3","MBConv","Fire","Bottleneck","Pass"]
    eca_module_list = ["ECAInvertedResidualv2","ECAInvertedResidualv3", "ECABottleneck","GhostBottleneck", "Pass"]
    
    m_args = []
    m_kernel = 3
    #depth = depth-1
    if n_pooling<MAX_NUM_POOLING:
        m_stride = trial.suggest_int(m_name+"/stride", low=1, high=2, step=1)
    else:
        m_stride = 1
        
    if m_stride==1:
        m_repeat = trial.suggest_int(m_name+"/repeat", low=1, high=8, step=1)
    else:
        m_repeat = 1
    
    # ECA Module : 1 / Normal Module : 2
    module_pick = trial.suggest_int(m_name+"/module_pick", low=1, high=2)
    if module_pick == 1:
        # ECA Module
        module_idx = trial.suggest_int(m_name+"/module_name", low=1, high=5)
        m = eca_module_list[module_idx-1]
        if m == "ECAInvertedResidualv2":
            m_c = trial.suggest_int(m_name+"/c_ev2", low=16*depth, high=32*depth, step=16)
            m_t = trial.suggest_int(m_name+"/t_ev2", low=1, high=4)
            m_args = [m_c, m_t, m_stride]
        elif m == "ECAInvertedResidualv3":
            m_t = round(trial.suggest_float(m_name+"/t_ev3", low=1.0, high=6.0, step=0.1), 1)
            m_c = trial.suggest_int(m_name+"/c_ev3", low=16*depth, high=32*depth, step=16)
            m_se = trial.suggest_int(m_name+"/se_ev3", low=0, high=1, step=1)
            m_hs = trial.suggest_int(m_name+"/hs_ev3", low=0, high=1, step=1)
            m_args = [m_kernel, m_t, m_c, m_se, m_hs, m_stride]
        elif m == "ECABottleneck":
            m_c = trial.suggest_int(m_name+"/c_eb", low=16*depth, high=32*depth, step=16)
            m_args = [m_c]
        elif m == "GhostBottleneck":
            m_t = round(trial.suggest_float(m_name+"/t_gb", low=1.0, high=6.0, step=0.1), 1)
            m_c = trial.suggest_int(m_name+"/c_gb", low=16*depth, high=32*depth, step=16)
            m_se = trial.suggest_int(m_name+"/se_gb", low=0, high=1, step=1)
            m_args = [m_kernel, m_t, m_c, m_se, m_stride]
    elif module_pick == 2:
        # Normal Module
        module_idx = trial.suggest_int(m_name+"/module_name", low=1, high=6)
        m = normal_module_list[module_idx-1]
        if m == "InvertedResidualv2":
            m_c = trial.suggest_int(m_name+"/c_v2", low=16*depth, high=32*depth, step=16)
            m_t = trial.suggest_int(m_name+"/t_v2", low=1, high=4)
            m_args = [m_c, m_t, m_stride]
        elif m == "InvertedResidualv3":
            m_t = round(trial.suggest_float(m_name+"/t_v3", low=1.0, high=6.0, step=0.1), 1)
            m_c = trial.suggest_int(m_name+"/c_v3", low=16*depth, high=32*depth, step=16)
            m_se = trial.suggest_int(m_name+"/se_v3", low=0, high=1, step=1)
            m_hs = trial.suggest_int(m_name+"/hs_v3", low=0, high=1, step=1)
            m_args = [m_kernel, m_t, m_c, m_se, m_hs, m_stride]
        elif m == "MBConv":
            m_c = trial.suggest_int(m_name+"/c_mb", low=16*depth, high=32*depth, step=16)
            m_t = trial.suggest_int(m_name+"/t_mb", low=1, high=4)
            m_args = [m_t, m_c, m_stride, m_kernel]
        elif m == "Fire":
            m_s = trial.suggest_int(m_name+"/s_f", low=16*depth, high=32*depth, step=16)
            m_e = trial.suggest_int(m_name+"/e_f", low=m_s*2, high=m_s*4, step=m_s)
            m_args = [m_s, m_e, m_e]
        elif m == 'Bottleneck':
            m_c = trial.suggest_int(m_name+"/c_b", low=16*depth, high=32*depth, step=16)
            m_args = [m_c]
            
    if not m == "Pass":
        if m_stride==1:
            return [m_repeat, m, m_args], True
        else:
            return [m_repeat, m, m_args], False
    else:
        return None, None
    
def add_pooling(trial,depth):
    m_name = 'mp'+str(depth)
    m = trial.suggest_categorical(
        m_name,
        ["MaxPool",
         "Pass"])
    if not m == "Pass":
        return [1, m, [3,2,1]]
    else:
        return None
    
def search_model(trial: optuna.trial.Trial, CLASSES) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_pooling = 0 # Modify 1 -> 0
    # Example) ImageSize with downsampling
    # 32 -> 16 -> 8 -> 4 -> 2 (need 3 times downsampling(=stride2)) <- Competition size
    # 128 -> 64 -> 32 -> 16 -> 8 -> 4 (need 5 times downsampling(=stride2)) <- General size
    # 224 -> 112 -> 56 -> 28 -> 14 -> 7

    # Start Conv (or Depthwise)
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = 1
    m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=24, step=8)
    
    if MAX_NUM_POOLING==3:
        m1_stride = 1
    else:
        m1_stride = 2
        
    m1_activation = trial.suggest_categorical(
        "m1/activation", ["ReLU", "Hardswish"]
        )
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    model.append([m1_repeat, m1, m1_args])
    
    # First Pooling Layer(or Pass)
    if MAX_NUM_POOLING>3:
        pool_args = add_pooling(trial, 1)
        if pool_args is not None:
            model.append(pool_args)
            n_pooling+=1
        
    # Module Layers (depths = max_depth)
    for depth in range(2,MAX_DEPTH+3):
        module_args, use_stride = add_module(trial, depth, n_pooling)
        if module_args is not None:
            model.append(module_args)
            if use_stride:
                if n_pooling<MAX_NUM_POOLING:
                    pool_args = add_pooling(trial, depth)
                    if pool_args is not None:
                        model.append(pool_args)
                        n_pooling+=1
            else:
                n_pooling+=1
    
    # Last Conv (or Pass)
    # Modify
    #m_last = trial.suggest_categorical("m_last",["Conv","Pass"])
    m_last = "Conv"
    if not m_last == "Pass":
        last_dim = trial.suggest_int("last_dim", low=512, high=1024, step=256)
        model.append([1, "Conv", [last_dim, 1, 1]])
        
    # GAP -> Classifier
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "Flatten", []])
    model.append([1, "Linear", [CLASSES]])
    
    if n_pooling==MAX_NUM_POOLING:
        return model
    else:
        return None

def objective(trial: optuna.trial.Trial, device, args, train_loader, test_loader):
    learning_rate = args.lr
    image_size = args.image_size
    num_epochs = args.epoch
    LIMIT_MACS = args.LIMIT_MACS
    data_type = args.data_type
    
    model_config = copy.deepcopy(MODEL_CONFIG)
    model_config["input_size"] = [image_size, image_size]
    model_config["backbone"] = search_model(trial,args.CLASSES)
    
    if model_config["backbone"] is None:
        # Stride 조건을 못채운 모델의 경우 Pruned
        raise TrialPruned()
    
    model_instance = Model(model_config, verbose=False)
        
    macs = calc_macs(model_instance.model, (3, image_size, image_size))
    
    global BEST_MODEL_SCORE, BEFORE_PRUNED, PRUNED_BACKBONE_SET

    if macs<=LIMIT_MACS:
        if BEFORE_PRUNED:
            current_backbone_set = get_architecture_set(model_config["backbone"])
            if similar_architecture_pruned(PRUNED_BACKBONE_SET, current_backbone_set) == True:
                # Pruned
                print("Similar pruned.")
                raise TrialPruned()
                
        print(f"[Trial : {trial.number}] Found a lightweight Model, Model macs : {macs} <= {LIMIT_MACS}")
        print("------ Model Architecture ------")
        print(model_config["backbone"])
        print("--------------------------------")
        print("Start Train & Testing....")
        
        # optimizer setting
        # Standard : https://github.com/kuangliu/pytorch-cifar
        if data_type == "CIFAR10":
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model_instance.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif data_type == "CUSTOM":
            ############################
            # TODO Hyperparameter Search
            ############################
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model_instance.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        # TODO : ImageNet, CIFAR100
        
        # Search Model Architecture -> Model Train and Testing -> Best Model save (*.yaml, *.pth)
        test_score = train_fn(model_instance.model, args.METRIC, args.CLASSES, num_epochs, train_loader, test_loader, loss_fn, optimizer, scheduler, device)
        
        if test_score is None:
            # Architecture drop 
            PRUNED_BACKBONE_SET = get_architecture_set(model_config["backbone"])
            print("Pruned.")
            BEFORE_PRUNED = True
            raise TrialPruned()
        else:
            BEFORE_PRUNED = False
            # Architecture save to wandb
            if BEST_MODEL_SCORE < test_score :
                BEST_MODEL_SCORE = test_score
                model_config["backbone"] = cl_backbone = CommentedSeq(model_config["backbone"])
                model_config["input_size"] = cl_input_size = CommentedSeq(model_config["input_size"])
                cl_backbone.fa.set_flow_style()
                cl_input_size.fa.set_flow_style()
                with open(f'./configs/model/model_{BEST_MODEL_SCORE * 10000 // 1}.yaml', 'w') as f:
                    yaml.dump(model_config, f, Dumper=yaml.RoundTripDumper)
    else:
        raise TrialPruned()
    return test_score, macs

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--METRIC", type=str, default="F1", help="TODO") # ACC , F1
    parser.add_argument("--LIMIT_MACS", type=int, default=100000000, help="TODO")
    parser.add_argument("--lr", type=float, default=1e-2, help="TODO")
    parser.add_argument("--epoch", type=int, default=50, help="TODO")
    parser.add_argument("--image_size", type=int, default=32, help="TODO")
    parser.add_argument("--batch_size", type=int, default=128, help="TODO")
    parser.add_argument("--CLASSES", type=int, default=10, help="TODO")
    parser.add_argument("--MAX_DEPTH", type=int, default=5, help="TODO")
    parser.add_argument("--data_type", type=str, default="CIFAR10", help="TODO") # CIFAR10, CIFAR100, IMAGENET, CUSTOM
    parser.add_argument("--data_root", type=str, default="../", help="TODO")
    parser.add_argument("--study_name", type=str, default="automl_search", help="TODO")
    parser.add_argument("--seed", type=int, default=17, help="TODO")
    parser.add_argument("--trial", type=int, default=10000, help="TODO")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    global MAX_NUM_POOLING, MAX_DEPTH
    MAX_NUM_POOLING = DICT_POOLING[args.image_size]
    if args.image_size == 32 and args.data_type != "CIFAR10":
        MAX_NUM_POOLING=4
    MAX_DEPTH = args.MAX_DEPTH
    
    # ADD
    if MAX_NUM_POOLING > MAX_DEPTH:
        print("MAX_DEPTH is too low.")
        return
    
    if args.data_type=="CUSTOM":
        train_loader, test_loader = get_dataset(data_type=args.data_type, data_root='../', image_size=args.image_size, batch_size=args.batch_size)
    else:
        train_loader, test_loader = get_dataset(data_type=args.data_type, data_root='../', image_size=args.image_size)
    
    print(f"Start Architecture Search.... (Limit MACs : {args.LIMIT_MACS})")
    # Search Algorithm
    # TODO Sampler
    sampler = optuna.samplers.MOTPESampler(n_startup_trials=11, seed=args.seed)
    rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=rdb_storage,
        study_name=args.study_name,
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, device, args, train_loader, test_loader), n_trials=args.trial)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"{key}:{value}")

if __name__ == '__main__':
    main()