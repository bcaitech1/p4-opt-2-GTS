import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from src.utils.train_utils import AverageMeter
from src.utils.torch_utils import init_params

# TODO
# 1. Pruned
# 2. add Metric F1 Score

def train_fn(model, METRIC, CLASSES, trial, num_epochs, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, scheduler_name, pruner, device):
    loss_fn = loss_fn.to(device)
    init_params(model)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    train_loss = AverageMeter()
    # best_score
    best_score = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_data_loader:
            # Optimizer.zero_grad
            for param in model.parameters():
                param.grad = None
            
            images, labels = images.to(device).float(), labels.to(device)
            current_batch_size = images.shape[0]
            
            model_pred = model(images)
            
            loss_out = loss_fn(model_pred, labels)
            train_loss.update(loss_out.detach().item(), current_batch_size)
            
            scaler.scale(loss_out).backward()
            scaler.step(optimizer)
            scaler.update()
        
        test_accuracy, test_f1, test_loss = test_fn(val_data_loader, model, CLASSES, device)
        if METRIC=="ACC":
            test_score = test_accuracy
        elif METRIC=="F1":
            test_score = test_f1
        print(f"\nEpoch #{epoch+1} train loss : [{train_loss.avg}] test loss : [{test_loss}] test score : [{test_score}]\n")
      
        if pruner is not None:
            pruner.add_train_info(trial, epoch, test_score)
            if pruner.train_prune():
                return None
            
        # Scheduler
        if scheduler is not None:
            if scheduler_name == "cosine":
                scheduler.step()
            elif scheduler_name == "reduce":
                scheduler.step(test_score)
                
        if best_score < test_score:
            best_score = test_score

    return best_score

def test_fn(val_data_loader, model, CLASSES, device):
    label_list = [i for i in range(CLASSES)]
    preds, gt = [], []
    model.eval()
    test_loss = AverageMeter()
    loss_fn = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for images, labels in val_data_loader:
            images, labels = images.to(device).float(), labels.to(device)
            current_batch_size = images.shape[0]

            model_pred = model(images)

            losses = loss_fn(model_pred, labels)
            
            _, pred = torch.max(model_pred, 1)
            
            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            
            test_loss.update(losses.detach().item(), current_batch_size)
            
    test_accuracy = accuracy_score(gt, preds)
    test_f1 = f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)
    # validation loss
    return test_accuracy, test_f1, test_loss.avg