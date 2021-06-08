import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from src.utils.train_utils import AverageMeter
from src.utils.torch_utils import init_params

# TODO
# 1. Pruned
# 2. add Metric F1 Score

def train_fn(model, num_epochs, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, device):
    loss_fn = loss_fn.to(device)
    init_params(model)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    train_loss = AverageMeter()
    # best_score
    best_accuracy = 0
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
        
        test_accuracy = test_fn(val_data_loader,model,device)
        print(f"\nEpoch #{epoch+1} train loss : [{train_loss.avg}] test accuracy : [{test_accuracy}]\n")
        # Pruned Function (epoch)
        # >>> TODO <<<
        if epoch==0 and test_accuracy<0.45:
            return None
        #########################
            
        # Scheduler
        if scheduler is not None:
            scheduler.step()
        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy

    return best_accuracy

def test_fn(val_data_loader, model, device):
    preds, gt = [], []
    model.eval()
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
    test_accuracy = accuracy_score(gt, preds)
    # validation loss
    return test_accuracy