import torch
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_f1, model):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        model.eval()
        torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)

def freeze(model,idx):
  for m in list(model.children()):
    for param in m.parameters():
      param.requires_grad=True

  for m in list(model.children())[:idx]:
    for param in m.parameters():
      param.requires_grad=False

def Unfreeze(model):
  for m in list(model.children()):
    for param in m.parameters():
      param.requires_grad=True

def rand_bbox(W, H, lam, device):
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = (W * cut_rat).type(torch.long)
    cut_h = (H * cut_rat).type(torch.long)
    # uniform
    cx = torch.randint(W, (1,)).to(device)
    cy = torch.randint(H, (1,)).to(device)
    x1 = torch.clamp(cx - cut_w // 2, 0, W)
    y1 = torch.clamp(cy - cut_h // 2, 0, H)
    x2 = torch.clamp(cx + cut_w // 2, 0, W)
    y2 = torch.clamp(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, device, alpha=1.0, p=0.5):
    if np.random.random() > p:
        return x, y, torch.zeros_like(y), 1.0
    W, H = x.size(2), x.size(3)
    shuffle = torch.randperm(x.size(0)).to(device)
    cutmix_x = x

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(device)

    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    cutmix_x[:, :, x1:x2, y1:y2] = x[shuffle, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / float(W * H)).item()
    y_a, y_b = y, y[shuffle]
    return cutmix_x, y_a, y_b, lam