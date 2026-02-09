import os

import torch

import maskgen.model as m


def set_device()->str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def list_dir(dirpath:str)->list[str]:
    """
    Return all non-hidden entries in dirpath.
    Skips files/folders starting with '.' (e.g., .DS_Store) to
    ensure compatibility with Mac
    """
    return [name for name in os.listdir(dirpath) if not name.startswith('.')]


def param_groups(model:torch.nn.Module, config:dict) -> list[dict]:
    '''
    separate model's parameters to groups to selectively perform weight decay
    '''
    decay, no_decay = [], []
    norm_types = (torch.nn.GroupNorm, torch.nn.LayerNorm, m.GlobalResponseNorm)

    for name, module in model.named_modules():
        for pn, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pn == "bias":
                no_decay.append(p)
            elif isinstance(module, norm_types):
                no_decay.append(p)
            else:
                decay.append(p)

    return [
        {"params": decay, "weight_decay": config['weight_decay']},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def save_model(model, ema_model, optimizer, scheduler, metrics, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         'metric'                   : metrics,
         'epoch'                    : epoch,
         'ema_model'                : ema_model.state_dict() },
         path)


def load_model(model, ema_model, device, optimizer=None, scheduler=None, path='./checkpoint.pth'):
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        scheduler = None
    epoch = checkpoint['epoch']
    metrics = checkpoint['metric']
    return model, ema_model, optimizer, scheduler, epoch, metrics
