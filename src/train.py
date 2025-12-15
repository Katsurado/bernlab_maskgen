import argparse
import numpy as np
import torch
import maskgen.model as m
import torch.optim as optim
import torch.amp as amp
import json
import torch.optim.swa_utils as swa
import wandb
import gc

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from maskgen.data import ImageData
from maskgen.utils import *
from tqdm.auto import tqdm
from torchinfo import summary


def train_epoch(model:torch.nn.Module, 
                ema_model:optim.swa_utils.AveragedModel, 
                criterion:torch.nn.Module, 
                dataloader:DataLoader, 
                optimizer:optim.Optimizer, 
                scheduler:lr_scheduler.LRScheduler, 
                scaler:amp.grad_scaler.GradScaler, 
                device:str, 
                config:dict) -> float:
    '''
    train one epoch
    '''
    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
    train_loss = -1

    for i, (images, masks) in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, memory_format=torch.channels_last)

        with torch.autocast(device):
            output = model(images)
            loss = criterion(output, masks)

        scaler.scale(loss).backward()

        if config['clip_grad'] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad'])

        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)

        batch_bar.set_postfix(
            loss = f'{loss.item()}',
            lr = f"{float(optimizer.param_groups[0]['lr'])}"
        )

        batch_bar.update
        train_loss = loss.item()

    if scheduler is not None:
        scheduler.step()

    batch_bar.close()

    return train_loss

@torch.no_grad()
def validate(model:optim.swa_utils.AveragedModel, 
             criterion:torch.nn.Module, 
             dataloader:DataLoader, 
             device:str)->float:
    '''
    compute val loss
    '''
    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val Cls.', ncols=5)
    val_loss = -1

    for i, (images, masks) in enumerate(dataloader):
        # Move images to device
        images, masks = images.to(device), masks.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, masks)

        batch_bar.set_postfix(
            loss = f'{loss.item()}'
        )

        val_loss = loss.item()
        batch_bar.update()
    
    batch_bar.close()

    return val_loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_pth", type=str)
    args = parser.parse_args()

    config_path = args.config_pth
    with open(config_path, 'r') as f:
        config = json.load(f)

    # global settings
    device = set_device()
    root = config['root']
    NUM_CHANNELS = config['channels']
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.set_float32_matmul_precision('high')

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # set up dataset and data loader
    train_dataset = ImageData(root, 'train', True, config)
    val_dataset = ImageData(root, 'val', False, config)
    test_dataset = ImageData(root, 'test', False, config)

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    
    # define models
    model = m.Net(NUM_CHANNELS, config)
    model = model.to(device=device, memory_format=torch.channels_last)

    model_summary = str(summary(model, (config['batch_size'], 3, config['img_size'], config['img_size'])))
    arch_file   = open('/'.join([checkpoint_dir, 'model_arch.txt']), "w")
    file_write  = arch_file.write(model_summary)
    arch_file.close()

    model = torch.compile(model)

    # define loss, optimizer, ema model, scaler, scheduler and shit
    params = param_groups(model, config)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(params, lr=config['lr'])
    
    warmup = lr_scheduler.LinearLR(optimizer, total_iters=config["warmup_epoch"])
    cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler_params']['T-max'])

    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config["warmup_epoch"]]
    )

    ema_model = swa.AveragedModel(model, multi_avg_fn=swa.get_ema_multi_avg_fn(config['ema']))

    scaler = amp.grad_scaler.GradScaler(device)

    # track wandb results
    wandb.login(key=config["wandb_key"]) # API Key is in your wandb account, under settings (wandb.ai/settings)

    run = wandb.init(
        entity = config['wandb_params']['entity'],
        name = config['wandb_params']['name'], ## Wandb creates random run names if you skip this field
        reinit = config['wandb_params']['reinit'], ### Allows reinitalizing runs when you re-run this cell
        id = config['wandb_params']['id'], ### Insert specific run id here if you want to resume a previous run
        resume = config['wandb_params']['resume'], ### You need this to resume previous runs, but comment out reinit = True when using this
        project = config['wandb_params']['project'], ### Project should be created in your wandb account
        config = config ### Wandb Config for your run
    )

    e = 0
    best_val_loss = 1e9
    metrics = {}

    for epoch in range(e,  config['epochs']):
        print("\nEpoch {}/{}".format(epoch+1, config['epochs']))
        train_loss = train_epoch(model, ema_model, criterion, train_loader, 
                                 optimizer, scheduler, scaler, device, config)
        curr_lr = float(optimizer.param_groups[0]['lr'])
        print("\nEpoch {}/{}: Train Loss {:.04f}\t Learning Rate {:.04f}"
              .format(epoch + 1, config['epochs'], train_loss, curr_lr))
        valid_loss = validate(ema_model, criterion, val_loader, device)

        metrics.update({
            'lr': curr_lr,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'epoch': epoch,
        })

        save_model(model, ema_model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
        wandb.save(os.path.join(config['checkpoint_dir'], 'last.pth'))
        print("Saved epoch model")

        if valid_loss < best_val_loss:
            save_model(model, ema_model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best.pth'))
            print("Saved best val model")

        if run is not None:
            run.log(metrics)

