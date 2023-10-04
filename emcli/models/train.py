"""
Train, log, and save models with neural networks
"""
import click
import logging
import yaml
import random
import argparse
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
from torch import optim

from emcli.models.utils import lookup_torch_dtype
from emcli.models.utils import init_sweep_config

cfg = {
    'epochs': 5,
    'batch_size': 1,
    'learning_rate': 1e-5,
    'val_percent': 0.1,
    'weight_decay': 1e-8,
    'momentum': 0.999,
    'gradient_clipping': 1.0,
    'num_workers': None,
    'amp': False, # mixed precision
    'dtype': 'float32',
    'save_checkpoint': True
}
criterion = nn.MSELoss()

def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        no_wandb: bool = False,
        parallel: bool = False,
        sweep: bool = False,
        cfg: dict = None,
    ):
    dtype = lookup_torch_dtype(cfg['dtype'])
    # (Initialize logging)
    if not no_wandb:
        import wandb
        wandb_run = wandb.init(project=cfg['wandb_project_name'], 
            resume='allow', anonymous='must',
            dir=cfg['path_wandb'])
        wandb_run.config.update(
            dict(epochs=cfg['epochs'], 
                batch_size=cfg['batch_size'], 
                img_size=cfg['img_size'][0],
                learning_rate=cfg['learning_rate'],
                weight_decay = cfg['weight_decay'],
                loss_function = cfg['loss_function'],
                activation = cfg['activation'],
                num_extra_convs = cfg['num_extra_convs'],
                train_len = train_loader.dataset.__len__(),
                val_len = val_loader.dataset.__len__(),
                val_percent=cfg['val_percent'], 
                save_checkpoint=cfg['save_checkpoint'],
                amp=cfg['amp'])
        )
    else:
        wandb_run = None

    logging.info(f'''Starting training:
        Epochs:          {cfg['epochs']}
        Batch size:      {cfg['batch_size']}
        Image size:      {cfg['img_size']}        
        Learning rate:   {cfg['learning_rate']}
        Weight decay:    {cfg['weight_decay']}
        Loss function:   {cfg['loss_function']}
        Activation:      {cfg['activation']}
        Extra convs:     {cfg['num_extra_convs']}
        Training size:   {train_loader.dataset.__len__()}
        Validation size: {val_loader.dataset.__len__()}
        Checkpoints:     {cfg['save_checkpoint']}
        Device:          {device.type}
        Mixed Precision: {cfg['amp']}
        Num. Workers:    {cfg['num_workers']}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], 
                              momentum=cfg['momentum'], foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=cfg['lr_patience'])
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

    # criterion = nn.CrossEntropyLoss() if model.out_channels > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # Begin training
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=train_loader.dataset.__len__(), 
                  desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img',
                  #disable=(sweep==False) # disable tqdm if printing to log instead of console
                  ) as pbar:
            for i, batch in enumerate(train_loader):
                inputs, targets, targets_mask = batch

                assert inputs.shape[1] == model.in_channels, \
                    f'Network has been defined with {model.in_channels} input channels, ' \
                    f'but loaded images have {inputs.shape[1]} channels. Please check that ' \
                    'the inputs are loaded correctly.'

                # todo: check if these need requires_grad = true
                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=dtype)
                targets_mask = targets_mask.to(device=device, dtype=dtype)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=cfg['amp']):
                    pred = model(inputs)
                    loss = criterion(pred, targets, targets_mask)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clipping'])
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(inputs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if not no_wandb:
                    wandb_run.log({
                        'train loss': loss.item(),
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'avg loss/img': epoch_loss / float(i+1)})

        # Evaluation round
        # division_step = (len(train_loader) // (5 * cfg['batch_size']))
        if 1: # division_step > 0:
            if 1: # global_step % division_step == 0:
                val_score = evaluate(model, val_loader, criterion, 
                                     device, cfg['amp'], dtype, cfg=cfg,
                                     wandb_run=wandb_run)
                scheduler.step(val_score)

        # Save model every epoch
        if cfg['save_checkpoint']:
            Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(Path(cfg['path_checkpoints']) / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

@click.command()
@click.option('--load', '-f', type=str, default=None, help='Load model from a .pth file')
@click.option('--cfg_path', type=str, default='runs/unet/default/config/config.yaml', help='Path to config yaml')
@click.option('--parallel', is_flag=True, show_default=True, default=False, help='Enable parallel training')
@click.option('--no_wandb', is_flag=True, show_default=True, default=False, help='Disable wandb logs')
@click.option('--verbose', is_flag=True, show_default=True, default=False, help='Set true to print verbose logs')
@click.option('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array')
@click.option('--num_tasks',type=int, default=1, help='Total number of SLURM tasks when script is called in job array')
@click.option('--sweep', is_flag=True, show_default=True, default=False, help='If true, indicates that program is running a hyperparameter sweep')
def main(load, cfg_path, parallel, no_wandb, verbose, task_id, num_tasks, sweep):
    """
    This function trains the machine learning which is specified in cfg
    """

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Import cfg and set seeds
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    cfg['path_sweep_cfg'] = cfg_path
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    if verbose:
        print('Default model configuration:')
        pprint(cfg)

    # Initialize hyperparameter sweep
    if sweep:
        cfg = init_sweep_config(cfg, cfg['path_sweep_cfg'], task_id, num_tasks)

    if True: # cfg['model_type'] == 'fcnn':
        from emcli.models.fcnn.model import FCNN
        model = FCNN(dim_in=cfg['in_channels'],
            dim_out=cfg['out_channels'],
            n_layers=cfg['n_layers'],
            n_units=cfg['n_units'],
            resNet=cfg['resNet'], 
            n_res_blocks=cfg['n_res_blocks']
            )
    
    model = model.to(memory_format=torch.channels_last)

    if load:
        state_dict = torch.load(load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {load}')

    model.to(device=device)

    train_args = {
        'model' : model,
        'device' : device,
        'parallel' : parallel,
        'no_wandb' : no_wandb,
        'sweep' : sweep,
        'cfg' : cfg
    }
    try:
        train_model(**train_args)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(**train_args)

if __name__ == '__main__':
    main()