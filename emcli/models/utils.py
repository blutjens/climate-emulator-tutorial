import random
import yaml
from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, List, Any

import torch

def lookup_torch_dtype(dtype_name: str) -> Any:
    """
    Returns torch dtype given a string
    """
    if dtype_name == 'float16' or dtype_name == 'half':
        return torch.float16
    elif dtype_name == 'float32' or dtype_name == 'float':
        return torch.float32
    elif dtype_name == 'float64' or dtype_name == 'double':
        return torch.float64
    else:
        raise NotImplementedError('only float32 implemented')

def generate_dicts_recursive(input_dict, current_dict=None, depth=0):
    '''
    Recursively generates a list of dictionaries containing
    all possible combinations of the values
    source: chat gpt-3.5
    Args:
        keys list(): List of dictionary keys
        input_dict dict(): Input dictionary
    '''
    keys = list(input_dict.keys())

    if current_dict is None:
        current_dict = {}

    if depth == len(keys):
        return [current_dict]

    key = keys[depth]
    values = input_dict[key]
    result_dicts = []

    for value in values:
        new_dict = current_dict.copy()
        new_dict[key] = value
        result_dicts.extend(generate_dicts_recursive(input_dict, new_dict, depth + 1))

    return result_dicts

def init_sweep_config(cfg, path_sweep_cfg, task_id=1, num_tasks=1):
    '''
    Updates the cfg with a randomly drawn combination of 
    hyperparameters from the sweep config. 
    '''
    # Update logging paths
    cfg['path_wandb'] = Path(cfg['path_sweep'] / Path(f'task-{task_id}'))
    cfg['path_checkpoints'] = Path(cfg['path_sweep'] / Path(f'task-{task_id}/checkpoints'))
    Path(cfg['path_wandb']).mkdir(parents=True, exist_ok=True)

    # Update config with sweep parameters
    sweep_cfg = yaml.safe_load(open(cfg['path_sweep_cfg'], 'r'))
    # Initialize list of all possible cfg combinations
    list_of_sweep_cfgs = generate_dicts_recursive(sweep_cfg)
    print(f'Running {num_tasks}/{len(list_of_sweep_cfgs)} random sweep configurations on all tasks.')
    # Randomly shuffle the combinations and then draw the element with index
    # task.id. This is necessary because all tasks run on different GPUs, but
    # share the same random seed.
    random.shuffle(list_of_sweep_cfgs)
    current_sweep_cfg = list_of_sweep_cfgs[task_id-1] # minus 1 switches from 1 to zero indexing

    # Update the main config with the parameters chosen for this sweep
    cfg.update(current_sweep_cfg)
    print('Choosing sweep configuration:')
    pprint(current_sweep_cfg)

    return cfg