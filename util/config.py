from easydict import EasyDict
import torch
import os
import json

config = EasyDict()

def load_config_from_json(json_path):
    """
    Load config from json file and flatten it to config object
    """
    if not os.path.exists(json_path):
        print(f"Warning: Config file {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Flatten the json configs
    for category, items in data.items():
        for k, v in items.items():
            config[k] = v
            
    # Set device
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')

def update_config(config, extra_config):
    # Pass 1: Parse standard arguments
    for k, v in vars(extra_config).items():
        if k in config:
             # Only update if explicitly different from default? 
             # Or just overwrite. Argparse defaults might be issue.
             # If option.py has defaults, they will overwrite json.
             # We should ensure option.py defaults match json or are None.
             config[k] = v
        else:
            # If it's a new arg from argparse not in json
             config[k] = v
             
    # Pass 2: Re-evaluate device in case cuda flag changed
    config.device = torch.device('cuda') if config.get('cuda', False) else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')

# Load default config immediately
default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'config.json')
load_config_from_json(default_config_path)
