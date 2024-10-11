from dataclasses import dataclass
import os
import random
import numpy as np
import torch

@dataclass
class CONFIG:
    debug: bool = True

    # Model
    V: int = 512  # 258 utf-8 characters + 2 special tokens
    P: int = 4
    T: int = 1024
    K: int = T // P  # Number of patches

    model_name: str = 'MEGABYTE'
    model_size: str = 'small'  # 'small' or 'large'

    ## Small
    if model_size == 'small':
        ### Global model
        n_layers_G: int = 6
        n_heads_G: int = 4
        d_G: int = 128
        d_head_G: int = d_G // n_heads_G
        d_ff_G: int = d_G * 4
        dropout_G: float = 0.1
        ### Local model
        n_layers_L: int = 4
        n_heads_L: int = 4
        d_L: int = 64
        d_head_L: int = d_L // n_heads_L
        d_ff_L: int = d_L * 4
        dropout_L: float = 0.1
    ### Large
    elif model_size == 'large':
        ### Global model
        n_layers_G: int = 12
        n_heads_G: int = 8
        d_G: int = 256
        d_head_G: int = d_G // n_heads_G
        d_ff_G: int = d_G * 4
        dropout_G: float = 0.1
        ### Local model
        n_layers_L: int = 8
        n_heads_L: int = 8
        d_L: int = 128
        d_head_L: int = d_L // n_heads_L
        d_ff_L: int = d_L * 4
        dropout_L: float = 0.2

    flash_attention: bool = False

    # Vocabulary
    PAD_ID: int = 256
    EOS_ID: int = 257

    # data
    validation_size: float = 0.2
    shakespeare_id = "data/shakespeare.txt"
    wiki_id = "wikimedia/wikipedia"
    dataset_id = wiki_id

    # Device
    device: torch.device = None

    # Training
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 2e-5  # 5e-4 ~ 1e-6

    # Generation
    max_len: int = 8192
    temperature: float = 1.0

    # Seed
    seed: int = 101

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f"Seed: {seed}")

def configure_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpu = torch.cuda.device_count()
        print("> Running on GPU", end=' | ')
        print("Num of GPUs: ", num_gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("> Running on MPS")
    else:
        device = torch.device("cpu")
        print("> Running on CPU")
    return device