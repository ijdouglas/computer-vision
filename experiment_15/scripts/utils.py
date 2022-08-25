import torch
import os
import math
import random 
import numpy as np
import time

def setup_seed(seed):
    if seed < 0:
        if os.getenv('SATOSHI_SEED') is not None and seed == -2:
            seed = int(os.getenv('SATOSHI_SEED'))
            print("env seed used")
        else:
            import math
            seed = int(10**4*math.modf(time.time())[0])
            seed = seed
    print("random seed",seed)
    return seed

def make_deterministic(seed, strict=False):
    #https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        #https://github.com/pytorch/pytorch/issues/7068#issuecomment-515728600
        torch.backends.cudnn.enabled = False
        print("strict reproducability required! cudnn disabled. make sure to set num_workers=0 too!")