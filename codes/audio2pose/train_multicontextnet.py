from dataloaders.beat import CustomDataset
from dataloaders.build_vocab import Vocab
import pickle
import numpy as np
from utils import config
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scripts.MulticontextNet import GestureGen
from scripts.Logger import Logger
import wandb
import random
import uuid
from tqdm import tqdm
from multicontextnet_trainer import Trainer

def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = config.parse_args()    
    set_seed(args.random_seed)
    logger = Logger(args)
    train_data = CustomDataset(args, "train")
    val_data = CustomDataset(args, "val")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available() == True
   
    model = GestureGen(args)
    if args.continue_training:
          model.load_state_dict(torch.load(args.pretrained_model))

    trainer = Trainer(args, device, train_data, val_data, model, logger)

    trainer.train()

