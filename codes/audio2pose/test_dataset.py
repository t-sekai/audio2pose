import torch
from dataloaders.beat import CustomDataset
from dataloaders.build_vocab import Vocab
import pickle
import numpy as np

config_file = open("camn_config.obj", 'rb') 
args = pickle.load(config_file)

train_data = CustomDataset(args, "train")
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=args.batch_size,  
    shuffle=False,  
    drop_last=True,
)

data = next(iter(train_loader))
facial = data["facial"]

mean_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
std_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")

facial = facial*std_facial+mean_facial

print(facial)
print(facial.min(), facial.max(), facial.std(),facial.mean())