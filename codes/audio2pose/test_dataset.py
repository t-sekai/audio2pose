import torch
from dataloaders.beat import CustomDataset
from utils import config
from dataloaders.build_vocab import Vocab

args = config.parse_args()
train_data = CustomDataset(args, "train")
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=args.batch_size,  
    shuffle=False,  
    drop_last=True,
)

data = next(iter(train_loader))
print(data["audio"])