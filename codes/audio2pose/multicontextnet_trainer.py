from time import time
import numpy as np
import torch
import os

class Trainer():
    def __init__(self, args, device, train_data, val_data, model, logger):
        # Set up data loading
        self.mean_facial = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy"))
        self.std_facial = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy"))
        self.mean_audio = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy"))
        self.std_audio = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy"))

        self.batch_size = args.batch_size
        self.train_data = train_data
        self.train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=self.batch_size,  
            shuffle=True,  
            drop_last=True,
        )
        self.val_data = val_data

        # Set up model and loss functions
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.target_loss_function = torch.nn.HuberLoss()
        self.smooth_loss_function = torch.nn.CosineSimilarity(dim=2)
        self.mse_loss_function = torch.nn.MSELoss()
        self.to_flame = torch.from_numpy(np.load('mat_final.npy')).to(device)

        # Set up training/validation parameters
        self.epochs = args.epochs
        self.target_weight = args.target_weight
        self.smooth_weight = args.smooth_weight
        self.smooth_weight = args.smooth_weight
        self.val_size = args.val_size
        self.log_period = args.log_period
        self.val_period = args.val_period

        # Set up logging
        self.logger = logger
        self._iter = 0
        self._ep_idx = 0
        self._start_time = time()

        # Checkpoint
        self.ckpt_exp_dir = f'{args.wandb_project}-{args.wandb_group}-{str(args.random_seed)}'
        self.save_period = args.save_period
        self.save_ckpt = args.save_ckpt
        self.ckpt_dir = args.ckpt_dir
        self.ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_exp_dir)
        if self.save_ckpt:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            iteration=self._iter,
            epoch=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def save_checkpoint(self, name):
        if self.save_ckpt:
            pth_path = os.path.join(self.ckpt_path, f'multicontextnet-{name}.pth')
            torch.save(self.model.state_dict(), pth_path)
    
    def expressive_loss_function(output, target): # max squared error over blendshape for each frame, then take the mean
        loss = torch.mean(torch.max((output - target) ** 2, dim=-1).values)
        return loss

    def val(self):
        self.model.eval()
        val_target_loss_st = []
        #val_expressive_loss_st = []
        val_smooth_loss_st = []
        val_cnt = 0

        val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size,  
            shuffle=True,  
            drop_last=True,
        )
        
        for _, data in enumerate(val_loader):
            in_audio = data['audio']
            facial = data['facial']
            in_id = data["id"]
            in_word = data["word"]
            in_emo = data["emo"]

            in_audio = in_audio.cuda()
            facial = facial.cuda()
            in_id = in_id.cuda()
            in_word = in_word.cuda()
            in_emo = in_emo.cuda()

            pre_frames = 4
            in_pre_face = facial.new_zeros((facial.shape[0], facial.shape[1], facial.shape[2] + 1)).cuda()
            in_pre_face[:, 0:pre_frames, :-1] = facial[:, 0:pre_frames]
            in_pre_face[:, 0:pre_frames, -1] = 1 

            out_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
            target_loss = self.target_loss_function(out_face@self.to_flame, facial@self.to_flame) + self.target_loss_function(out_face[:,:,6:14], facial[:,:,6:14])
            #expressive_loss = self.expressive_loss_function(out_face, facial)
            smooth_loss = 1 - self.smooth_loss_function(out_face[:,:-1,:], out_face[:,1:,:]).mean()

            val_target_loss_st.append(target_loss.item())
            #val_expressive_loss_st.append(expressive_loss.item())
            val_smooth_loss_st.append(smooth_loss.item())                        
            
            val_cnt += 1
            if val_cnt >= self.val_size:
                break
        return {
			"target_loss": float(np.average(val_target_loss_st)),
			"smooth_loss": float(np.average(val_smooth_loss_st)),
			#"expressive_loss": float(np.average(val_expressive_loss_st)),
		}

    def train(self):
        for self._ep_idx in range(self.epochs):
            for it, data in enumerate(self.train_loader):
                self.model.train()
                in_audio = data['audio']
                facial = data['facial']
                in_id = data["id"]
                in_word = data["word"]
                in_emo = data["emo"]

                in_audio = in_audio.cuda()
                facial = facial.cuda()
                in_id = in_id.cuda()
                in_word = in_word.cuda()
                in_emo = in_emo.cuda()

                pre_frames = 4
                in_pre_face = facial.new_zeros((facial.shape[0], facial.shape[1], facial.shape[2] + 1)).cuda()
                in_pre_face[:, 0:pre_frames, :-1] = facial[:, 0:pre_frames]
                in_pre_face[:, 0:pre_frames, -1] = 1 
                
                self.optimizer.zero_grad()
                out_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
                target_loss = self.target_loss_function(out_face@self.to_flame, facial@self.to_flame) + self.target_loss_function(out_face[:,:,6:14], facial[:,:,6:14]) # to account for eye movement
                #expressive_loss = self.expressive_loss_function(out_face, facial)
                smooth_loss = 1 - self.smooth_loss_function(out_face[:,:-1,:], out_face[:,1:,:]).mean()
                loss = self.target_weight * target_loss  + self.smooth_weight * smooth_loss# + expressive_weight * expressive_loss
                loss.backward()
                self.optimizer.step()

                if it % self.log_period == 0:
                    train_metrics = {
                        "target_loss": float(target_loss.item()),
                        "smooth_loss": float(smooth_loss.item()),
                        #"expressive_loss": float(expressive_loss.item()),
                    }
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [train] [target loss]: {train_metrics["target_loss"]} [smooth loss]: {train_metrics["smooth_loss"]}')

                if it % self.val_period == 0:
                    val_metrics = self.val()
                    val_metrics.update(self.common_metrics())
                    self.logger.log(val_metrics,'val')
                    print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [val] [target loss]: {val_metrics["target_loss"]} [smooth loss]: {val_metrics["smooth_loss"]}')

                self._iter += 1
            if (self._ep_idx+1) % self.save_period == 0:
                self.save_checkpoint(str(self._ep_idx+1))
        self.logger.finish()
