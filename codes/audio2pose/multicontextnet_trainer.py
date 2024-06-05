from time import time
import numpy as np
import torch
import os

import trimesh
from blendshapes import BLENDSHAPE_NAMES

class Trainer():
    def __init__(self, args, device, train_data, val_data, model, logger):
        # Set up data loading
        self.mean_facial = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")).float()
        self.std_facial = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")).float()
        self.mean_audio = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")).float()
        self.std_audio = torch.from_numpy(np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")).float()

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

        # Set up blendshape to flame parameters
        self.bs_to_flame = torch.from_numpy(np.load('mat_final.npy')).to(device)
        self.flame_to_bs = self.bs_to_flame.pinverse()
        self.predict_flame = args.predict_flame
        self.pre_frames = args.pre_frames

        # Set up blendshape to vertices
        self.V_factor = 100
        self.V_basis = torch.tensor(trimesh.load('bs/Basis.obj').vertices, dtype=torch.float32) * self.V_factor
        self.V_bs = torch.stack([torch.tensor(trimesh.load(f'bs/exp/{bs_name}.obj').vertices, dtype=torch.float32) for bs_name in BLENDSHAPE_NAMES[:51]]) * self.V_factor
        self.V_deltas = (self.V_bs - self.V_basis.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(device)

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
        val_V_smooth_loss_st = []
        val_flame_smooth_loss_st = []
        #val_bs_smooth_loss_st = []
        val_cnt = 0

        val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size,  
            shuffle=True,  
            drop_last=True,
        )
        
        for _, data in enumerate(val_loader):
            in_audio = data['audio']
            bs_facial = data['facial'] * self.std_facial + self.mean_facial
            in_id = data["id"]
            in_word = data["word"]
            in_emo = data["emo"]

            in_audio = in_audio.cuda()
            bs_facial = bs_facial.cuda()
            in_id = in_id.cuda()
            in_word = in_word.cuda()
            in_emo = in_emo.cuda()
            
            if self.predict_flame:
                flame_facial = torch.cat((bs_facial @ self.bs_to_flame, bs_facial[:,:,6:14]), dim=-1)
                in_pre_face = flame_facial.new_zeros((flame_facial.shape[0], flame_facial.shape[1], flame_facial.shape[2])).cuda()
                in_pre_face[:, 0:self.pre_frames] = flame_facial[:, 0:self.pre_frames]

                flame_out_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
                target_loss = self.target_loss_function(flame_out_face, flame_facial)
                flame_smooth_loss = 1 - self.smooth_loss_function(flame_out_face[:,:-1,:], flame_out_face[:,1:,:]).mean()

                val_target_loss_st.append(target_loss.item())
                val_flame_smooth_loss_st.append(flame_smooth_loss.item()) 

            else: 
                in_pre_face = bs_facial.new_zeros((bs_facial.shape[0], bs_facial.shape[1], bs_facial.shape[2] + 1)).cuda()
                in_pre_face[:, 0:self.pre_frames, :-1] = bs_facial[:, 0:self.pre_frames]
                in_pre_face[:, 0:self.pre_frames, -1] = 1 

                bs_pred_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
                V_pred_face = torch.sum(bs_pred_face.unsqueeze(3).unsqueeze(4) * self.V_deltas, axis=2)
                V_gt_face = torch.sum(bs_facial.unsqueeze(3).unsqueeze(4) * self.V_deltas, axis=2)
                
                target_loss = self.target_loss_function(V_pred_face, V_gt_face)
                V_smooth_loss = 1 - self.smooth_loss_function(V_pred_face[:,:-1,:], V_pred_face[:,1:,:]).mean()

                val_target_loss_st.append(target_loss.item())
                val_V_smooth_loss_st.append(V_smooth_loss.item())      
            
            val_cnt += 1
            if val_cnt >= self.val_size:
                break
        if self.predict_flame:
            return {
                "target_loss": float(np.average(val_target_loss_st)),
                "flame_smooth_loss": float(np.average(val_flame_smooth_loss_st)),
            }
        else:
            return {
                "target_loss": float(np.average(val_target_loss_st)),
                "V_smooth_loss": float(np.average(val_V_smooth_loss_st)),
                #"bs_smooth_loss": float(np.average(val_bs_smooth_loss_st)),
                #"expressive_loss": float(np.average(val_expressive_loss_st)),
            }

    def train(self):
        for self._ep_idx in range(self.epochs):
            for it, data in enumerate(self.train_loader):
                self.model.train()
                in_audio = data['audio']
                bs_facial = data['facial'] * self.std_facial + self.mean_facial
                in_id = data["id"]
                in_word = data["word"]
                in_emo = data["emo"]

                in_audio = in_audio.cuda()
                bs_facial = bs_facial.cuda()
                in_id = in_id.cuda()
                in_word = in_word.cuda()
                in_emo = in_emo.cuda()
                
                if self.predict_flame:
                    flame_facial = torch.cat((bs_facial @ self.bs_to_flame, bs_facial[:,:,6:14]), dim=-1)
                    in_pre_face = flame_facial.new_zeros((flame_facial.shape[0], flame_facial.shape[1], flame_facial.shape[2])).cuda()
                    in_pre_face[:, 0:self.pre_frames] = flame_facial[:, 0:self.pre_frames]

                    self.optimizer.zero_grad()
                    flame_out_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
                    target_loss = self.target_loss_function(flame_out_face, flame_facial)
                    flame_smooth_loss = 1 - self.smooth_loss_function(flame_out_face[:,:-1,:], flame_out_face[:,1:,:]).mean()
                    loss = self.target_weight * target_loss  + self.smooth_weight * flame_smooth_loss
                    loss.backward()
                    self.optimizer.step()

                    if it % self.log_period == 0:
                        train_metrics = {
                            "target_loss": float(target_loss.item()),
                            "flame_smooth_loss": float(flame_smooth_loss.item()),
                        }
                        train_metrics.update(self.common_metrics())
                        self.logger.log(train_metrics, 'train')
                        print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [train] [target loss]: {train_metrics["target_loss"]} flame smooth loss]: {train_metrics["flame_smooth_loss"]}')

                else: 
                    in_pre_face = bs_facial.new_zeros((bs_facial.shape[0], bs_facial.shape[1], bs_facial.shape[2] + 1)).cuda()
                    in_pre_face[:, 0:self.pre_frames, :-1] = bs_facial[:, 0:self.pre_frames]
                    in_pre_face[:, 0:self.pre_frames, -1] = 1 

                    self.optimizer.zero_grad()
                    bs_pred_face = self.model(in_pre_face,in_audio=in_audio,in_text=in_word, in_id=in_id, in_emo=in_emo)
                    V_pred_face = torch.sum(bs_pred_face.unsqueeze(3).unsqueeze(4) * self.V_deltas, axis=2)
                    V_gt_face = torch.sum(bs_facial.unsqueeze(3).unsqueeze(4) * self.V_deltas, axis=2)

                    target_loss = self.target_loss_function(V_pred_face, V_gt_face)
                    V_smooth_loss = 1 - self.smooth_loss_function(V_pred_face[:,:-1,:], V_pred_face[:,1:,:]).mean()
                    #expressive_loss = self.expressive_loss_function(out_face, facial)
                    #flame_smooth_loss = 1 - self.smooth_loss_function(flame_out_face[:,:-1,:], flame_out_face[:,1:,:]).mean()
                    #bs_smooth_loss = 1 - self.smooth_loss_function(bs_out_face[:,:-1,:], bs_out_face[:,1:,:]).mean()
                    loss = self.target_weight * target_loss  + self.smooth_weight * V_smooth_loss #self.smooth_weight * flame_smooth_loss + self.smooth_weight * bs_smooth_loss# + expressive_weight * expressive_loss
                    loss.backward()
                    self.optimizer.step()

                    if it % self.log_period == 0:
                        train_metrics = {
                            "target_loss": float(target_loss.item()),
                            "V_smooth_loss": float(V_smooth_loss.item())
                            #"expressive_loss": float(expressive_loss.item()),
                        }
                        train_metrics.update(self.common_metrics())
                        self.logger.log(train_metrics, 'train')
                        print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [train] [target loss]: {train_metrics["target_loss"]} vertices smooth loss]: {train_metrics["V_smooth_loss"]}')

                if it % self.val_period == 0:
                    val_metrics = self.val()
                    val_metrics.update(self.common_metrics())
                    self.logger.log(val_metrics,'val')
                    if self.predict_flame:
                        print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [val] [target loss]: {val_metrics["target_loss"]} [flame smooth loss]: {val_metrics["flame_smooth_loss"]}')
                    else:
                        print(f'[{self._ep_idx}][{it}/{len(self.train_loader)}]: [val] [target loss]: {val_metrics["target_loss"]} [vertices smooth loss]: {val_metrics["V_smooth_loss"]}')

                self._iter += 1
            if (self._ep_idx+1) % self.save_period == 0:
                self.save_checkpoint(str(self._ep_idx+1))
        self.logger.finish()
