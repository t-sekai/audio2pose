# UI
import tkinter as tk
from tkinter import font
import sounddevice as sd
import librosa
import soundfile as sf
import time
from PIL import Image, ImageTk
import numpy as np
from ctypes import windll

# GestureGen
from pythonosc import udp_client
import torch
from dataloaders.beat import CustomDataset
from dataloaders.build_vocab import Vocab
import pickle
import os
from utils.other_tools import load_checkpoints
from models.camn import CaMN
from scripts.MulticontextNet import GestureGen
from joints_list import JOINTS_LIST
from blendshapes import BLENDSHAPE_NAMES
import json
import threading
from collections import OrderedDict


# ---------------------------------------------- GestureGen Initialization -------------------------- #

class HMGestureGen():
    def __init__(self, blender_load_path, template_speaker = 17):
        # arguments
        self.blender_load_path = blender_load_path
        self.template_speaker = template_speaker

        # set up args
        camn_config_file = open("camn_config.obj", 'rb') 
        gesturegen_config_file = open("gesturegen_config.obj", 'rb')
        self.gesturegen_args = pickle.load(gesturegen_config_file)
        self.camn_args = pickle.load(camn_config_file)

        # set up std/mean
        self.mean_facial = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.facial_rep}/json_mean.npy")).float()
        self.std_facial = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.facial_rep}/json_std.npy")).float()
        self.mean_audio = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.audio_rep}/npy_mean.npy")).float()
        self.std_audio = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.audio_rep}/npy_std.npy")).float()
        self.mean_pose = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.pose_rep}/bvh_mean.npy")).float()
        self.std_pose = torch.from_numpy(np.load(self.camn_args.root_path+self.camn_args.mean_pose_path+f"{self.camn_args.pose_rep}/bvh_std.npy")).float()

        # set up template
        test_data = CustomDataset(self.camn_args, "test")
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False,)
        for its, template in enumerate(self.test_loader):
            if template['id'][0] == self.template_speaker:
                break
        self.template = template
        test_demo = self.camn_args.root_path + self.camn_args.test_data_path + f"{self.camn_args.pose_rep}_vis/"
        test_seq_list = sorted(os.listdir(test_demo))
        self.template_bvh = test_seq_list[its]

        # set up multicontextnet
        print('Loading MulticontextNet')
        self.multicontextnet = GestureGen(self.gesturegen_args)
        self.multicontextnet.load_state_dict(torch.load('tmp/multicontextnet-no-text-normalized.pth'))
        self.multicontextnet = self.multicontextnet.cuda().eval()

        # set up camn
        print('Loading CaMN')
        self.camn = CaMN(self.camn_args)
        states = torch.load('tmp/camn.bin')
        new_weights = OrderedDict()
        for k, v in states['model_state'].items():
            new_weights[k[7:]] = v
        self.camn.load_state_dict(new_weights)
        self.camn = self.camn.cuda().eval()
        
        # set up input data
        self.audio_data = None
        self.orig_sr = None
        self.pred_facial = None
        self.pred_pose = None
    
    def load_gandhi_speech(self):
        test_audio_file = 'test_audio/gandhi-speech.wav'
        self.audio_data, self.orig_sr = librosa.load(test_audio_file, duration=30, sr=None) # np array
    
    def load_recording(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        duration = int(len(y) / sr)
        self.audio_data, self.orig_sr = librosa.load(file_path, sr=None, duration=duration)

    def predict(self, predict_pose=True):
        out_audio = librosa.resample(self.audio_data, orig_sr=self.orig_sr, target_sr=16000) # convert to 16khz
        out_audio = torch.from_numpy(out_audio).unsqueeze(0)
        audio = (out_audio - self.mean_audio) / self.std_audio

        in_audio = audio.cuda()
        in_id = torch.zeros((1, 1)).int().cuda()
        in_id[0] = self.template_speaker
        in_emo = torch.zeros((1, in_audio.shape[1]//16000*15)).int() + 0
        in_emo = in_emo.cuda()
        pre_frames = 4
        in_pre_facial = torch.zeros((1,in_audio.shape[1]//16000*15, 52)).float().cuda()
        in_pre_facial[:, 0:pre_frames, -1] = 1 

        pred_facial_normed = self.multicontextnet(in_pre_facial, in_audio=in_audio, in_id=in_id, in_emo=in_emo)
        self.pred_facial = np.array(pred_facial_normed.cpu().detach()[0] * self.std_facial + self.mean_facial)

        if predict_pose:
            pre_frames = 4
            pre_pose = torch.zeros((1, pred_facial_normed.shape[1], self.gesturegen_args.pose_dims + 1)).cuda()
            pre_pose[:, 0:pre_frames, :-1] = self.template['pose'][:, 0:pre_frames]
            pre_pose[:, 0:pre_frames, -1] = 1

            out_dir_vec = self.camn(pre_seq=pre_pose, in_audio=in_audio, in_facial=pred_facial_normed, in_id=in_id, in_emo=in_emo)
            self.pred_pose = np.array((out_dir_vec.cpu().detach().reshape(-1, self.camn_args.pose_dims) * self.std_pose) + self.mean_pose)
    
    def result2target_vis(self, bvh_file, res_frames):
        ori_list = JOINTS_LIST["beat_joints"]
        target_list = JOINTS_LIST["spine_neck_141"]
        file_content_length = 431

        template_bvh_path = f"{self.camn_args.root_path}/datasets/beat_cache/beat_4english_15_141/test/bvh_rot_vis/{self.template_bvh}"

        short_name = bvh_file.split("\\")[-1][11:]
        save_file_path = os.path.join(self.blender_load_path, os.path.join('b6_correct_no_root',f'res_{short_name}'))
        
        with open(template_bvh_path,'r') as pose_data_pre:
            pose_data_pre_file = pose_data_pre.readlines()
            ori_lines = pose_data_pre_file[:file_content_length]
            offset_data = np.fromstring(pose_data_pre_file[file_content_length], dtype=float, sep=' ')

        ori_lines[file_content_length-2] = 'Frames: ' + str(res_frames) + '\n'
        ori_lines[file_content_length-1] = 'Frame Time: 0.066667\n'

        write_file = open(save_file_path,'w+')
        write_file.writelines(i for i in ori_lines[:file_content_length])    
        write_file.close() 

        with open(save_file_path,'a+') as write_file: 
            with open(bvh_file, 'r') as pose_data:
                data_each_file = []
                pose_data_file = pose_data.readlines()
                for j, line in enumerate(pose_data_file):
                    if not j:
                        pass
                    else:          
                        data = np.fromstring(line, dtype=float, sep=' ')
                        data_rotation = offset_data.copy()   
                        for iii, (k, v) in enumerate(target_list.items()): # here is 147 rotations by 3
                            data_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[iii*3:iii*3+3]
                        data_each_file.append(data_rotation)
        
            for line_data in data_each_file:
                line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                write_file.write(line_data[1:-2]+'\n')
    
    def bs2json(self, facial, bvh_file):
        short_name = bvh_file.split("\\")[-1][11:-4]
        save_file_path = os.path.join(self.blender_load_path, os.path.join('b4_json_rename',f'res_{short_name}.json'))
        with open(save_file_path, "w") as res_json:
            new_frames_list = []
            time = 0.016666666666666666
            for weights in facial:
                new_frames_list.append({'weights': weights, 'time': time, 'rotation': []})
                time += 1/15
            json_new = {"names":BLENDSHAPE_NAMES[:-1], "frames": new_frames_list}
            json.dump(json_new, res_json)

    def save_for_blender(self):
        if self.pred_facial is None or self.pred_pose is None:
            raise Exception('Face/Pose not predicted before saving for Blender!')
        
        # generate bvh file
        res_file = os.path.join("ui_src","result_raw_demo.bvh")
        with open(res_file, 'w+') as f_real:
            for line_id in range(self.pred_pose.shape[0]): #,args.pre_frames, args.pose_length
                line_data = np.array2string(self.pred_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                f_real.write(line_data[1:-2]+'\n')  
        res_frames = self.pred_pose.shape[0] - 1
        self.result2target_vis(res_file, res_frames)

        # generate json file
        self.bs2json(self.pred_facial.tolist(), res_file)

        # generate wav file
        short_name = res_file.split("\\")[-1][11:-4]
        save_file_path = os.path.join(self.blender_load_path, os.path.join('b2_wave_rename',f'res_{short_name}.wav'))
        sf.write(save_file_path, self.audio_data, self.orig_sr)

    def play_audio(self, out_audio, sr, init_time):
        time.sleep(init_time - time.time())
        sd.play(out_audio, sr)
        sd.wait()

    def send_udp(self, out_face, init_time, reset_duration=1.0):
        outWeight = out_face

        outWeight = outWeight * (outWeight >= 0)

        client = udp_client.SimpleUDPClient('127.0.0.1', 5008)
        osc_array = outWeight.tolist()
        
        fps = 15
        time.sleep(init_time - time.time())
        for i in range(len(osc_array)):
            for j, out in enumerate(osc_array[i]):
                client.send_message('/' + str(BLENDSHAPE_NAMES[j]), out)

            elpased_time = time.time() - init_time
            sleep_time = 1.0/fps * (i+1) - elpased_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        # reset blendshape to 0
        final_weight = outWeight[-1]
        step_decrement = final_weight / (fps*reset_duration)
        
        for i in range(int(fps*reset_duration)):
            final_weight -= step_decrement
            final_weight = np.maximum(final_weight, 0)

            for j, out in enumerate(final_weight.tolist()):
                client.send_message('/' + str(BLENDSHAPE_NAMES[j]), out)
            time.sleep(reset_duration / fps)
        
        final_weight = np.zeros((51,))
        for j, out in enumerate(final_weight.tolist()):
            client.send_message('/' + str(BLENDSHAPE_NAMES[j]), out)
    
    def send_to_ue5(self):
        init_time = time.time() + 1

        # facial_ue = pred_facial
        # facial_ue[:,:,8:24] = 0
        # # blinking animation
        # blink_interval = 5 # in seconds
        # for i in range(blink_interval*15,limit_sec*15,blink_interval*15):
        #     blink_duration = 2 # in frames
        #     for j in range(i-blink_duration,i): # start blinking
        #         facial_ue[:,j,8:10] = 1-(i-j)/blink_duration
        #     for j in range(i, i+blink_duration):
        #         facial_ue[:,j,8:10] = 1-(j-i)/blink_duration

        udp_thread = threading.Thread(target=self.send_udp, args=(self.pred_facial, init_time, 0.5))
        udp_thread.daemon = True

        audio_thread = threading.Thread(target=self.play_audio, args=(self.audio_data, self.orig_sr, init_time+0.17))
        audio_thread.daemon = True

        udp_thread.start()
        audio_thread.start()

        udp_thread.join()
        audio_thread.join()

# -------------------------------------------------------- UI ---------------------------------------- #

class GestureGenUI():
    def __init__(self, gesturegen: HMGestureGen):
        self.gesturegen = gesturegen
        self.recording = False
        self.audio_data = None
        self.start_time = None

        sd.default.samplerate = 44100

    def init_ui(self):
        windll.shcore.SetProcessDpiAwareness(1)
        self.root = tk.Tk()
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.config(bg="#F6FCFE")

        # icon
        image_icon = tk.PhotoImage(file="ui_src/icon.png")
        self.root.iconphoto(True, image_icon)

        # logo
        image = Image.open("ui_src/image.png")
        resized_image = image.resize((200, 200))
        self.logo_image = ImageTk.PhotoImage(resized_image)

        # font
        custom_font = font.Font(family="Courier",size=35,weight="bold")

        # title
        label = tk.Label(self.root, text="GestureGen", font=custom_font, bg="#F6FCFE",fg="#3B3024" )
        label.place(relx=0.5, rely=0.02, anchor="n")

        # start record button
        self.start_button = tk.Button(self.root, text="Start Recording",font=(custom_font.actual()['family'], 20), command=self.toggle_recording,\
                                bg="#0B3353",fg="white", width=18)
        self.start_button.place(relx=0.2,rely=0.5)

        self.animate_label = tk.Label(self.root, text="Animate on", font=(custom_font.actual()['family'], 15, 'bold'), bg="#F6FCFE",fg="#3B3024" )

        # to ue5 button
        self.to_ue5 = tk.Button(self.root, text="UE5",font=(custom_font.actual()['family'], 20), command=self.send_to_ue5,\
                                bg="#0B3353",fg="white", width=8)

        # to blender button
        self.to_blender = tk.Button(self.root, text="Blender",font=(custom_font.actual()['family'], 20), command=self.save_for_blender,\
                                bg="#0B3353",fg="white", width=8)

        # recording timer
        self.timer_label = tk.Label(self.root, text="", font=(custom_font.actual()['family'], 20, 'bold'), bg="#F6FCFE", fg="#3B3024")
        self.timer_label.place(relx=0.15,rely=0.85)
    

    def start(self):      
        logo = tk.Button(self.root, image=self.logo_image, bg="#F6FCFE", highlightthickness = 0, bd = 0, command=self.load_demo)
        logo.place(relx=0.5, rely=0.125, anchor="n")  
        self.root.mainloop()
    
    def load_demo(self):
        self.gesturegen.load_gandhi_speech()
        self.gesturegen.predict(predict_pose=True)
        self.to_ue5.place(relx=0.2,rely=0.7)
        self.to_blender.place(relx=0.52,rely=0.7)
        self.animate_label.place(relx=0.5, rely=0.625, anchor="n")

    def send_to_ue5(self):
        self.gesturegen.send_to_ue5()
    
    def save_for_blender(self):
        self.gesturegen.save_for_blender()

    def toggle_recording(self):

        if not self.recording:
            # Start recording
            self.start_time = time.time()
            self.recording = True
            self.start_button.config(text="Stop Recording", bg="#C4E4F4", fg="black")
            self.to_ue5.place_forget()
            self.to_blender.place_forget()
            self.animate_label.place_forget()
            
            self.audio_data = sd.rec(120 * sd.default.samplerate, samplerate=sd.default.samplerate, channels=2, dtype=np.float32)
            self.start_timer()
        else:
            # Stop recording
            self.recording = False
            self.start_button.config(text="Start Recording", bg="#0B3353", fg="white")
            sd.stop()
            sd.wait()

            if self.audio_data is not None:
                elapsed_time = time.time() - self.start_time
                expected_samples = int(elapsed_time * sd.default.samplerate)
                self.audio_data = self.audio_data[:expected_samples]

                record_file_path = 'ui_src/recording.wav'
                sf.write(record_file_path, self.audio_data, samplerate=sd.default.samplerate)

                self.gesturegen.load_recording(record_file_path)
                self.gesturegen.predict(predict_pose=True)

                self.to_ue5.place(relx=0.2,rely=0.7)
                self.to_blender.place(relx=0.52,rely=0.7)
                self.animate_label.place(relx=0.5, rely=0.625, anchor="n")

    # Function to update the recording timer
    def start_timer(self):
        if self.recording:
            elapsed_time = time.time() - self.start_time
            self.timer_label.config(text=f"Recording : {int(elapsed_time)} seconds")
            self.timer_label.after(1000, self.start_timer)

if __name__ == '__main__':
    gesturegen = HMGestureGen(blender_load_path='S:/rendervideo_mod/rendervideo', template_speaker=17)
    ui = GestureGenUI(gesturegen)
    ui.init_ui()
    ui.start()