# UI
import tkinter as tk
from tkinter import font
import sounddevice as sd
import soundfile as sf
import time
from PIL import Image, ImageTk
import numpy as np
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

# GestureGen
from pythonosc import udp_client
import sounddevice as sd
import torch
from dataloaders.beat import CustomDataset
from dataloaders.build_vocab import Vocab
import pickle
import os
from utils.other_tools import load_checkpoints
from models.camn import CaMN
from scripts.MulticontextNet import GestureGen

# ---------------------------------------------- GestureGen Initialization -------------------------- #

class MHGestureGen():
    def __init__(self, model_path = 'tmp/multicontextnet-no-text.pth', template_speaker = 17):
        # arguments
        self.model_path = model_path
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
        test_seq_list = os.listdir(test_demo).sort()
        self.template_bvh = test_seq_list[its]

        # set up model
        model_path = 'tmp/multicontextnet-no-text.pth'
        net = GestureGen(self.gesturegen_args)
        net.load_state_dict(torch.load(model_path))
        net = net.cuda().eval()

        facial_norm = False

# -------------------------------------------------------- UI ---------------------------------------- #

recording = False
audio_data = None
start_time = None

sd.default.samplerate = 44100

def toggle_recording():
    global recording
    global audio_data
    global start_time

    if not recording:
        # Start recording
        start_time = time.time()
        recording = True
        start_button.config(text="Stop Recording", bg="#C4E4F4", fg="black")
        audio_data = sd.rec(5 * sd.default.samplerate, samplerate=sd.default.samplerate, channels=2, dtype=np.float32)
        start_timer()
    else:
        # Stop recording
        recording = False
        start_button.config(text="Start Recording", bg="#0B3353", fg="white")
        sd.stop()
        sd.wait()

        if audio_data is not None:
            elapsed_time = time.time() - start_time
            expected_samples = int(elapsed_time * sd.default.samplerate)
            audio_data = audio_data[:expected_samples]

            sf.write('ui_src/recording.wav', audio_data, samplerate=sd.default.samplerate)

            to_ue5.place(relx=0.2,rely=0.7)
            to_blender.place(relx=0.52,rely=0.7)
            animate_label.place(relx=0.5, rely=0.625, anchor="n")

# Function to update the recording timer
def start_timer():
    global start_time
    if recording:
        elapsed_time = time.time() - start_time
        timer_label.config(text=f"Recording : {int(elapsed_time)} seconds")
        timer_label.after(1000, start_timer)

root = tk.Tk()
root.geometry("600x600")
root.resizable(False, False)
root.title("Voice Recorder")
root.config(bg="#F6FCFE")

# icon
image_icon = tk.PhotoImage(file="ui_src/icon.png")
root.iconphoto(True, image_icon)

# logo
image = Image.open("ui_src/image.png")
resized_image = image.resize((200, 200))
photo = ImageTk.PhotoImage(resized_image)
myimage = tk.Label(image=photo, bg="#F6FCFE")
myimage.place(relx=0.5, rely=0.125, anchor="n")

# font
custom_font = font.Font(family="Courier",size=35,weight="bold")

# title
label = tk.Label(root, text="GestureGen", font=custom_font, bg="#F6FCFE",fg="#3B3024" )
label.place(relx=0.5, rely=0.02, anchor="n")

# start record button
start_button = tk.Button(root, text="Start Recording",font=(custom_font.actual()['family'], 20), command=toggle_recording,\
                         bg="#0B3353",fg="white", width=18)
start_button.place(relx=0.2,rely=0.5)

animate_label = tk.Label(root, text="Animate on", font=(custom_font.actual()['family'], 15, 'bold'), bg="#F6FCFE",fg="#3B3024" )

# to ue5 button
to_ue5 = tk.Button(root, text="UE5",font=(custom_font.actual()['family'], 20), command=None,\
                         bg="#0B3353",fg="white", width=8)

# to blender button
to_blender = tk.Button(root, text="Blender",font=(custom_font.actual()['family'], 20), command=None,\
                         bg="#0B3353",fg="white", width=8)

# recording timer
timer_label = tk.Label(root, text="", font=(custom_font.actual()['family'], 20, 'bold'), bg="#F6FCFE", fg="#3B3024")
timer_label.place(relx=0.15,rely=0.85)


root.mainloop()