import json
import os
import sys
import pandas as pd
import pickle
import math
import shutil
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *

if __name__ == "__main__":

    joint_list = {
        "beat_joints" : {
            'Hips':         [6,6],
            'Spine':        [3,9],
            'Spine1':       [3,12],
            'Spine2':       [3,15],
            'Spine3':       [3,18],
            'Neck':         [3,21],
            'Neck1':        [3,24],
            'Head':         [3,27],
            'HeadEnd':      [3,30],

            'RShoulder':    [3,33], 
            'RArm':         [3,36],
            'RArm1':        [3,39],
            'RHand':        [3,42],    
            'RHandM1':      [3,45],
            'RHandM2':      [3,48],
            'RHandM3':      [3,51],
            'RHandM4':      [3,54],

            'RHandR':       [3,57],
            'RHandR1':      [3,60],
            'RHandR2':      [3,63],
            'RHandR3':      [3,66],
            'RHandR4':      [3,69],

            'RHandP':       [3,72],
            'RHandP1':      [3,75],
            'RHandP2':      [3,78],
            'RHandP3':      [3,81],
            'RHandP4':      [3,84],

            'RHandI':       [3,87],
            'RHandI1':      [3,90],
            'RHandI2':      [3,93],
            'RHandI3':      [3,96],
            'RHandI4':      [3,99],

            'RHandT1':      [3,102],
            'RHandT2':      [3,105],
            'RHandT3':      [3,108],
            'RHandT4':      [3,111],

            'LShoulder':    [3,114], 
            'LArm':         [3,117],
            'LArm1':        [3,120],
            'LHand':        [3,123],    
            'LHandM1':      [3,126],
            'LHandM2':      [3,129],
            'LHandM3':      [3,132],
            'LHandM4':      [3,135],

            'LHandR':       [3,138],
            'LHandR1':      [3,141],
            'LHandR2':      [3,144],
            'LHandR3':      [3,147],
            'LHandR4':      [3,150],

            'LHandP':       [3,153],
            'LHandP1':      [3,156],
            'LHandP2':      [3,159],
            'LHandP3':      [3,162],
            'LHandP4':      [3,165],

            'LHandI':       [3,168],
            'LHandI1':      [3,171],
            'LHandI2':      [3,174],
            'LHandI3':      [3,177],
            'LHandI4':      [3,180],

            'LHandT1':      [3,183],
            'LHandT2':      [3,186],
            'LHandT3':      [3,189],
            'LHandT4':      [3,192],

            'RUpLeg':       [3,195],
            'RLeg':         [3,198],
            'RFoot':        [3,201],
            'RFootF':       [3,204],
            'RToeBase':     [3,207],
            'RToeBaseEnd':  [3,210],

            'LUpLeg':       [3,213],
            'LLeg':         [3,216],
            'LFoot':        [3,219],
            'LFootF':       [3,222],
            'LToeBase':     [3,225],
            'LToeBaseEnd':  [3,228],
            },
        
        "beat_141" : {
                'Spine':       3 ,
                'Neck':        3 ,
                'Neck1':       3 ,
                'RShoulder':   3 , 
                'RArm':        3 ,
                'RArm1':       3 ,
                'RHand':       3 ,    
                'RHandM1':     3 ,
                'RHandM2':     3 ,
                'RHandM3':     3 ,
                'RHandR':      3 ,
                'RHandR1':     3 ,
                'RHandR2':     3 ,
                'RHandR3':     3 ,
                'RHandP':      3 ,
                'RHandP1':     3 ,
                'RHandP2':     3 ,
                'RHandP3':     3 ,
                'RHandI':      3 ,
                'RHandI1':     3 ,
                'RHandI2':     3 ,
                'RHandI3':     3 ,
                'RHandT1':     3 ,
                'RHandT2':     3 ,
                'RHandT3':     3 ,
                'LShoulder':   3 , 
                'LArm':        3 ,
                'LArm1':       3 ,
                'LHand':       3 ,    
                'LHandM1':     3 ,
                'LHandM2':     3 ,
                'LHandM3':     3 ,
                'LHandR':      3 ,
                'LHandR1':     3 ,
                'LHandR2':     3 ,
                'LHandR3':     3 ,
                'LHandP':      3 ,
                'LHandP1':     3 ,
                'LHandP2':     3 ,
                'LHandP3':     3 ,
                'LHandI':      3 ,
                'LHandI1':     3 ,
                'LHandI2':     3 ,
                'LHandI3':     3 ,
                'LHandT1':     3 ,
                'LHandT2':     3 ,
                'LHandT3':     3 ,
            },
        
        "beat_27" : {
                'Spine':       3 ,
                'Neck':        3 ,
                'Neck1':       3 ,
                'RShoulder':   3 , 
                'RArm':        3 ,
                'RArm1':       3 ,
                'LShoulder':   3 , 
                'LArm':        3 ,
                'LArm1':       3 ,     
            },
    }

    #calculate mean and build cache for data. 
    target_fps = 15
    ori_list = joint_list["beat_joints"]
    target_list = joint_list["beat_141"]
    ori_data_path = "S:/audio2pose/datasets/beat_english_v0.2.1/"
    #wave cache from a = librosa.load(sr=16000) and np.save(a)
    ori_data_path_npy = "S:/audio2pose/datasets/beat_english_v0.2.1/"
    ori_data_path_ann = "S:/audio2pose/datasets/beat_english_v0.2.1/"
    cache_path = f"S:/audio2pose/datasets/beat_cache/beat_4english_{target_fps}_141/"
    reduce_factor_json = int(60/target_fps)
    reduce_factor_bvh = int(120/target_fps)
    print(f"target_fps: {target_fps}, reduce json {reduce_factor_json}, reduce bvh {reduce_factor_bvh}")
    speakers = sorted(os.listdir(ori_data_path),key=str,)

    npy_s_k = []

    load_type = "train"
    if not os.path.exists(f"{cache_path}"): 
        os.mkdir(cache_path)
    if not os.path.exists(f"{cache_path}{load_type}/"): 
        os.mkdir(f"{cache_path}{load_type}/")
        os.mkdir(f"{cache_path}{load_type}/wave16k/")
        os.mkdir(f"{cache_path}{load_type}/bvh_rot/")
        os.mkdir(f"{cache_path}{load_type}/bvh_full/")
        os.mkdir(f"{cache_path}{load_type}/bvh_rot_vis/")
        os.mkdir(f"{cache_path}{load_type}/facial52/")
        os.mkdir(f"{cache_path}{load_type}/text/")
        os.mkdir(f"{cache_path}{load_type}/emo/")
        os.mkdir(f"{cache_path}{load_type}/sem/")     

    for speaker in [1]:#replace to 1, 31 for all speakers
        all_data = [file for file in os.listdir(ori_data_path_npy+str(speaker)) if file.endswith(".npy")]
        npy_all = []
        for ii, file in enumerate(tqdm(all_data)):
            file = file[:-4]
            shutil.copy(f"{ori_data_path_npy}/{file.split('_')[0]}/{file}.npy", f"{cache_path}{load_type}/wave16k/{file}.npy")
            try:shutil.copy(f"{ori_data_path}/{file.split('_')[0]}/{file}.TextGrid", f"{cache_path}{load_type}/text/{file}.TextGrid")
            except: print(f"{file}.TextGrid")
            try: shutil.copy(f"{ori_data_path_ann}/{file.split('_')[0]}/{file}.txt", f"{cache_path}{load_type}/sem/{file}.txt")
            except: print(f"{file}.txt")
            try: shutil.copy(f"{ori_data_path_ann}/{file.split('_')[0]}/{file}.csv", f"{cache_path}{load_type}/emo/{file}.csv")
            except: print(f"{file}.csv")
            npy_all.extend(list(np.load(f"{ori_data_path_npy}/{file.split('_')[0]}/{file}.npy")))

        npy_all = np.array(npy_all)
        npy_s_k.append([len(npy_all)/16000/60])

    np.save(f"{cache_path}{load_type}/wave16k/npy_s_k.npy", npy_s_k)


    # calculate global mean and std for average all speakers
    npy_s_v = []
    json_s_v = []
    bvh_s_v = []

    npy_path = f"{cache_path}{load_type}/wave16k/"
    bvh_path = f"{cache_path}{load_type}/bvh_rot/"
    json_path = f"{cache_path}{load_type}/facial52/" 
            
    for i in [1]:
        npy_s_v.append([np.load(f"{npy_path}npy_mean_{i}.npy"), np.load(f"{npy_path}npy_std_{i}.npy")])
        json_s_v.append([np.load(f"{json_path}json_mean_{i}.npy"), np.load(f"{json_path}json_std_{i}.npy")])
        bvh_s_v.append([np.load(f"{bvh_path}bvh_mean_{i}.npy"), np.load(f"{bvh_path}bvh_std_{i}.npy")])

    all_length = 0
    new_m = np.zeros_like(npy_s_v[0][0])
    new_s = np.zeros_like(npy_s_v[0][0])
    for i, (m, s) in enumerate(npy_s_v):
        all_length += npy_s_k[i][0]
        new_m += npy_s_k[i][0] * m
    new_m /= all_length
    for i, (m, s) in enumerate(npy_s_v):
        new_s += ((s**2) + (m-new_m)**2) * npy_s_k[i][0]
    new_s /= all_length
    new_s = np.sqrt(new_s)
    print(f" audio mean: {new_m}, std: {new_s}") 
    np.save(f"{npy_path}npy_mean.npy", new_m)
    np.save(f"{npy_path}/npy_std.npy", new_s)  

    new_m = np.zeros_like(json_s_v[0][0])
    new_s = np.zeros_like(json_s_v[0][0])
    all_length = 0
    for i, (m, s) in enumerate(json_s_v):
        all_length += npy_s_k[i][0]
        new_m += npy_s_k[i][0] * m
    new_m /= all_length
    for i, (m, s) in enumerate(json_s_v):
        new_s += ((s**2) + (m-new_m)**2) * npy_s_k[i][0]
    new_s /= all_length
    new_s = np.sqrt(new_s)
    print(f" json mean: {new_m}, std: {new_s}") 
    np.save(f"{json_path}json_mean.npy", new_m)
    np.save(f"{json_path}/json_std.npy", new_s)

    new_m = np.zeros_like(bvh_s_v[0][0])
    new_s = np.zeros_like(bvh_s_v[0][0])
    all_length = 0
    for i, (m, s) in enumerate(bvh_s_v):
        all_length += npy_s_k[i][0]
        new_m += npy_s_k[i][0] * m
    new_m /= all_length
    for i, (m, s) in enumerate(bvh_s_v):
        new_s += ((s**2) + (m-new_m)**2) * npy_s_k[i][0]
    new_s /= all_length
    new_s = np.sqrt(new_s)

    print(new_m.shape)
    print(new_s.shape)
    np.save(f"{bvh_path}bvh_mean.npy", new_m)
    np.save(f"{bvh_path}/bvh_std.npy", new_s)