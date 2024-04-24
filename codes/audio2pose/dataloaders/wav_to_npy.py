import os
import librosa
import numpy as np

# Specify the base directory of your audio files
base_dir = "/tsc003-beat-vol/audio2pose/datasets/beat_english_v0.2.1" #"S:\\audio2pose\\datasets\\beat_english_v0.2.1\\"

# Iterate over each subdirectory in the base directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)    
    # Check if it's indeed a directory
    if os.path.isdir(subdir_path):
        # List all WAV files in this subdirectory
        for file in os.listdir(subdir_path):
            if file.endswith(".wav"):
                file_path = os.path.join(subdir_path, file)
                
                # Load the audio file with librosa
                audio, sr = librosa.load(file_path, sr=16000)
                
                # Prepare the .npy filename
                npy_filename = os.path.splitext(file)[0] + ".npy"
                npy_path = os.path.join(subdir_path, npy_filename)
                
                # Save the audio data as a .npy file
                np.save(npy_path, audio)
                #print(f"Saved {npy_path}")