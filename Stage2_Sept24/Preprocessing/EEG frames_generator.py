

import mne
import os
from os.path import abspath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import morlet
from tqdm import tqdm
import itertools
import cv2
import warnings
# import pywt
from matplotlib import cm
from multiprocessing import Manager
warnings.filterwarnings("ignore")
from multiprocessing import Process


from skimage.restoration import denoise_wavelet
from multiprocessing import Pool


import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import mne
import cv2
from multiprocessing import Pool


file_path="/media/kashraf/Elements/data_gen_may_2021/EEG data/Audio/"
filename=os.listdir(file_path)
path_montage="/media/kashraf/Elements/data_gen_may_2021/montage/"
montage=mne.channels.read_montage(path_montage+"//"+"neuroscan64ch.loc")
raw_data=[]
for file in tqdm(filename):
    files=mne.io.read_raw_cnt(file_path+"/"+file,montage=montage, preload=True,verbose=False);
    raw_data.append(files)

## Selecting channels to include
good_ch= mne.pick_channels(raw_data[0].info['ch_names'], include=[],
                        exclude=["EKG","EMG",'VEO','HEO','Trigger'])
mne.pick_info(raw_data[0].info,sel=good_ch,copy=False,verbose=False)

for f in tqdm(raw_data):
    mne.pick_info(f.info,sel=good_ch,copy=False)
    f.set_montage(montage)


# In[15]:


info = raw_data[0].info


# Define folder to label mapping
folder_label_map = {
    'cl2': 0,
    'cl4': 1,
    'cl6': 2,
    'cl8': 3
}

def generate_image(data):
    """Generate topomap from averaged data more efficiently."""
    # Generate the topomap as a numpy array without rendering the figure
    fig, ax = plt.subplots(figsize=(1.5, 1.5))  # Use smaller figure to speed up
    im, _ = mne.viz.plot_topomap(data, pos=info, contours=False, show=False, axes=ax)
    
    # Convert the figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.close(fig)  # Close the figure to save memory
    return img

class WTGenerator:
    def __init__(self, freqs, sfreq=250, n_cycles=None):
        self.freqs = freqs
        self.sfreq = sfreq
        self.n_cycles = n_cycles

    def get_wt_array(self, data):
        """Extract wavelet transform of the epoch data."""
        data = np.reshape(data, (1, 64, 176))
        epoch_data = mne.EpochsArray(data, info=info)
        wavelets = mne.time_frequency.tfr_array_morlet(epoch_data, sfreq=self.sfreq, freqs=self.freqs,
                                                       n_cycles=self.n_cycles, n_jobs=12, output="power")
        return wavelets

    def serialize_example(self, frames, label):
        """Create a tf.train.Example message ready to be written to a file."""
        feature = {
            'frames': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frames.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def process_trial(self, trial_file, folder, output_path, label):
        """Process a single trial and save as TFRecord."""
        trial_data = np.load(trial_file)  # Load the EEG data (n_channels x n_times)

        # Extract wavelet transform and average across frequencies
        wt_data = np.mean(self.get_wt_array(trial_data), axis=2)  # Averaging across frequencies
        wt_data = wt_data[:, :, :176]  # Restrict to 176 time samples

        num_frames = 16  # Number of frames to generate
        frame_size = wt_data.shape[2] // num_frames  # Determine the window size for each frame

        frames = []
        for j in range(num_frames):
            start_idx = j * frame_size
            end_idx = (j + 1) * frame_size if j < num_frames - 1 else wt_data.shape[2]
            window_data = np.mean(wt_data[:, :, start_idx:end_idx], axis=2)[0]

            img = generate_image(window_data)  # Generate topomap image for the averaged window
            frames.append(img)

        frames = np.array(frames)

        # Save as TFRecord
        tfrecord_path = os.path.join(output_path, f"{folder}_trial_{os.path.splitext(os.path.basename(trial_file))[0]}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            example = self.serialize_example(frames, label)
            writer.write(example)

    def generate_3D_data_tfrecord(self, base_path, output_path):
        """Generate and save the 3D data as TFRecord, using folder names for labels."""
        all_tasks = []
        
        # Collect all tasks to process in parallel
        for folder, label in folder_label_map.items():
            folder_path = os.path.join(base_path, folder)

            # Loop through each trial file in the folder
            trial_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
            for trial_file in trial_files:
                all_tasks.append((trial_file, folder, output_path, label))

        # Process the trials in parallel using multiprocessing
        with Pool(processes=8) as pool:  # Adjust the number of processes as per your system
            pool.starmap(self.process_trial, all_tasks)

# Example usage
freqs = np.arange(13, 30)  # Beta range (13-30 Hz)
sfreq = 250  # Sampling frequency
n_cycles = np.linspace(3, 7, len(freqs))  # Number of cycles for wavelet transform


# Path to the folder containing 'cl2', 'cl4', 'cl6', 'cl8'
base_path = "/media/kashraf/TOSHIBA EXT/Dissertation/stage2/audio/ERP"

# Path to save individual TFRecord files for each trialc
output_path = "/media/kashraf/TOSHIBA EXT/Dissertation/stage2/audio/Topomap_movie/beta"

# Create the WT generator instance
wt_gen = WTGenerator(freqs=freqs, sfreq=sfreq, n_cycles=n_cycles)

# Generate TFRecords for all trials using folder labels
wt_gen.generate_3D_data_tfrecord(base_path=base_path, output_path=output_path)

