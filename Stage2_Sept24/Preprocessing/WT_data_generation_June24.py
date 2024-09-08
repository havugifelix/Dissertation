
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
file_path="/media/kashraf/Elements/data_gen_may_2021/EEG data/Visual/"
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


# In[2]:


data_path = "/media/kashraf/TOSHIBA EXT/Journal_2023/Data/Visual/NPY"
dest_path = "/media/kashraf/TOSHIBA EXT/Journal_2023/Data/Visual/Wavelet/Real/ERP8"
n_channels = 64
n_times = 176
sfreq = 250  # Sampling frequency

data = np.load (os.path.join(data_path,"Visual_ERP_6.npy"))


info = raw_data[0].info




from skimage.restoration import denoise_wavelet
from multiprocessing import Pool


class wt_generator:
    def __init__(self, data,freqs,sfreq =250,cl=None,n_cycles=None):
        self.data = data
        self.freqs =freqs
        self.sfreq = sfreq
        self.n_cycles = n_cycles
        self.cl=cl

    
    def get_wt_array (self):
        epoch_data =mne.EpochsArray(self.data,info=info)
        # EXtract epoch data and get wt 
        wavelets = mne.time_frequency.tfr_array_morlet(epoch_data,sfreq=self.sfreq,freqs=self.freqs,
                                                       n_cycles=self.n_cycles,n_jobs=12,output="power")

        
        return wavelets 
    
    def generate_3D_data(self,path,sample_index):
        wt_data = np.mean (self.get_wt_array(),axis =2)
        wt_data = wt_data[:,:,:176]
#         print(wt_data.shape)
#         all_imgs = np.empty((wt_data.shape[0], 176,224, 224, 3), dtype=np.uint8)
        
        # Loop through each trial and channel
        for i in tqdm(range(wt_data.shape[0])):
            # Transpose data
            data = np.transpose(wt_data[i], (1, 0))

            # Generate images in parallel
            with Pool() as pool:

                #imgs = pool.map(generate_image, (data[j] for j in range(data.shape[0])))
                np.save(os.path.join(path,self.cl,"3D_WT_data_"+"trial_"+str(sample_index)), data)


    
path ="/media/kashraf/TOSHIBA EXT/Dissertation/stage2/wavelet/Mean_power_wt/Beta/full"
freqs = np.arange (13,30)
sfreq = 250
n_times = 176
pad_length = sfreq-n_times
n_cycles = np.linspace(3, 7, len(freqs))

for i, data in enumerate(tqdm(data)):
    data = np.expand_dims(data,axis=0)
#     print(data_padded.shape)
    wave_gen = wt_generator(data,freqs=freqs,cl="cl6",n_cycles=n_cycles)
    wave_gen.generate_3D_data(path=path,sample_index=i)

