
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class NpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, npy_dir, batch_size, shuffle=True):
        self.npy_dir = npy_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get a list of all subdirectories in npy_dir
        self.classes = sorted(os.listdir(npy_dir))

        # Create a mapping from npy file paths to their corresponding labels
        self.file_to_label = {}
        for i, label in enumerate(self.classes):
            label_dir = os.path.join(npy_dir, label)
            npy_files = os.listdir(label_dir)
            for npy_file in npy_files:
                self.file_to_label[os.path.join(label_dir, npy_file)] = i

        self.npy_files = list(self.file_to_label.keys())
        self.num_samples = len(self.npy_files)
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_npy_files = [self.npy_files[k] for k in batch_indices]
        batch_x = []
        batch_y = []
        for npy_file in batch_npy_files:
            npy_data = np.load(npy_file)
            x = np.reshape(npy_data, (176, 64)) 
            y = self.file_to_label[npy_file]
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.array(batch_x)
        min_val = batch_x.min()
        max_val = batch_x.max()
        batch_x = (batch_x-min_val)/(max_val-min_val)
        batch_y = np.array(batch_y)
        batch_y_true = to_categorical(batch_y, num_classes=len(self.classes))
        return batch_x, batch_y_true, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

