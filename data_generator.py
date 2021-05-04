# DataGenerator class
# Sublass of keras' Sequence class
# Role: supply data to model during training and validation

import tensorflow as tf
import numpy as np

base_directory = './'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=8, dim=(10, 240, 320), n_channels=3, n_classes=2, folder='data/', shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.folder = folder
        self.shuffle = shuffle
        self.on_epoch_end()

    # Calculate how many batches during training/validation
    # len = number of data samples / batch size
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Get data by batch index
    # Call data_generation method
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.data_generation(list_IDs_temp)
        return X, y

    # At the end of each epoch, generate batch indexes
    # And shuffle data if needed
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generate data from list of ids
    def data_generation(self, list_IDs_temp):
        # Initialize numpy array
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # For each id, load data from folder
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(base_directory + self.folder + ID + '.npy')
            y[i] = self.labels[ID]

        return X, y
