# DataGenerator class
# Sublass of keras' Sequence class
# Role: supply data to model during training and validation

import tensorflow as tf
import numpy as np

base_directory = './'

# This class is responsible for data preparation.  It loads video segments from the video folder.  Each segment consists
# 10 frames.  The segment are batched in groups of 8 segments.  The segments are fed to the model.
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=8, dim=(10, 240, 320), n_channels=3, n_classes=2, folder='data/', shuffle=True):
        self.dim = dim                      # dimension of each segment (typically 10 frames)
        self.batch_size = batch_size        # the number of segments in each batch (typically 8, but changeable)
        self.labels = labels                # the labels from a dictionary of segment IDs and segment classes (transition and normal)
        self.list_IDs = list_IDs            # the list of the segment IDs
        self.n_channels = n_channels        # the number of RGB channels (3 in this case)
        self.n_classes = n_classes          # two classes (normal and transition)
        self.folder = folder                # output folder where data will be logged
        self.shuffle = shuffle              # boolean indicating whether to shuffle the list of IDs (associating IDs with different batches)
        self.on_epoch_end()                 # callback executed after each iteration of training (epoch)

    # Calculate how many batches during training/validation
    # len = number of data segments / batch size
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
        self.indexes = np.arange(len(self.list_IDs))        # create list of indices
        if self.shuffle:                                    # shuffle indices if needed
            np.random.shuffle(self.indexes)

    # Generate data from list of ids
    def data_generation(self, list_IDs_temp):
        # Initialize numpy array
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # X is one input batch
        y = np.empty((self.batch_size), dtype=int)                  # y is the target (label) of each batch)

        # For each id, load data from folder
        for i, ID in enumerate(list_IDs_temp):
            # load batch from base folder and file
            X[i,] = np.load(base_directory + self.folder + ID + '.npy')
            # Get ground-truth labels obtained from constructor
            y[i] = self.labels[ID]

        return X, y
