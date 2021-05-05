import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from data_generator import DataGenerator

base_directory = './'

# function used to read video data and save normal or transit blocks to folder
def save_data(fromarray=transcriptions, height=240, width=320, folder='data_001/', idnumber=1):
    # array to store ids of normal blocks
    normals = []
    # array to store ids of transit blocks
    transits = []
    # dictionary to store target y value (class) of each block
    labels = {}
    os.chdir(base_directory + 'Suturing/video/')
    # for each element in fromarray (store video file names)
    for arr in fromarray:
        # use CV2 to capture the video file
        cap = cv2.VideoCapture(arr['file'][:-4] + '_capture1.avi')
        i = 1
        # Initialize numpy array to store frames of Red, Green, Blue channels of video
        red_frames = np.empty((0, height, width))
        green_frames = np.empty((0, height, width))
        blue_frames = np.empty((0, height, width))
        # while reading the capture, only store 1 frame for every 3 frames
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            if i%3 == 1:
                # Resize the frame to reduce the computation during training
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # Cast the frame as a numpy array
                f = np.asarray(frame)
                # Update the color of frame from BGR to RGB
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                # Apprend frame to its appropriate channel
                red_frames = np.append(red_frames, np.expand_dims(f[:,:,0], axis=0), axis=0)
                green_frames = np.append(green_frames, np.expand_dims(f[:,:,1], axis=0), axis=0)
                blue_frames = np.append(blue_frames, np.expand_dims(f[:,:,2], axis=0), axis=0)
            i += 1
        # Release the capture when finishing reading
        cap.release()
        # Normalize the value of each element to range [0, 1]
        red_frames = red_frames / 255.0
        green_frames = green_frames / 255.0
        blue_frames = blue_frames / 255.0

        # For each transciption (transcribe where each gesture starts and ends)
        for k, t in enumerate(arr['transcription']):
            # Save the normal block
            # Calculate the left most frame of 1 gesture
            left = (t[0] + 1) // 3
            # Calculate the right most frame of 1 gesture
            right = (t[1] - 1) // 3
            # Calculate the number of normal blocks in a gesture
            num_blocks = (right - left + 1) // 10
            for index in range(num_blocks):
                # Each block has shape (10, height, width, 3)
                block = np.expand_dims(red_frames[left+index*10:left+(index+1)*10,:,:], axis=3)
                block = np.append(block, np.expand_dims(green_frames[left+index*10:left+(index+1)*10,:,:], axis=3), axis=3)
                block = np.append(block, np.expand_dims(blue_frames[left+index*10:left+(index+1)*10,:,:], axis=3), axis=3)
                # Store normal block
                npy_name = 'id_' + str(idnumber)
                temp_obj = {'id': npy_name, 'file': arr['file'], 'label': 0}
                normals.append(temp_obj)
                labels[npy_name] = 0
                np.save(base_directory + folder + npy_name + '.npy', block)
                idnumber += 1

            # Save transit blocks
            if k < (len(arr['transcription']) - 1):
                # Each transit block has the last 5 frames of 1 gesture and the 1st 5 frames of the next gesture
                # Calculate the left most frame of a transit block
                ind = (t[1] - 1) // 3 - 4
                block = np.expand_dims(red_frames[ind:ind+10,:,:], axis=3)
                block = np.append(block, np.expand_dims(green_frames[ind:ind+10,:,:], axis=3), axis=3)
                block = np.append(block, np.expand_dims(blue_frames[ind:ind+10,:,:], axis=3), axis=3)
                # Store transit block
                npy_name = 'id_' + str(idnumber)
                temp_obj = {'id': npy_name, 'file': arr['file'], 'label': 1}
                transits.append(temp_obj)
                labels[npy_name] = 1
                np.save(base_directory + folder + npy_name + '.npy', block)
                idnumber += 1

    return normals, transits, labels

# function to create 3D CNN model to classify normal and transit blocks
def create_model(height=240, width=320):
    # shape of input: 1 block has 10 frames x height x width x 3 channels (RGB)
    input = tf.keras.Input((10, height, width, 3))

    # 1st Conv3D block includes Conv3D with 8 filters, MaxPool3D and BatchNormalization
    x = layers.Conv3D(filters=8, kernel_size=(3,3,3), activation='relu')(input)
    x = layers.MaxPool3D(pool_size=(2,2,2))(x)
    x = layers.BatchNormalization()(x)

    # 2nd Conv3D block includes Conv3D with 16 filters, MaxPool3D and BatchNormalization
    x = layers.Conv3D(filters=16, kernel_size=(3,3,3), activation='relu')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2))(x)
    x = layers.BatchNormalization()(x)

    # 3rd Conv3D block includes Conv3D with 32 filters, MaxPool3D and BatchNormalization
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(input)
    x = layers.MaxPool3D(pool_size=(1,2,2))(x)
    x = layers.BatchNormalization()(x)

    # Fully-connected block includes GlobalAveragePooling3D, Fully-Connected layer with 512 units and DropOut for Regularization
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.DropOut(0.7)(x)

    # output shape (1,) produces value between [0, 1]
    output = layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.Model(input, output, name='3DCNN')
    return model

# Create model
model = create_model(240, 320)

# Create data generator for training and validation
params = {
    'dim': (10, 240, 320),
    'batch_size': 16,
    'n_classes': 2,
    'n_channels': 3,
    'folder': 'data_001/',
    'shuffle': True
}
train_generator = DataGenerator(training_ids, labels, **params)
val_generator = DataGenerator(validation_ids, labels, **params)

learning_rate = 0.001
metrics = [keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall()]
# Compile model, using binary cross-entropy for loss
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
# Train model in 100 epochs
model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=100, shuffle=True)
