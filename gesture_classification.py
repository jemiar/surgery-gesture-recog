import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from data_generator import DataGenerator

base_directory = './'

# function to save gesture video data
def save_gesture(fromarray=transcriptions, height=240, width=320, folder='data_action/', idnumber=1):
    # array to store ids of gesture
    data = []
    # dictionary to store target labels of gestures
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

        for t in arr['transcription']:
            # Save gesture
            # Calculate the left most frame of 1 gesture
            left = (t[0] + 1) // 3
            # Calculate the right most frame of 1 gesture
            right = (t[1] - 1) // 3 + 1
            # Get the target value of gesture class
            c = t[2]
            block = np.expand_dims(red_frames[left:right,:,:], axis=3)
            block = np.append(block, np.expand_dims(green_frames[left:right,:,:], axis=3), axis=3)
            block = np.append(block, np.expand_dims(blue_frames[left:right,:,:], axis=3), axis=3)
            # Store gesture
            npy_name = 'id_' + str(idnumber)
            temp_obj = {'id': npy_name, 'file': arr['file'], 'gesture': c}
            data.append(temp_obj)
            labels[npy_name] = c
            np.save(base_directory + folder + npy_name + '.npy', block)
            idnumber += 1

    return data, labels

# function to create Convolutional LSTM model to classify gesture
def create_model(height=240, width=320):
    # shape of input: variable number frames x height x width x 3 channels (RGB)
    input = tf.keras.Input((None, height, width, 3))

    # 1st ConvLSTM block includes Conv2D with 8 filters, MaxPool2D and BatchNormalization
    # Using keras TimeDistributed layer to apply Conv2D, MaxPool2D and BatchNormalization to each frame
    conv2D_1 = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu')
    x = layers.TimeDistributed(conv2D_1)(input)
    maxpool_1 = layers.MaxPool2D(pool_size=(2,2))
    x = layers.TimeDistributed(maxpool_1)(x)
    batchnorm_1 = layers.BatchNormalization()
    x = layers.TimeDistributed(batchnorm_1)(x)

    # 2nd ConvLSTM block includes Conv2D with 16 filters, MaxPool2D and BatchNormalization
    # Using keras TimeDistributed layer to apply Conv2D, MaxPool2D and BatchNormalization to each frame
    conv2D_2 = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')
    x = layers.TimeDistributed(conv2D_2)(x)
    maxpool_2 = layers.MaxPool2D(pool_size=(2,2))
    x = layers.TimeDistributed(maxpool_2)(x)
    batchnorm_2 = layers.BatchNormalization()
    x = layers.TimeDistributed(batchnorm_2)(x)

    # Flatten x to supply it to LSTM layer
    flatten = layers.Flatten()
    x = layers.TimeDistributed(flatten)(x)

    # LSTM layer with 64 output units at the last LSTM block
    x = layers.LSTM(64, return_sequences=False)(x)
    # Fully-connected layer with 32 units and L2 regularization
    x = layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.01))(x)

    # output shape (10,)
    output = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(input, output, name='ConvLSTM')
    return model

# Create model
model = create_model(240, 320)

# Create data generator for training and validation
params = {
    'dim': (10, 240, 320),
    'batch_size': 1,
    'n_classes': 10,
    'n_channels': 3,
    'folder': 'data_action/',
    'shuffle': True
}
train_generator = DataGenerator(training_ids, labels, **params)
val_generator = DataGenerator(validation_ids, labels, **params)

learning_rate = 0.001
metrics = [keras.metrics.CategoricalAccuracy()]
# Compile model, using categorical cross-entropy for loss, and stochastic gradient descent
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.SGD(learning_rate=learning_rate), metrics=metrics)
# Train model in 100 epochs
model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=100, shuffle=True)
