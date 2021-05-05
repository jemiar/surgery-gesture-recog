# surgery-gesture-recog

This project uses a 2-stage solution to recognize gestures in a robot-assisted surgery. The 2 stages are:
1. Distinguish blocks of 10 frames if they are transition blocks between 2 consecutive gestures, or they are normal blocks inside a gesture. We use 3DCNN for this stage.
2. Classify gestures using Convolutional LSTM.

We use the JIGSAWS dataset for training and validation.

File description:
* data_generator.py: used to load data from folder and supply them to models
* transit_classification.py: used to classify a 10-frame block if it is a transition block or normal one
* gesture_classification.py: used to classify gestures
