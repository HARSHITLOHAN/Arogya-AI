# Arogya-AI
Your digital personal trainer

Arogya AI tracks your body movements throughout a workout. It will help correct for posture, keep track of your reps and make sure you get fit the right way.

Quick Start
In a new environment, run pip install -r requirments.txt

Deployment
To run your own version of DeepFit, use python3 integration_Module.py

Development
To get started with your own predictions, you can use ArogyaAI_algo.py

The algo should use the TFLite model packaged as ArogyaAI_algo_v3.tflite.

The input required is an array of size 36, which denotes the X coordinates and Y coordinates of 18 keypoints.

See Inference Notebook for an example.

Demo
https://github.com/user-attachments/assets/4b95e1c8-a941-494f-8913-3385e38adcbf


Methodology
pipeline

We have implemented a wrapper around the Pose Detection API from Google's AI framework, MediaPipe, to achieve three tasks:

1. Keypoint Detection
The pre-trained MediaPipe landmark model in use is a Convolutional Neural Network and is trained with an input layer of size [1,256,256,3], which is accepted as incoming video feed from the webcam (256 x 256 pixels’ RGB values). The result received from the output layer is of shape [33, 5]. It translates to the 33 keypoints being detected by the model. The 5 additional parameters in the output map to the X, Y, and Z coordinates in the image (or video frame) and two factors for visibility and presence.

2. Pose Classification
18 of the 33 detected keypoints are used to train a pose classifier. The model tells us what workout is being performed, which leads us to task 3.

3. Pose Correction
Once the workout has been identified by our algo model, we use these keypoints to calculate the angles between limbs and compare it against benchmarks to identify if the person has an appropritate posture for an exercise. Apart from posture correction, these keypoints are also used to count the number of reps correctly performed for the workout.

The following image shows the keypoints provided by the Pose Landmark Model from MediaPipe:

keypoints

Implementation
Dataset
We utilize the MMFit dataset to train a neural network to identify what exercise the person is performing.
MMFit dataset is a collection of inertial sensor data from smartphones, smartwatches and earbuds worn by participants while performing full-body workouts, and time-synchronised multi-viewpoint RGB-D video, with 2D and 3D pose estimates.
We make use of the 2D pose estimates present in MMFit.
The data is split into train, test, and valdiation sets.
Since the dataset is large, it is not part of the repo. It can be downloaded here.
The dataset originally contains around 1.1 million frames worth of data (~800 minutes of video), which was filtered out for training the model. We only retained the labeled frames and removed all the noise for our prototype. This left us with a total of 375,753 frames.
Input Normalization
Since we plan to work with a live video feed, input normalization becomes a crucial component of the architecture. The model should be agnostic about how far away a person is standing from the camera, the height of the person, or the camera angle. To counter all these variables, we use a technique outlined in the MATEC paper to normalize the keypoints around the center of gravity. For this, first, the length of the body is calculated using the distances between certain detected keypoints.
