**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890 "CNN Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

The model that was deployed for the behavioral cloning project was adopted from the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that was similarly used for their end-to-end driving test. In the paper they conclude that their tests have 'empirically demonstrated that CNNs are able to learn the entire task of lane and road following without manual decomposition'. Given the thorough research already conducted, the paper had proved to be a good starting point as a design basis. 

The model is a deep convolution neural network that is constructed from a normalization layer, followed by five convolutional layers, and three fully connected layers. Convolution layers have shown that they perform hierachical feature distinction when it comes to image recognition and after a few layers when a general feature simulacrum is constructed it can then pass the input towards the fully connected layers to perform parameter classification. The architecture designed and used in model.py is visually illustrated in the diagram below.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1   |
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1         |
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |convolution2d_5  |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dropout_1 (Dropout)             |(None, 50)        |0       |dense_2          |
|dense_3 (Dense)                 |(None, 10)        |510     |dropout_1        |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |252219  |                 |

### 2. Attempts to reduce overfitting in the model
I added a dropout layer with a keep probability of 0.50 after the second fully connected layer of 50 nodes to minimize overfitting. (model.py line 42)
The model deploys RELU activations to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18) to avoid saturation in vision perception and to minimize error in gradient descent. 


The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, which uses the running averages of gradients otherwise known as momentum to allow us to converge to minimization. The training was performed on a gaming laptop with GTX 960M graphics card so the computations were manageable on the hardware. (model.py line 25).

### 4. Training Data Selected

I had found that testing the model with the sample training data provided in the project resources was a way to eliminate variance in validating the accuracy of my model. Therefore the data that was used was from the sample data, and so the strategy used to account for recovery and sharp turn scenarios was in the emphasis of data augmenetation.

## Data Preprocessing

### 1. Image Augumentation

For training, I used certain augmentation techniques along with a batch generator to theoretically generate an infinite number of images:

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

- Randomly choose right, left or center images.
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- Randomly flip image left/right and adjust to negative steering angle
- Randomly translate image horizontally with steering angle adjustment (0.002 change in angle per pixel shift)
- Randomly translate image vertically
- Randomly altering image brightness (lighter or darker)

Using both the left and right images is helpful as they represent samples for how to recover back to center lane driving. The flipped images remove a left turning bias in the track.The horizontal translation is particulary useful in the sharp turns or curves as they provide more extreme representations of how to handle scenarios of large changing deltas in direction. 

### Examples of Augmented Images

The following are examples of transformations performed on the images

![](examples/center.jpg?raw=true)

**Center Image**

![](examples/left.jpg?raw=true)

**Left Image**

![](examples/right.jpg?raw=true)

**Right Image**

|**Original Image**|**Flipped Image**|
|------------------|-----------------|
|![](examples/center.jpg?raw=true)|![](examples/flip.jpg?raw=true)|

|**Original Image**|**Translated Image**|
|------------------|-----------------|
|![](examples/center.jpg?raw=true)|![](examples/translate.jpg?raw=true)|

|**Original Image**|**Increased Brightness Image**|
|------------------|-----------------|
|![](examples/center.jpg?raw=true)|![](examples/bright.jpg?raw=true)|


## Training, Validation and Testing

- Since this was a regression network that aimed at optimizing steering angle for a given image sequence I used mean squared error as my loss function
- I used the Adam optimizer for the applied momentum technique with learning rate of 1.0e-4
- I used ModelCheckpoint from Keras to save the model after an epoch if the validation loss was lower than the last recorded minimum
- It was found that validation accuracy did not continue to improve after 8 training epochs.

## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
- (Assistance) How to Simulate a Self-Driving Car: https://www.youtube.com/watch?v=EaY5QiZwSP4
