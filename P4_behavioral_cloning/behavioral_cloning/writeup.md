# **Behavioral Cloning** 

## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* ` model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network, which is defined in the `nn.py` file. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. And `utils.py` contains the functions to preprocess the images.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 53 (model.py lines 9-13) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, I used several data augmentation techniques like:
*   flipping images horizontally 
*   using left and right images 
to help the model generalize. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an `Adam` optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Udacity sample data was used for training.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed the [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture to get my own Behavioral Cloning model because the Nvidia model has been proven successfully in self-driving. It is referred widely and recommended in the class.

In order to measure how well the model works, I split my image and steering angle data into a training and validation set. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

The hardest part was getting the data augmentation to work. I had a working solution early on but because there were a lot of syntax errors and minor omissions, it took a while to piece everything together. One of the problems, for example, was that I was incorrectly applying the data augmentation techniques and until I created a visualization of what the augmented images looked like, I was not able to train a model that drives successfully around the track.

#### 2. Final Model Architecture

The final model architecture can be shown clearly by its code below: 

```python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout

def model(loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model
```


Here is a visualization of the architecture 

![nn](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/images/nvdia.png)

#### 3. Creation of the Training Set & Training Process

To create the training data, I used the Udacity sample data as a base. For each image, normalization would be applied before the image was fed into the network. In my case, a training sample consisted of four images:
*   Center camera image
*   Center camera image flipped horizontally
*   Left camera image
*   Right camera image

Here are some sample images of raw and preprocessed images:
**Raw**
![raw](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/images/nvdia.png)

**RGB**
![rgb](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/images/rgb.png)

**Cropped**
![cropped](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/images/cropped.png)

**Resized**
![resized](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/images/resized.png)

I found that this was sufficient to meet the project requirements for the first track. Because I am running out my term time soon, so I just stop here.

To make the car run on the same track, additional data augmenation techniques like adding random brightness, shearing and horizontal shifting could be applied.

The provided sample data set (minus the 20% validation set) has 27696 images when the data is augemented. The network was then trained for 2 epochs in the workspace for approximately 4 hours.

The model was then tested on the track to ensure that the model was performing as expected. You can get the final effect via the [video](https://viewhjeftibwrx.udacity-student-workspaces.com/files/home/workspace/CarND-Behavioral-Cloning/video.mp4).