# Traffic Sign Recognition

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and matplot methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
    * 34799
* The size of the validation set is ?
    * 4410
* The size of test set is ?
    * 12630
* The shape of a traffic sign image is ?
    * (32, 32, 3)
* The number of unique classes/labels in the data set is ?
    * 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
First, it shows some samples of the traffic signs and their names, giving some intuitions of the data set:

![traffic sign samples](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/results/traffic%20sign%20samples.png)

And then a histogram shows the distribution of the labels over the training set:
![hist](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/results/hist.png)

At last, I wrote a function `sign, sign_samples=N` to show N samples of the specific `sign`. The following shows 5 samples of 'Road work' sign:
![samples](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/results/samples.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not convert the RGB images into grayscale because color plays an important part in traffic sign.

I just scale the values into [-1, 1] range using `scale` function to normalize the data.

Then I fed the data into the `LeNet` DNN to train.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5    | 1x1 stride, VALID padding, outputs 10x10x16    									|
| RELU					|		|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16 |
| Flat	      	| outputs = 400|
| Fully connected		| shape=(400, 300), Output = 300 |
| RELU					|		|
| Fully connected		| shape=(300, 200), Output = 200 |
| RELU					|		|
| Fully connected		| shape=(200, 120), Output = 120 |
| RELU					|		|
| Fully connected		| shape=(120, 84), Output = 84 |
| RELU					|		|
| Dropout					|	0.8|
| Fully connected		| shape=(84, 60), Output = 60 |
| RELU					|		|
| Dropout					|	0.8|
| Fully connected		| shape=(60, 43), Output = 43 |
| Softmax				| |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer and the following super parameters:
* batch size: 64
* epochs: 50
* learning rate: 0.001
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.8

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.950
* test set accuracy of 0.936

Because I was running out of my team time, I just tried the `LeNet` method.
 

### Test a Model on New Images

#### 1. Choose 14 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 14 German traffic signs that I found on the [web](https://en.wikipedia.org/wiki/Road_signs_in_Germany):

![tested](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/results/tested.png)

The image quality of these pictures are really good. It should not be difficult to recognize. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

[(24, '24'), (30, '30'), (13, '13'), (28, '28'), (12, '12'), (38, '38'), (26, '26'), (18, '18'), (31, '31'), (23, '23'), (14, '14'), (22, '22'), (29, '28'), (19, '19')]

The model was able to correctly guess 13 of the 14 traffic signs, which gives an accuracy of 92.9%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the `Output Top 5 Softmax Probabilities For Each Image Found on the Web` part of the IPython notebook.

The top five soft max probabilities were:

```text
TopKV2(values=array([[  5.04024148e-01,   4.35211450e-01,   4.11281623e-02,
          1.46519514e-02,   2.56993435e-03],
       [  9.99912143e-01,   6.14749151e-05,   1.04697128e-05,
          7.90067043e-06,   7.20746539e-06],
       [  1.00000000e+00,   3.07032895e-29,   3.03393449e-36,
          1.83790109e-36,   1.65790435e-37],
       [  9.77957785e-01,   6.50603464e-03,   6.30431343e-03,
          2.44352361e-03,   2.06547114e-03],
       [  1.00000000e+00,   3.94303220e-19,   4.00153500e-20,
          3.15242267e-21,   8.00325174e-25],
       [  1.00000000e+00,   6.11365906e-12,   5.68691708e-12,
          2.57626863e-12,   8.49433588e-13],
       [  1.00000000e+00,   6.11425965e-10,   4.55693667e-16,
          7.40851908e-18,   9.77467232e-19],
       [  1.00000000e+00,   1.22924249e-11,   1.68240822e-21,
          2.37083777e-22,   4.24540249e-27],
       [  9.99603331e-01,   3.63874628e-04,   3.26981899e-05,
          2.13391260e-08,   1.59666058e-08],
       [  9.99999523e-01,   4.22713413e-07,   2.11468065e-09,
          1.16407262e-09,   2.52290133e-10],
       [  9.99999881e-01,   7.79136684e-08,   2.00467500e-08,
          2.95090063e-09,   2.41057840e-09],
       [  9.99995828e-01,   3.98995053e-06,   2.87903987e-07,
          5.95677641e-09,   4.75107687e-09],
       [  6.09381914e-01,   2.04652414e-01,   1.22916244e-01,
          3.75010073e-02,   2.32977346e-02],
       [  1.00000000e+00,   3.84757305e-15,   1.56210494e-21,
          1.53160399e-22,   1.07721299e-26]], dtype=float32), indices=array([[24, 31,  1,  0, 18],
       [30, 11, 20, 25, 26],
       [13, 39, 12,  9,  1],
       [28, 41, 20,  0, 32],
       [12, 25,  9, 38, 13],
       [38,  9, 13, 34, 36],
       [26, 18, 20, 22, 25],
       [18, 26, 38, 23, 20],
       [31, 25, 19, 23, 21],
       [23, 20, 18, 29, 19],
       [14, 26, 15, 29,  4],
       [22, 18, 25, 26, 34],
       [28, 23, 29, 36,  3],
       [19, 27, 23, 31, 35]], dtype=int32))
```
For example, in the sign 24 'Road narrows on the right' recognization processing, the `LeNet` is not so sure about it. It gives similar softmax probabilities on 24 and 31 signs. 

But for sign 30 'Beware of ice/snow', it gives a 100% probability!

