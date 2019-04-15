## Writeup for Advanced Lane Finding

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

**I organized all my codes in the notebook "Advanced Lane Finding.ipynb", which has the same structure and outline as this writeup. So, you can easily find the counterpart python code in the notebook.**

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in 'Camera Calibration' part of the notebook. You can get the core function `calibrate_camera` in 'restruct the codes' part.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the cheeseboard images using the `cv2.undistort()` function and obtained this result: 

![test image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/camera%20calibration.png]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![test image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/test1.jpg]

The corresponding undistorted image is:
![undist image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/undist1.jpg]

> Can you see the change of the rear part of the white car？Obviously，right？

One more example. The original image is:
![test image 2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/test2.jpg]

The corresponding undistorted image is:
![undist image 2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/undist2.jpg]
 
> Notice the bushes on the top-right corner of the image to get the undistortion effect.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This part is the key to the lane detection. I tried HLS color space, x and y gradient, magnitude and direction gradient, and their combination. I tested different parameter values, and finally got the best result.

Here's an example of my output for this step. The original image:
![test image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/test1.jpg]

the x-gradient binary:
![grad_x image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/grad_x.jpg]

the y-gradient binary:
![grad_y image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/grad_y.jpg]

the maginitude-gradient binary:
![mag image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/mag.jpg]

the direction-gradient binary:
![dir image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/dir.jpg]

the hls s-channel binary:
![hls image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/hls.jpg]

And their combination result:
![cmb image 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/test_images/cmb1.jpg]

> The s-channel of HLS colorspace does a great contribution to detect the lanes. And the combination method works well.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The function `warp()` acted the perspective transform, which    took an image (`img`), source (`src`) and destination (`dst`) points as inputs.  I hardcoded the source and destination points as following:

```python
src_coordinates = np.float32(
    [[280,  700],  # Bottom left
     [595,  460],  # Top left
     [725,  460],  # Top right
     [1125, 700]]) # Bottom right

dst_coordinates = np.float32(
    [[250,  720],  # Bottom left
     [250,    0],  # Top left
     [1065,   0],  # Top right
     [1065, 720]]) # Bottom right   
```

It worked as expected: 

![warp][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/warped.png]

And you can get a more clear understanding of this 'bird-eye' transform through the binary image:
![warp binary][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/warped_binary.png]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use histgram, windowing and 2nd order polynomial fitting to find the lane lines as following: 
* hist: to get histgram for detecting the peaks hinting the lanes.
* windowing: to use sliding windows to detect pixels for the lane lines.
* polynomial fitting: a 2nd-order polynomial fitting to fit out the smooth curve of the lane. 

There are two examples:
![example 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/lane_detected1.png]

![example 2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/lane_detected2.png]

And we can use `fit_similar_lines` to get a more accurate searching area by using the previous polynomial to skip the sliding window, like the following:

![example 1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/search1.png]

![example 2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/search2.png]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in two functions: `curvature_radius` and `car_offset` just following the instructs of the course. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used Minv to wrap the detected lines back to the original image, and show the final result. Here is an example of my result on a test image:

![detected1][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/detected1.jpg]

And another:
![detected2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/detected2.jpg]

And then add the curvature and offset info onto the image:
![detected1_info][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/detected1_info.jpg]

![detected2][https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/detected2_info.jpg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I encapsulated all the codes in the class `ProcessPipeline` for the video processing.

Here's a [link to my video result](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/project_video_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am very glad to finish this project, and this is a great challenge for me. 
 
I think there are still some issues should be considered to get a more fine process pipeline:

* the source and destination points of perspective transform were hardcoded, is there some methods to do this more clever?
* how to share more informations frame to frame to reduce the computation of finding the lane line?
* more colorspace should be tried to get a better binary method.

These are what I can find at this moment, and I will never stop thinking on this subjects.
