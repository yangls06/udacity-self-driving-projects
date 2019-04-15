# **Finding Lane Lines on the Road** 

### **Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. How my pipeline works

My pipeline is encapsulated in the *lane_lines()* function, consisted of 5 steps as follows:
1. converting the image to grayscale
2. smothing the image using Gaussian blur
3. detecting edges using Canny method
4. selecting a suitable ROI using polygon mask
5. detecting lane lines using Hough transformation 
6. merging the lane lines detected by droping the wrong-slope ones, grouping left/right ones and ployfitting the two parts
7.  drawing the lane lines

In order to draw a single line on the left and right lanes, I did not modify the draw_lines() function. Instead, I wrote a *merge_lines* function which is the step 6 as previously mentioned. 

You can control the *merge_lines* function call by the parameter *need_merge* of function *lane_lines*. 

This is the images with lane lines not merged using *lane_lines(raw_img, need_merge=False)*:

![lines not merged](https://view54dc5dc0.udacity-student-workspaces.com/view/CarND-LaneLines-P1/not_merged.png)

And this is the images with lane lines  merged using *lane_lines(raw_img, need_merge=True)*:

![lines not merged](https://view54dc5dc0.udacity-student-workspaces.com/view/CarND-LaneLines-P1/merged.png)


### 2. Potential shortcomings

Basically there are two shortcomings with my current pipeline: 
1. the merging line method is rough. I just dropped the lines with slope absolute values less than 0.5 and judged them to left or right ones by their slope negative/positive sign. 
2. the assumption of my pipeline is that: all lanes are lines. But actually, they are not. Curves are everywhere, especially at turning.


### 3. Suggest possible improvements to your pipeline

A possible improvement to the first shortcoming would be to use a more sophisticated grouping method which takes the concrete values (not just signs) into consideration. 

Another potential improvement to the second shortcoming could be to use curve detection method instead of the just-line detection for finding the lanes.
