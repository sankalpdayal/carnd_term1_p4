# **Advanced Lane Finding Project**

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
[image2]: ./examples/undistort_output_road.png "Undistorted Road"
[image3]: ./examples/binary_gradient.png "Binary Gradient"
[image4]: ./examples/binary_color.png "Binary Color"
[image5]: ./examples/binary_color_gradient.png "Binary Combined"
[image6]: ./examples/persp_transform.png "Perspective Transformation"
[image7]: ./examples/persp_transform_all.png "Perspective Transformation and Binary Thresholding"
[image8]: ./examples/lane_pix_poly.png "Identified Lanes and Fitted Polynomial"
[image9]: ./examples/identified_lanes.png "Identified Lane, Radius of Curvature and Distance from Center"


### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. In the following writeup, I will describe my approach and results while covering all the rubric points.

### Camera Calibration

The code for this step is contained under the title of Camera Calibration in the first 2 code cell of the IPython notebook located in "./Advanced_Lane_Finding.ipynb".

1. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
2. To get the (x, y) pixel position of each of the corners in the image plane I used cv2.findChessboardCorners(). Every calibration image is first converted to gray scale and then searched for corners.
3. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
4. I then save these distortion coefficients into a pickle file to be used later.
5. I applied this distortion correction to the calibration1.jpg using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (Images)

#### 1. Distortion correction test images

Using the same distortion coefficients and  `cv2.undistort()`. The code is under the title Step1: Distortion correction on test images. I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Binary image creation containing lane pixels.  

I explored both the color and gradient thresholds based approach to generate a binary image. The code for this is under the title 'Step2: Binary image containing lane pixels' in the notebook. 

1. For gradient based apporach I visualized gradient in x and y, gradient magnitude and gradient direction. The function is defined as `abs_sobel_thresh()`. For gradient based approach I used combination of all the gradients to create binary image. The function is defined in `apply_sobel_combined()`. Following image shows the visualizations.

![alt text][image3]

2. For color based approach I visualized RGB channels, HSL channels, LUV channels and Lab channels separately. It can be observed that
	1. S of HLS can detect both white and yellow well
	2. L of LUV can also detect white better than yellow
	3. B of Lab is doing good job on detecting yellow.
	
For color based approach, I used combination of S channel for HLS, L from LUV and B from Lab to create binary image. The function is defined in  `apply_color_thresholds()`. Following image shows the visualizations.

![alt text][image4]

For finaly binary image, I used combined color and gradent based approaches. Here are my output for this step for all test images.

![alt text][image5]

#### 3. Perspective Transformation 

The code for my perspective transform includes a function called `create_birds_eye_view()`, which is defined in 3rd cell under the title Step3: Perspective Transform. The `create_birds_eye_view()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_perspective = np.float32([[280,669],[1027,669],[679,447],[600,447]])
dst_perspective = np.float32([[280,720],[1027,720],[1027,0],[280,0]])
```

Also at this step I defined a region of interest where the expected lane lines will be . The function is defined in the 1st cell under the title Step3: Perspective Transform. The function is called `region_of_interest()'

I verified that my perspective transform was working as expected by visualizing the warped counterpart of an image where lanes are actually parallel. Idea is that in warped image they should appear parallel. The output is as showing in the following image.

![alt text][image6]

Then I applied the binary thresholding on the transformed image. The result for all test images is shown here

![alt text][image7]

#### 4.  Identify Lanes and Fit Polynomial

In this step I used the binary thresholded image and used the histogram method to identify lane pixels. The method is described in the function `identify_lanes()` in the 1st cell under the title of Step4: Identify Lanes and Fit Polynomial. This function does the following steps

1. Takes in the binary thresholded image after the birds eye view transformation. 
2. Calculates the histogram for the image.
3. For the lower half of the image, divides it into left and right regions. For each region gets the x location where histogram shows maximum value. These become starting serach positions.
4. Now it divides the image into windows in y. Starting from bottom most window, it creates a region around the starting serach positions. If there are more than some percentage of white pixels in that region, it declares it as a good finding and these good findings become the next x positions for the region.
5. This step is continued till all the windows are covered. 
6. Function returns ndexes of left and right lanes in a non zero matrix and overlayed image showing windows it searched.

Once the lane pixels are identified, a second order polynomial is fit onto these pixles for both left and right seprately. The function doing this is defined in 2nd cell under the same title and is called ` get_poly_fit_both_lanes()`

The result of the lane pixel identification and polynomical fitting on test images is shown in the following image. 

![alt text][image8]

#### 5. Radius of Curvature and Distance From Lane Center

I did this in cells under the title Step5: Radius of Curvature and Distance From Lane Center. The function to get radius of curvature is `get_radius_of_curvature()` and to get distance from lane center is `get_center_position()`. 

#### 6. Final Pipeline

Final pipeline is follows
1. Undistort the image
2. Apply binary thresholding using color (I tried sobel based as well but no gain was found, so dropped it)
3. Get region of interest
4. Get warped image by applying perspective transformation
5. Using histogram and windowing based method identify lanes.
6. Using the identified pixels, fit 2nd order polynomial.
7. Using the polynomial, get radius of curvature and distance from center.

I implemented this step in 1st cell under the title  Final pipeline is follows. Here is are my results on a test images. The green region shows the area identified as lanes.

![alt text][image9]

---

### Pipeline (video)

#### Now since the time based information is also available, I used following ideas for better detection of lanes now

1. For every frame, calculate difference in curvature, how parallel they are and distance between them.
2. Using these parameters decide if current frame has good lane detection or not.
3. If yes, then check if either right lane is bad or left.
4. Depending on how much the coeffecients of fit have changed compared to previous frame, decide if current frame is good or not. This is done for left and right lanes separately.
5. Depending on if either is good or bad, update in history for each lane accordingly.
6. If the lane is detected as bad, the measurement is not kept in history and a counter is increased counting how many frames were bad in past. If this counter goes beyond a threshold, complete history dropped and algorithm starts fresh.
7. If the lane is detected as good, the measurement is added to history. There is a maximum length on history and as more new measurements come in, older are dropped.
8. Final output is given as average over history.
9. In history, polynomial coefficients and x positions from the fit are stored. For best current estimate, average over the history is taken.

In order to apply this idea, I created a class called `Line()` that stores current and past estimates and uses them to get the best estimates. This class is defined in the 1st cell below the title Pipeline (video). Also to determine if current measurement is good or not, I defined a function `check_status_of_detection()`. This is in the 2nd cell under the same title. The final pipeline for video based lane detection looks like as follows.

1. Undistort the image
2. Apply binary thresholding using color (I tried sobel based as well but no gain was found, so dropped it)
3. Get region of interest
4. Get warped image by applying perspective transformation
5. Using histogram and windowing based method identify lanes.
6. Using the identified pixels, fit 2nd order polynomial.
7. Using the polynomial, get radius of curvature and distance from center.
8. Using polynomial, and raidus of curvature estimate from both lanes, estimate difference in curvature, non parallelism, error in size
9. If above parameters are not ideal, then check for each lane separately if the change in values of polynomial coefficients is less than a threshold and using this result identify if current measurement for each lane is good or bad.
10. Update the measurements accordingly.
11. Get smoothened values of poly coefficients for each lane and x fit.
12. Using the coefficients and x fit get new curvature and distance from center.

This pipeline is defined in the 3rd cell under the same title by the name of function `lane_detect_pipeline()`. It has flag `smoothening_on` which can be set to `False` to turn off smoothing.   

This is the result of algorithm on the test video.

[![Advanced Lane Detection with Smoothing](http://img.youtube.com/vi/5HKC0a_817M/0.jpg)](https://www.youtube.com/watch?v=5HKC0a_817M)

To compare it against the detection without smoothing is given here.

[![Advanced Lane Detection (Comparison of Smoothing)](http://img.youtube.com/vi/jqcxObyJOp0/0.jpg)](https://www.youtube.com/watch?v=jqcxObyJOp0)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems that I faced

1. The major problem I faced is for binary image detection where only lane pixels have to be identified. I felt that there is no universal threshold that works for all lighting conditions. 
2. Other problem I faced was defining a logic to tell if the current measurement is good or not to decide if to store it in history or not. I have used gamut of features to do so but I think the thresholds identified are very specific to certain conditions.
3. In generaly I believe that there are so many hand crafted thresholding and logic in the whole pipeline, it would be better if we can create some sort of unified, pipeline that can be tuned all together for better lane detection. Somethine like we do in deep learning frameworks.

Possible solutions

1. The thresholds have to be adaptive depending on the current lighting conditions. The lighting conditions can be estimated may be from the image itself.
2. The tuning should be automated. May be we can define an ideal radius of curvature using google maps for a given location. And during the run, using the GPS location, we can fetch the radius of curvature and tune our algorithm such that it reduced the RMS error. This might not be possible in current problem since we only have video and no GPS coordinates.
3. Also I believe that in general the algorithm has to be tested for different lighting conditions like day, night, raining etc. And apply some correction depending on the these.