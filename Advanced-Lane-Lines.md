## Advanced Lane Finding Project

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.

<div  align="center">    
<img src="output_images/demo.gif" width=60% height=60% border=0/>
</div>

---

### Introduction
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In this step, We mainly use several functions.
- `cv2.findChessboardCorners()`: used to get `imgpoints`
- `cv2.drawChessboardCorners()`: used to visualize result
- `cv2.calibrateCamera()`: used to get `mtx`(camera matrix) and `dist`(distortion coefficients)
- `cv2.undistort(img, mtx, dist, None, mtx)`: used to undistort a test image   

The code for this step is contained in the cell (`In [2] & In [3]`) of the IPython notebook located in "Advanced-Lane-Lines.ipynb". The result is as follows:

<div  align="center">    
<img src="output_images/Calibration_result.png" width=90% height=90% border=0/>
</div>

### Pipeline (single images)

#### 1. Correcting for Distortion.

- Step 1: Load the data of camera matrix and distortion coefficients has been saved locally.
```
with open( "./output_images/wide_dist_pickle.p", "rb" ) as f:
    data = pickle.load(f)   
mtx = data['mtx']
dist = data['dist']
```
- Step 2: Undistorting a test image.
```
img_path_list = glob.glob("./test_images/*.jpg")
image = mpimg.imread(img_path_list[2])
undist_img = undistort(image, mtx, dist)
```
The result is as follows:
<div  align="center">    
<img src="output_images/compare.png" width=100% height=100% border=0/>
</div>

#### 2. Perspective transform.

The code for my perspective transform includes a function called `perspective_Transform(image, src = None, dst = None)`, which appears in the cell (`In [7]`) of the IPython notebook located in "Advanced-Lane-Lines.ipynb".  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    img_size = (image.shape[1], image.shape[0])
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])

    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])   
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 340, 720      | 
| 589, 457      | 340, 0        |
| 698, 457      | 995, 0        |
| 1145, 720     | 995, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<div  align="center">    
<img src="output_images/Transform_compare.png" width=100% height=100% border=0/>
</div>

#### 3. Thresholded binary image. 

- Step 1: Implement the basic method

    - **Abs_sobel**:`abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255))`
    
    - **Magnitude of the Gradient**:`mag_thresh(image, sobel_kernel=3, mag_thresh=(0,255))`  
    - **Direction of the Gradient**:`dir_threshold(image, sobel_kernel=3, thresh=(0., np.pi/2))`  
    - **HLS Color Thresholding**`def hls_select(image, channel='h',thresh=(0, 255))`

- Step 2: Combining Thresholds   
I used `S` `L` color thresholds and gradient thresholds of `x` to generate a binary image. The corresponding code is `gradient_Threshold(img,thresh_dic)`.

- Step 3: Morphological Transformations   
In order to reduce noise pixels, I used [cv2.erode()](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
to process binary image.

Here's an example of my output for this step.  (note: this is not actually from one of the test images)
<div  align="center">    
<img src="output_images/Binary_compare.png" width=100% height=100% border=0/>
</div>

#### 4. Finding the Lines and Fit a Polynomial
- Step 1: Find the index of lane pixels   
`leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)`
- Step 2: Fit a Polynomial   
```python
left_fit = np.polyfit(lefty, leftx, 2)   
right_fit = np.polyfit(righty, rightx, 2)
```
- Step 3: Visualize the result  

The result is as follows:
<div  align="center">    
<img src="output_images/fit_compare.png" width=100% height=100% border=0/>
</div>

#### 5. Measuring Curvature and The position of the vehicle with respect to center.
- Step 1: Find the index of lane pixels   
`leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)`
- Step 2: Calculate real-world lane curvature values  
```python
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    y_eval = np.max(ploty)
    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5 / np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad = (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5 / np.absolute(2*right_fit_cr[0])  ## Implement the calculation of the right line here
```
- Step 3: Calculate the distance from the center of the lane.  
```python
    car_position = binary_warped.shape[1]/2 * xm_per_pix
    h = binary_warped.shape[0] * ym_per_pix
    l_fit_x_int = left_fit_cr[0]*h**2 + left_fit_cr[1]*h + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*h**2 + right_fit_cr[1]*h + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center_dist = (car_position - lane_center_position)
```
- Step 4: Visualize the result.

I implemented this step in lines # through # in my code in `Advanced-Lane-Lines.ipynb` in the function `drawing_image()`.  Here is an example of my result on a test image:

<div  align="center">    
<img src="output_images/result_img.png" width=100% height=100% border=0/>
</div>

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- Opencv Morphological Transformations  
Erosion and Dilation of a thresholded binary image to reduce noise pixels and enhance lane line pixels.

- Problems  
The code of Video pipeline isn't good performance on `challenge_video.mp4`, there may be a problem with the part of combining thresholds. Need to tune combining thresholds for different scenarios.