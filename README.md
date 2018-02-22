# Self-Driving Car Engineer Nanodegree

## Project # 5: Vehicle Detection and Tracking

**Name: Kendrick Amezquita**

**Email: kendrickamezquita@gmail.com**

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/veh_non_veh.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/test1_bbox.png
[image4]:  ./output_images/test5_bbox.png
[image5]: ./output_images/test1_heatmap.png
[image6]: ./output_images/test5_heatmap.png

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The histogram of oriented gradients (HOG) is used in the pipeline to extract the shape features of vehicles for detection and tracking. In the file `lesson_functions.py`, the function `get_hog_features()` (code line 13-30) takes an image and extracts its HOG with the aid of `skimage.hog()`. 

The `skimage.hog()` function has a number of parameters that can be tuned to improve features detection: `orientations`, `pixels_per_cell`, `cells_per_block`, as well as image color space.

Several images are provided in vehicle and non-vehicle datasets. In the image below, a sample of each class is depicted. 

![alt text][image1]

In view of the datasets, the `skimage.hog()` parameters where chosen with the aim of highlighting the features of the images. The following inset shows the output of `get_hog_features()` (applied to the images above) with the parameters selected as: Channel `Y` of color space `YCrCb`, `orientations`: 8, `pixels_per_cell`: 8, `cells_per_block`: 2.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The final choice of HOG parameters comes from a trade-off between classification accuracy (on a test set randomly chosen from the given datasets) and performace. We aimed at an accuracy greater than 99.7% with a performance of at least 1.5 sec/frame on the video processing pipeline.

The image feature vector is composed by the HOG feature vector combined with a color histogram feature vector and a spatially binned representation of the image. The final parameters of the image feature vector can be found in  `training_classifier.py` code line 79-89.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For the vehicle detection, a support vector classifier (SVC) was trained and tested on the datasets provided in the course. 

First, the features of the 7500 images  (randomly chosen) in the datasets were extracted and scaled (`training_classifier.py` code line 91-116). Then, the sklearn function `GridSearchCV()` was implemented to automatically determine the best configuration parameters for our classifier from the parameters set: `{'kernel':('linear', 'rbf'), 'C':[0.001, 0.1, 1, 10]}`. The SVC is trained in the file `training_classifier.py`code line 125-130.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window method was implemented following the code suggested in the lesson (file `video_car_lanelines.py` code line 60-108). 

First, a region of interest (ROI) is chosen (`y_start_stop = [350, 650]`)  to extract the history of gradient from. Later, a sliding window is used to sub-sample the overall HOG feature vector of the ROI (with `cells_per_step = 1` and  `scale = [1., 2.]`). Color histogram and spatial binning are also apply to this window to obtain the complete feature vector of the corresponding window.

Once again, the optimization parameters were chosen as a trade-off between accuracy and pipeline performance.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the images below, the vehicle detection outcome is presented.

![alt text][image3]
![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The final video ouput can be seen here:

[![](http://img.youtube.com/vi/vQfw3TrSdXg/0.jpg)](http://www.youtube.com/watch?v=vQfw3TrSdXg)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Following the method presented in class, positive detections are recorded in each frame of the video. Then, a heatmap is created and saved into a qeue (`maxlen = 8`). The heatmaps in the qeue are summed up and thesholded to identify and update the vehicle positions every 5 frames.  The function `scipy.ndimage.measurements.label()` is used to identify individual blobs in the resultant heatmap.  Each blob is assumed to be a single vehicle and bounding boxes are used to cover the area of each blob. (`video_car_lanelines.py` code line 110-119).

Below, two images example are presented to illustrate the method's outcome.

![alt text][image5]
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the problems I faced during the implementation of the pipeline was how to tune the parameters so as to improve the detection accuracy while maintaining an acceptable performance. Even though, support vector machines seem to work well for this application, the use of deep neural networks could improve the robustness of vehicles detection and possibly increase performance due to process vectorization.

