#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 18:11:36 2017

@author: kendrick
"""

import numpy as np
import cv2
import glob
import time
import pickle

from random import shuffle

from sklearn.svm import LinearSVC
from sklearn import svm, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from skimage.feature import hog
from lesson_functions import *

from pathlib import Path

# Read in cars and notcars
cars = [] 
notcars = []

path = 'small_dataset/'
images = glob.glob(path+'*.jpeg')
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

path = 'vehicles/GTI_Right/'
images = glob.glob(path + '*.png')
for image in images:
    cars.append(image)
    
path = 'vehicles/GTI_Left/'
images = glob.glob(path + '*.png')
for image in images:
    cars.append(image)
    
path = 'vehicles/GTI_Far/'
images = glob.glob(path + '*.png')
for image in images:
    cars.append(image)

path = 'vehicles/KITTI_extracted/'
images = glob.glob(path + '*.png')
for image in images:
    cars.append(image)
    
path = 'non-vehicles/GTI/'
images = glob.glob(path + '*.png')
for image in images:
    notcars.append(image)
    
path = 'non-vehicles/Extras/'
images = glob.glob(path + '*.png')
for image in images:
    notcars.append(image)
    
shuffle(cars)
shuffle(notcars)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 7500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### Tweak feature parameters.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16   # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 650] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

#my_clf = Path('clf.p')
#if not (my_clf.is_file()):
    
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)

t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

dist_clf={"clf": clf,
          "X_scaler": X_scaler,
          "color_space": color_space,
          "orient": orient,
          "pix_per_cell": pix_per_cell,
          "cell_per_block": cell_per_block,
          "hog_channel": hog_channel,
          "spatial_size": spatial_size,
          "hist_bins": hist_bins,
          "spatial_feat": spatial_feat}

with open('dict_clf.p', 'wb') as f:
    pickle.dump(dist_clf, f)
        