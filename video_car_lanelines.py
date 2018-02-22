#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 03:37:52 2017

@author: kendrick
"""
import numpy as np
import cv2

import pickle
import os
import time

from sklearn.svm import LinearSVC
from sklearn import svm, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

from skimage.feature import hog
from lesson_functions import *
from Advanced_LaneLines_LIB import *

def find_cars(img, y_start_stop, scale, dict_clf): #HOG sub-sampling

    clf = dict_clf["clf"]
    X_scaler = dict_clf["X_scaler"]
    color_space = dict_clf["color_space"]
    orient = dict_clf["orient"]
    pix_per_cell = dict_clf["pix_per_cell"]
    cell_per_block = dict_clf["cell_per_block"]
    hog_channel = dict_clf["hog_channel"]
    spatial_size = dict_clf["spatial_size"]
    hist_bins = dict_clf["hist_bins"]
    spatial_feat = dict_clf["spatial_feat"]
            
    img_tosearch = img[y_start_stop[0]:y_start_stop[1],:,:]
    
    if color_space != 'BGR':
        if color_space == 'HSV':
            ctrans_tosearch  = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch  = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)
    
    # Compute individual channel HOG features for the entire image
    hog = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float64)
    box_list = []
    
    for scl in scale:
    
        if scl != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scl), np.int(imshape[0]/scl)))
                
        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                #print("Here")
                # Extract HOG for this patch
                hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
        
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
        
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))
              
                # Get color features
                hist_features = color_hist(subimg, nbins=hist_bins)
                if spatial_feat == True:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    features = np.hstack((spatial_features, hist_features, hog_features))
                else:
                    features = np.hstack((hist_features, hog_features))
        
                # Scale features and make a prediction
                test_features = X_scaler.transform(features.reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = clf.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scl)
                    ytop_draw = np.int(ytop*scl)
                    win_draw = np.int(window*scl)
                    box_list.append([(xbox_left, ytop_draw+y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0])])                    
                
    # Add heat to each box in box list
    heatmap = add_heat(heatmap,box_list)
        
    # Apply threshold to help remove false positives
    #heatmap = apply_threshold(heatmap, 1)
    
    # Visualize the heatmap when displaying    
    #heatmap = np.clip(heatmap, 0, 255)

    return heatmap

f = open('calibration.p', 'rb')
[mtx, dist] = pickle.load(f)
f.close()

f = open('dict_clf.p', 'rb')
dict_clf = pickle.load(f)
f.close()

left_line = Line()
right_line = Line()
left_fit = []
right_fit = []
M, invM = perspective_parameters((1280,720))

fail = 0

y_start_stop = [350, 650] # Min and max in y to search in slide_window()
scale = [1., 2.]

file = 'project_video.mp4'

# Define the codec and create VideoWriter object
fname=os.path.splitext(file)[0]
print(fname)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(fname+'_output.mp4',fourcc, 20.0, (1280,720))

cap = cv2.VideoCapture(file)

from collections import deque
history_heatmap = deque(maxlen = 8)

frame_counter=0

print(time.strftime('%X %x %Z'))
t=time.time()
while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if ret == True:
        
        frame_counter +=1

        result, left_fit, right_fit, fail = lane_lines(frame, left_line, right_line, left_fit, right_fit, mtx, dist, M, invM, fail)
        
        if frame_counter == 1:
            thrs = 10
        else:
            thrs = 35          
        
        if not bool(frame_counter%5):
            heatmap = np.sum(history_heatmap, 0)
            heatmap = apply_threshold(heatmap, thrs)
            heatmap = np.clip(heatmap, 0, 255)
            labels = label(heatmap)
        else:
            history_heatmap.append(find_cars(frame, y_start_stop, scale, dict_clf))
            
        draw_img = draw_labeled_bboxes(np.copy(result), labels)
        out.write(draw_img)

    else:
        t2 = time.time()
        print(round(t2-t, 2))
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()