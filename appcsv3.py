#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras

import sys

import matplotlib.pyplot as plt
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules

import cv2
import csv
import os
import numpy as np
from scipy.io import loadmat

import numpy as np
import cv2


# %% Functions

def listFile(path, ext):
    
    '''    

    Parameters
    ----------
    path : string
        Directory of processing images. 
    ext : string
        Desired file extention.

    Returns
    -------
    A list of all files with specific extension in a directory (including subdirectory).

    '''

    filename_list, filepath_list = [], []
    # r = root, d = directories, f = files
    for r, d, f in os.walk(path):
        for filename in f:
            if ext in filename:
                filename_list.append(filename)
                filepath_list.append(os.path.join(r, filename))
    return sorted(filename_list), sorted(filepath_list)

def listTile(path):
    # Return a list of dir of tiles
    
    dir_list = []
    dirname_list = []
    for r, d, f in os.walk(path):
        if not d:
            dir_list.append(r)
            dirname_list.append(os.path.basename(r))
    return sorted(dirname_list), sorted(dir_list)


def non_max_suppression_merge(boxes, overlapThresh=0.5, sort=4):
    '''
    https://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
    '''
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    idxs = np.argsort(boxes[:,sort])
	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # expand the picked bounding box
        if np.where(overlap > overlapThresh)[0].size > 0:
            # find the largest bounding box with overlap
            xxx1 = min(x1[i], x1[idxs[np.where(overlap > overlapThresh)[0]]].min())
            yyy1 = min(y1[i], y1[idxs[np.where(overlap > overlapThresh)[0]]].min())
            xxx2 = max(x2[i], x2[idxs[np.where(overlap > overlapThresh)[0]]].max())
            yyy2 = max(y2[i], y2[idxs[np.where(overlap > overlapThresh)[0]]].max())
            boxes[i,:4] = [xxx1,yyy1,xxx2,yyy2]
            # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
    return boxes[pick]

# for multi-channel detection

def stitchDetection(detections, H, W, xsize=512, ysize=512, step=448):
    ''' stitch predictions on a single tile image
    '''
    
    # mask_overlap = np.zeros((H,W), dtype=float)
    x_overlap = xsize-step; y_overlap = ysize-step
    rows = []
    for row in range(step,W,step):
        rows.extend(list(range(row-32,row+x_overlap+32)))
    # mask_overlap[rows] = 1
    cols = []
    for col in range(step,H,step):
        cols.extend(list(range(col-32,col+y_overlap+32)))
    # mask_overlap[:,cols] = 1
    
    overlap_idx = []
    for i, detection in enumerate(list(detections)):
        box = list(map(int, detection[:-1]))
        if (box[0] in rows) or (box[1] in cols):
            overlap_idx.append(i)
    
    overlap_detections = detections[overlap_idx].copy()
    clean_mask = np.ones(detections.shape[0], dtype=bool)
    clean_mask[overlap_idx] = False
    clean_detections = detections[clean_mask].copy()
    
    if overlap_detections.size > 1:
        overlap_detections = non_max_suppression_merge(overlap_detections)
        clean_detections = np.append(clean_detections, overlap_detections, 
                                     axis=0)
        
    return clean_detections  


pATHTEST='/home/greenbaum-gpu/Reuben/keras-retinanet/images' #change this to the path that your image is in.
testnames, testpaths = listFile(pATHTEST, '.tif')


# print(model.summary())
# labels_to_names = {0: 'class 0', 1: 'class 1', 2: 'class 2', ...}, e.g.
labels_to_names = {0: 'uncertain', 1: 'yellow neuron', 2: 'yellow astrocyte', 
                    3: 'green neuron', 4: 'green astrocyte', 5: 'red neuron', 
                    6: 'red astrocyte'}

tHRESHOLD = 0.5 # threshold for detection confidence score
xsize = 512; ysize = 512; step = 448 # size and step of image patches for detection
classes = list(labels_to_names.values())
num_class = len(classes)
pATHRESULT = '/home/greenbaum-gpu/Reuben/keras-retinanet/output/csv_output' # output dir
all_detections = [[None for i in range(num_class)] for j in range(len(testnames))]
clean_detections = [[None for i in range(num_class)] for j in range(len(testnames))]

model_path = os.path.join('snapshots', 
                          'trainedmodel.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)
# Initialize count for relevant classes only (1 to 6).
count = np.zeros((1, 6), dtype=int)


# In[5]:


for i, testpath in enumerate(testpaths):
    CSV = os.path.join(pATHRESULT,testnames[i] + '_result.csv')

    fullimg_c1 = read_image_bgr(testpath)
    fullimg = np.zeros(fullimg_c1.shape, dtype=np.uint16)
    fullimg[:, :, 2] = fullimg_c1[:, :, 2].copy()  # BGR
    fullimg[:, :, 1] = fullimg_c1[:, :, 1]

    if (fullimg[:, :, 2].sum() > 0) & (fullimg[:, :, 1].sum() > 0):
        fulldraw = fullimg.copy() / 257  # RGB to save
        fulldraw = (fulldraw * 3).clip(0, 255)  # Increase brightness

        # Padding
        H0, W0, _ = fullimg.shape

        if not (H0 - ysize) % step == 0:
            H = H0 - H0 % step + ysize
        else:
            H = H0
        if not (W0 - xsize) % step == 0:
            W = W0 - W0 % step + xsize
        else:
            W = W0

        if W != W0 or H != H0:
            fullimg_pad = np.zeros((H, W, 3), dtype=np.uint16)
            fullimg_pad[0:H0, 0:W0] = fullimg.copy()
        else:
            fullimg_pad = fullimg.copy()

        n = 0
        raw_detections = np.empty((0, 6))

        for x in range(0, W, step):
            for y in range(0, H, step):
                offset = np.array([x, y, x, y])

                # Load image
                image = fullimg_pad[y:y + ysize, x:x + xsize]

                # Preprocess image for network
                image = preprocess_image(image)
                image, scale = resize_image(image)

                # Process image
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

                # Correct for image scale
                boxes /= scale
                boxes += offset
                boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, W0)
                boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, H0)

                # Select indices which have a score above the threshold
                indices = np.where(scores[0, :] > tHRESHOLD)[0]

                # Select those scores
                scores = scores[0][indices]

                # Find the order with which to sort the scores
                scores_sort = np.argsort(-scores)

                # Save detections
                image_boxes = boxes[0, indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[0, indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1
                )
                raw_detections = np.append(raw_detections, image_detections, axis=0)
    else:
        raw_detections = np.empty((0, 6))

    # Copy detections to all_detections
    for label in range(num_class):
        all_detections[i][label] = raw_detections[raw_detections[:, -1] == label, :-1]

        # Stitch detections
        detections = raw_detections[raw_detections[:, -1] == label, :-1].copy()
        if detections.size > 1:
            cleaned_detections = stitchDetection(detections, H0, W0, xsize, ysize, step)
        else:
            cleaned_detections = detections.copy()
        cleaned_detections = np.concatenate(
            [
                cleaned_detections,
                np.zeros([cleaned_detections.shape[0], 1]),
                np.ones([cleaned_detections.shape[0], 1]) * (i + 1),
            ],
            axis=1,
        )
        clean_detections[i][label] = cleaned_detections

        # Visualize detections and output
        if cleaned_detections.size > 1:
            for j, detection in enumerate(list(cleaned_detections)):
                b = list(map(int, detection[:4]))
                color = label_color(label)
                draw_box(fulldraw, b, color=color, thickness=2)
                target_directory= '/home/greenbaum-gpu/Reuben/keras-retinanet/output'


                # Save average intensity of the box area
                cleaned_detections[j, 5] = fullimg[b[1]:b[3], b[0]:b[2]].mean()
                with open(CSV, 'w', newline='') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',')
                    filewriter.writerow([raw_detections])

                cv2.imwrite(os.path.join(target_directory, testnames[i] + '_THRE_' + f"{tHRESHOLD:1.1f}" + ".png"), fulldraw)