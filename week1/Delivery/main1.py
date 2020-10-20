import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from ml_metrics import mapk

from histograms import *
from similarity_measures import *
from background_extraction import *
from evaluation import *


# It loads the images from the input path into an array
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
    return images


# --- TASK 1 ---
# 1.1 Calculate histograms from museum images
path = 'BBDD'
c = False   # True = compute color histograms, False = don't

gtquery_list = []
gt_query_file = 'qsd1_w1/gt_corresps.pkl'
with open(gt_query_file, 'rb') as gtfd:
    gtquery_list = pickle.load(gtfd)

museumIm, museumIds = load_images(path) # load the images and the Ids from the DB
histGray = calculate_gray_hist(museumIm)    # get the gray histograms
if c is True:
    histBlue, histGreen, histRed = calculate_color_hist(museumIm)   # get the color histograms

# 1.2 For each query, calculate histogram
path = 'qsd1_w1'
q1Im = load_images(path)
histGrayQ = calculate_gray_hist(q1Im)
if c is True:
    histBlueQ, histGreenQ, histRedQ = calculate_color_hist(q1Im)


# --- TASK 3 ---
# 3.1 Sort the museum images from the most similar to the least              (THIS IS JUST FOR GRAYSCALE, FOR NOW!!!!!!!!)
for i in range(len(histGrayQ)):
    euclid = []
    chiSquare = []
    for j in range(len(histGray)):
        print(i)
        print(j)
        dist = euclid_dist(histGrayQ[i], histGray[j])
        euclid.append(dist)
        dist = chiSquare_dist(histGrayQ[i], histGray[j])
        chiSquare.append(dist)

    idxE = np.argsort(euclid)
    sortedEuclid = np.array(euclid)[idxE]
    sortedIdxE = np.array(museumIds)[idxE]
    idxC = np.argsort(chiSquare)
    sortedChi = np.array(chiSquare)[idxC]
    sortedIdxC = np.array(museumIds)[idxC]


# --- TASK 4 ---
# 4.1 Return the top k images 
k = 2   # modify here

for i in range(len(histGrayQ)):
    avPredEuclid = metrics.mapk(gtquery_list,sortedIdxE,k)


# --- TASK 5 ---
# 5.1 Background extraction
 = remove_background()  #TODO


# --- TASK 6 ---
# 6.1 Evaluate
 = evaluate()  # TODO