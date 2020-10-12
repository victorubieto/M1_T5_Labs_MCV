import cv2
import numpy as np
import ml_metrics as metrics
import pickle
from utils import *
from histograms import *
from calculateDistances import *


# 1. Calculate histograms from museum images
path = 'BBDD'
c = False

gtquery_list = []
gt_query_file = 'qsd1_w1/gt_corresps.pkl'
with open(gt_query_file, 'rb') as gtfd:
    gtquery_list = pickle.load(gtfd)

museumIm, museumIds = load_images(path)
histGray = calculate_gray_hist(museumIm)
if c is True:
    histBlue, histGreen, histRed = calculate_color_hist(museumIm)


# 2. For each query, calculate histogram
path = 'qsd1_w1'
q1Im, q1Ids = load_images(path)
histGrayQ = calculate_gray_hist(q1Im)
if c is True:
    histBlueQ, histGreenQ, histRedQ = calculate_color_hist(q1Im)

# 3. Compute all distances and order the histograms, from most similar to less similar
# THIS IS JUST FOR GRAYSCALE, FOR NOWW!!!
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

    # 4. Return top K images
    k = 2
    for i in range(len(histGrayQ)):
        avPredEuclid = metrics.mapk(gtquery_list,sortedIdxE,k)



