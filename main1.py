import cv2
import numpy as np
import ml_metrics as metrics
import pickle
from utils import *
from histograms import *
from calculateDistances import *
import  matplotlib.pyplot as plt

metric = ['Euclid','ChiSquare','L1','Intersection','Hellinger']

# 1. Calculate histograms from museum images
path = 'BBDD'

gtquery_list = []
gt_query_file = 'qsd1_w1/gt_corresps.pkl'
with open(gt_query_file, 'rb') as gtfd:
    gtquery_list = pickle.load(gtfd)

museumIm, museumIds = load_images(path)
#Here call the histogram you want
histMus = calculate_rgbConcat_hist(museumIm)
# plt.plot(histMus[0])
# plt.xlim([0, 256])

# 2. For each query, calculate histogram
path = 'qsd1_w1'
q1Im, q1Ids = load_images(path)
histQ = calculate_rgbConcat_hist(q1Im)
# plt.plot(histQ[0])
# plt.xlim([0, 256])

# 3. Compute all distances and order the histograms, from most similar to less similar
sortedIds = []
allMetrics = []
for k in range(len(metric)):

    sortedIds_temp = []
    imDist = []
    for i in range(len(histQ)):
        #plt.plot(histQ[i])
        distFin = []
        for j in range(len(histMus)):
            if k == 0:
                #plt.plot(histMus[j])
                dist = euclid_dist(histQ[i], histMus[j])
                print('Euclid:', dist)
            elif k == 1:
                dist = chiSquare_dist(histQ[i], histMus[j])
                print('Chi',dist)
            elif k == 2:
                dist = L1_dist(histQ[i], histMus[j])
                print('L1:',dist)
            elif k == 3:
                dist = Hist_intersection(histQ[i], histMus[j])
                print('Intersection:',dist)
            else:
                dist = Hellinger_kernel(histQ[i], histMus[j])
                print('Hellinger:',dist)

            distFin.append(dist)
        ordMetric, idS = sort_ids(distFin, museumIds)
        imDist.append(distFin)
        sortedIds_temp.append(idS)
    allMetrics.append(imDist)
    sortedIds.append(sortedIds_temp)

# 4. Return top K images
k = 1
for i in range(len(metric)):
    mapkScore = metrics.mapk(gtquery_list,sortedIds[i] ,k)
    print(metric[i]+' '+str(mapkScore))



