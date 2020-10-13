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
path = 'smallMuseum'

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
path = 'smallQuery'
q1Im, q1Ids = load_images(path)
histQ = calculate_rgbConcat_hist(q1Im)
# plt.plot(histQ[0])
# plt.xlim([0, 256])

# 3. Compute all distances and order the histograms, from most similar to less similar
sortedIds = []

for i in range(len(histQ)):
    plt.plot(histQ[i])
    euclid = []
    chiSquare = []
    L1 = []
    inter = []
    hellinger = []
    allMetrics = []
    for j in range(len(histMus)):
        plt.plot(histMus[j])
        dist = euclid_dist(histQ[i], histMus[j])
        print('Euclid:', dist)
        euclid.append(dist)
        allMetrics.append(euclid)

        dist = chiSquare_dist(histQ[i], histMus[j])
        print('Chi',dist)
        chiSquare.append(dist)
        allMetrics.append(chiSquare)

        dist = L1_dist(histQ[i], histMus[j])
        print('L1:',dist)
        L1.append(dist)
        allMetrics.append(L1)

        dist = Hist_intersection(histQ[i], histMus[j])
        print('Intersection:',dist)
        inter.append(dist)
        allMetrics.append(inter)

        dist = Hellinger_kernel(histQ[i], histMus[j])
        print('Hellinger:',dist)
        hellinger.append(dist)
        allMetrics.append(hellinger)

    sortedIds_temp = []
    for k in range(5):
        idS = sort_ids(allMetrics[k],museumIds)
        sortedIds_temp.append(idS)
    sortedIds.append(sortedIds_temp)

# 4. Return top K images

k = 5
for j in range(len(histQ)):
    print('Image' + str(j))
    for i in range(len(metric)):
        mapkScore = metrics.mapk(gtquery_list,sortedIds[j] ,k)
        print(metric[i]+' '+str(mapkScore))



