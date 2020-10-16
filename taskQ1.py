import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import ml_metrics as metrics
from calculateDistances import *
from histograms import *


metricType = 'hell'
pathQ = 'qsd1_w1'
pathM = 'BBDD'

histMus,idsMus = labHist(pathM,64)
histQ,idsQ = labHist(pathQ,64)

with open(os.path.join(pathQ,'gt_corresps.pkl'), 'rb') as f:
    gtquery_list = pickle.load(f)

finalIds = []
for i in range(len(histQ)):
    allDist = []
    histQNorm = histQ[i]/np.max(histQ[i])
    histQNorm = np.array(histQNorm, dtype=np.float32)
    for j in range(len(histMus)):
        histMusNorm = histMus[j] / np.max(histMus[j])
        histMusNorm = np.array(histMusNorm, dtype=np.float32)
        if metricType == 'euclid':
            dist = euclid_dist(histQNorm,histMusNorm)
        elif metricType == 'chi':
            dist = chiSquare_dist(histQNorm,histMusNorm)
        elif metricType == 'corr':
            dist = corrDist(histQNorm, histMusNorm)
        elif metricType == 'histInt':
            dist = Hist_intersection(histQNorm, histMusNorm)
        elif metricType == 'hell':
            dist = Hellinger_kernel(histQNorm, histMusNorm)

        allDist.append(dist)
    allDist, idsSorted = zip(*sorted(zip(allDist, idsMus)))
    finalIds.append(list(idsSorted))

# 4. Return top K images
k = 5
mapkScore = metrics.mapk(gtquery_list,finalIds ,k)
print(metricType+' '+str(mapkScore))



