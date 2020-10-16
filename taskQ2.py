import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import ml_metrics as metrics
from calculateDistances import *
from histograms import *
from utils import *
from masks import *



# 5. Create a binary mask to evaluate the method & compute the descriptors on the foreground pixels.

# Load gt masks
pathQ2 = 'qsd2_w1'
q2Gt_mask, q2IdsMs = load_masks(pathQ2)
# Compute binary masks
q2Im = []
name = []
idsq2 = []
for file in os.listdir(pathQ2):
    if file.endswith('.jpg'):
        im = cv2.imread(os.path.join(pathQ2, file))
        q2Im.append(im)
        name = file.split('.')[0]
        idsq2.append(int(name))
idsq2Ord, q2ImOrd = zip(*sorted(zip(idsq2, q2Im)))

masks = get_mask_array(q2ImOrd)

# Compute the descriptors on the foreground pixels
q2Im_cropped = crop_imgarray(q2ImOrd)


# 6. Evaluation

# Calculate Precision, Recall and F1 metrics on the predicted masks
PRs, RCs, F1s = mask_evaluation(q2Gt_mask, masks)
print('precision:',np.mean(PRs))
print('recall:',np.mean(RCs))
print('f1:',np.mean(F1s))

#-------------------------------Calculate correspondences with cropped images-------------------------------------------
metricType = 'hell'
pathQ = 'qsd2_w1'
pathM = 'BBDD'

# for img in q2Im_cropped:
#     plt.figure()
#     plt.imshow(img)
histMus,idsMus = labHist(pathM,64)
histQ,idsQ = labHist2(q2Im_cropped,64,idsq2Ord)

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


