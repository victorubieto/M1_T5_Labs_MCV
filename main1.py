import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import os


# Generic functions
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
    return images


def calculate_gray_hist(listIm):
    histGray = []
    for img in listIm:
        print(img.max())
        print(img.shape)
        # cv2.imshow('og', img)
        # cv2.waitKey()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', imgGray)
        # cv2.waitKey()
        histr = cv2.calcHist([imgGray], [0], None, [256], [0, 256])
        histGray.append(histr)
        # plt.plot(histr)
        # plt.xlim([0, 256])
    return histGray


def calculate_color_hist(listIm):
    histBlue = []
    histGreen = []
    histRed = []
    for img in museumIm:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            if col == 'b':
                histBlue.append(histr)
            elif col == 'g':
                histGreen.append(histr)
            else:
                histRed.append(histr)

    return histBlue, histGreen, histRed


# Euclidean distance
def euclid_dist(vec1, vec2):
    d = np.sqrt(np.sum([(a - b) ** 2 for (a,b) in zip(vec1,vec2)]))
    return d


#Chi square distance
def chiSquare_dist(vec1,vec2):
    d = np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(vec1, vec2)])
    return d

# -----------------------------------------------------------------------------------------------------------------------

# 1. Calculate histograms from museum images
path = 'BBDD'
c = False

museumIm = load_images(path)
histGray = calculate_gray_hist(museumIm)
if c is True:
    histBlue, histGreen, histRed = calculate_color_hist(museumIm)

# Store them in a tmp folderr????

# 2. For each query, calculate histogram
path = 'qsd1_w1'
q1Im = load_images(path)
histGrayQ = calculate_gray_hist(q1Im)
if c is True:
    histBlueQ, histGreenQ, histRedQ = calculate_color_hist(q1Im)

# 3. Find the most similar histogram in the museum
# THIS IS JUST FOR GRAYSCALE, FOR NOWW!!!
euclid = []
chiSquare = []
for histQuery in histGrayQ:
    for hist in histGray:
        dist = euclid_dist(histQuery, hist)
        euclid.append(dist)
        dist = chiSquare_dist(histQuery,hist)
        chiSquare.append(dist)

