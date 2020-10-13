import cv2
import numpy as np
import ml_metrics as metrics
import pickle

# It computes the gray level histogram for each image inside the input array and returns an array of the histograms
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

# It computes all RGB level histograms for each image inside the input array and returns three arrays, each one
# containing the list of histograms of each color level
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
