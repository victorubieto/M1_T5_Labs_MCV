import cv2
import numpy as np


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
        histr = cv2.normalize(histr,histr)
        histGray.append(histr)
        # plt.plot(histr)
        # plt.xlim([0, 256])
    return histGray


def calculate_rgbConcat_hist(listIm):
    histRGBConcat = []
    for img in listIm:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist,hist)
            if col == 'b':
                histBlue = hist
            elif col == 'g':
                histGreen = hist
            else:
                histRed = hist

        concHist = np.concatenate((histBlue, histGreen, histRed),axis=None)
        histRGBConcat.append(concHist)

    return histRGBConcat



def calculate_hsv_hist(listIm):
    histHSV = []
    bins = 8
    for img in listIm:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [2*bins, bins, int(bins/2)], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        histHSV.append(hist.flatten())

    return histHSV


def calculate_rgb_hist(listIm):
    histRGB = []
    bins = 8
    for img in listIm:
        hist = cv2.calcHist([img], [0, 1, 2], None, [bins,bins,bins], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        hist.flatten()
        histRGB.append(hist)

    return histRGB


