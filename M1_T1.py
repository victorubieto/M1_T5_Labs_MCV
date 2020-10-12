# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:24:44 2020

@author: mohaa
"""


import cv2 as cv
import glob
from matplotlib import pyplot as plt
import numpy 


def getImagesPaths(path,extension):
    imgArray = []
    if extension == '.jpg':
        direction = glob.glob(str(path) + "\\*.jpg")
        for file in direction:
            imgArray.append(file)
            
    elif extension == '.png':
        for file in glob.glob(str(path) + "\\*.jpg"):
            imgArray.append(file)
    return imgArray

def getGrayImages(imgArray):
    imgGrayArray = []
    imgBGRArray = []
    
    for sample in imgArray:
        img = cv.imread(str(sample))
        imgBGRArray.append(img)
        imgGrayArray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return imgGrayArray, imgBGRArray

def getHistogram(imgArray, keyWord):
    histogramArray = []
    
    if keyWord == 'BGR':
        #channels: index== 0-->B, index== 1-->G, index== 2-->R
        for index in range(0, len(imgArray)):
            img = imgArray[index]
            histogramArray.append([[0],[0],[0]])
            histogramArray[index][0] = cv.calcHist([img],[0],None,[256],[0,256])
            histogramArray[index][1] = cv.calcHist([img],[1],None,[256],[0,256])
            histogramArray[index][2] = cv.calcHist([img],[2],None,[256],[0,256])
    elif keyWord == 'Gray':
        for img in imgArray:
            histogramArray.append(cv.calcHist([img],[0],None,[256],[0,256])) 
    return histogramArray
    
def getEuclideanDistance(reference, sample):
    global referencee, samplee
    
    referencee = []
    samplee = []
    
    tiny = 1e-15
    #'sample' is the sample desired image to compare with our DB 
    euclideanDistance = []
    k = 0
    correlation = []
    
    for i in range(0,len(sample)):
        samplee.append([])
        for j in range(0,len(sample[i])):
            samplee[i].append(float(sample[i][j] + tiny))
    for i in range(0,len(reference)):
        referencee.append([])
        for j in range(0,len(reference[i])):
            referencee[i].append(float(reference[i][j] + tiny))
    
    
    
    for imgHistogram in reference:
        euclideanDistance.append([])
        for i in range(0,len(imgHistogram)):
            # print('Iteracion : ' + str(k) + ' de ' + str(len(reference)))
            # print(len(imgHistogram))
            # print(len(imgHistogram[i]))
            # print(imgHistogram[i][0])
            # print('Sample : ' + str(sample[0][i][0]))
            dist = numpy.linalg.norm(imgHistogram[i][0]-sample[0][i])
            # print('Distancia : ' + str(dist))
            euclideanDistance[k].append(dist)
            # print('Euclidean Distance : ' + str(euclideanDistance))
        k = k + 1
    # print(euclideanDistance[120])
    return euclideanDistance
    




def main():
    global diff
    
    #define dif paths
    pathDB = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\BBDD\BBDD'
    pathSample = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\qsd1_w1\qsd1_w1'
    
    #get path of images
    jpgImagesPaths = getImagesPaths(pathDB,'.jpg')
    pngImagesPaths = getImagesPaths(pathDB,'.png')
    sampleImagesPaths = getImagesPaths(pathSample,'.jpg')
    
    #get images in gray and BGR
    jpgGrayImages, jpgBGRImages = getGrayImages(jpgImagesPaths)
    pngGrayImages, pngBGRImages = getGrayImages(pngImagesPaths)
    sampleGrayImages, sampleBGRImages = getGrayImages(sampleImagesPaths)
    
    #get hist of gray imgs
    jpgGrayImagesHistogram = getHistogram(jpgGrayImages, 'Gray')
    pngGrayImagesHistogram = getHistogram(pngGrayImages, 'Gray')
    sampleGrayImagesHistogram = getHistogram(sampleGrayImages, 'Gray')
    
    #get histogram of BGR imgs
    jpgBGRImagesHistogram = getHistogram(jpgBGRImages, 'BGR')
    pngBGRImagesHistogram = getHistogram(jpgBGRImages, 'BGR')
    sampleBGRImagesHistogram = getHistogram(sampleBGRImages, 'BGR')
        
    diff = getEuclideanDistance(jpgGrayImagesHistogram,sampleGrayImagesHistogram)
    

    
    plt.subplot(1,1,1)
    plt.plot(jpgGrayImagesHistogram[120],'Gray')
    # plt.subplot(4,1,2)
    # plt.plot(jpgBGRImagesHistogram[120][0],'Blue')
    # plt.subplot(4,1,3)
    # plt.plot(jpgBGRImagesHistogram[120][1],'Green')
    # plt.subplot(4,1,4)
    # plt.plot(jpgBGRImagesHistogram[120][2],'Red')
    # plt.subplot(4,1,3)
    # plt.plot(diff,'Black')
    

    plt.subplot(1,1,1)
    plt.plot(sampleGrayImagesHistogram[0],'k')
    # plt.subplot(4,1,2)
    # plt.plot(sampleBGRImagesHistogram[0][0],'c')
    # plt.subplot(4,1,3)
    # plt.plot(sampleBGRImagesHistogram[0][1],'y')
    # plt.subplot(4,1,4)
    # plt.plot(sampleBGRImagesHistogram[0][2],'m')
    
    for i in range(0,len(jpgGrayImagesHistogram)):
        plt.subplot(1,1,1)
        plt.plot(diff[i])


    
    # print(diff.index(min(diff)))
    # print(jpgImagesPaths[diff.index(min(diff))])
    # print('-----')
    # print(min(diff))
    # print(diff[120])
    
    
    
if __name__ == "__main__":
    main()