# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:11:36 2020

@author: mohaa
"""
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import numpy 
import heapq
from scipy.spatial import distance


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
    imgRGBArray = []
    
    for sample in imgArray:
        img = cv.imread(str(sample))
        imgRGBArray.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        imgGrayArray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        
    return imgGrayArray, imgRGBArray

def getHistogram(imgArray, keyWord):
    histogramArray = []
    
    if keyWord == 'RGB':
        #channels: index== 0-->B, index== 1-->G, index== 2-->R
        for index in range(0, len(imgArray)):
            img = imgArray[index]
            histogramArray.append([[0],[0],[0]])
            
            h_b = cv.calcHist([img],[0],None,[256],[0,256])
            h_g = cv.calcHist([img],[1],None,[256],[0,256])
            h_r = cv.calcHist([img],[2],None,[256],[0,256])
            
            histogramArray[index][0] = cv.normalize(h_b , h_b , 0 , 255 , cv.NORM_MINMAX)
            histogramArray[index][1] = cv.normalize(h_g , h_g , 0 , 255 , cv.NORM_MINMAX)
            histogramArray[index][2] = cv.normalize(h_r , h_r , 0 , 255 , cv.NORM_MINMAX)
            
    elif keyWord == 'Gray':
        for img in imgArray:
            h_g = cv.calcHist([img],[0],None,[256],[0,256])
            h_gNormalized = cv.normalize(h_g , h_g , 0 , 255 , cv.NORM_MINMAX)
            
            histogramArray.append(h_gNormalized) 
    return histogramArray

def getMetrics(metricName, reference, sample):
    metric = []
    bestResults = []
    
    for i in range(0,len(sample)):
        sampleImageHistogram = sample[i]
        metric.append([])
        
        for j in range(0,len(reference)):
            referenceImageHistogram = reference[j]
            
            if metricName == 'euclideanDistance':
                metric[i].append(distance.euclidean(sampleImageHistogram, referenceImageHistogram))
            
            elif metricName == 'chiSquare':
                metric[i].append(cv.compareHist(sampleImageHistogram, referenceImageHistogram, cv.HISTCMP_CHISQR))
            
            elif metricName == 'histogramIntersection':
                metric[i].append(cv.compareHist(sampleImageHistogram, referenceImageHistogram, cv.HISTCMP_INTERSECT))
            
            elif metricName == 'bhattacharyya':
                metric[i].append(cv.compareHist(sampleImageHistogram, referenceImageHistogram, cv.HISTCMP_BHATTACHARYYA))
            
            elif metricName == 'correlation':
                metric[i].append(cv.compareHist(sampleImageHistogram, referenceImageHistogram, cv.HISTCMP_CORREL))
            
            elif metricName == 'hellinger':
                metric[i].append(cv.compareHist(sampleImageHistogram, referenceImageHistogram, cv.HISTCMP_HELLINGER))
            
            
    for metric_n in metric:
        bestResults.append(heapq.nlargest(10, range(len(metric_n)), metric_n.__getitem__))
    return bestResults

def main():
    global correlationMetrics, chiSquareMetrics, histogramIntersectionMetrics, bhattacharyyaMetrics,hellingerMetrics, euclideanDistanceMetrics
    
    #define dif paths
    pathDB = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\BBDD\BBDD'
    pathSample = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\qsd1_w1\qsd1_w1'
    
    #get path of images
    jpgImagesPaths = getImagesPaths(pathDB,'.jpg')
    pngImagesPaths = getImagesPaths(pathDB,'.png')
    sampleImagesPaths = getImagesPaths(pathSample,'.jpg')
    
    #get images in gray and RGB
    jpgGrayImages, jpgRGBImages = getGrayImages(jpgImagesPaths)
    pngGrayImages, pngRGBImages = getGrayImages(pngImagesPaths)
    sampleGrayImages, sampleRGBImages = getGrayImages(sampleImagesPaths)
    
    #get hist of gray imgs
    jpgGrayImagesHistogram = getHistogram(jpgGrayImages, 'Gray')
    pngGrayImagesHistogram = getHistogram(pngGrayImages, 'Gray')
    sampleGrayImagesHistogram = getHistogram(sampleGrayImages, 'Gray')
    
    #get histogram of RGB imgs
    jpgRGBImagesHistogram = getHistogram(jpgRGBImages, 'RGB')
    pngRGBImagesHistogram = getHistogram(jpgRGBImages, 'RGB')
    sampleRGBImagesHistogram = getHistogram(sampleRGBImages, 'RGB')
    
    #get metrics
    correlationMetrics = getMetrics('correlation', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    chiSquareMetrics = getMetrics('chiSquare', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    histogramIntersectionMetrics = getMetrics('histogramIntersection', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    bhattacharyyaMetrics = getMetrics('bhattacharyya', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    euclideanDistanceMetrics = getMetrics('euclideanDistance', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    hellingerMetrics = getMetrics('hellinger', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    
    #plot first image
    # plt.subplot(4,1,1)
    # plt.plot(jpgGrayImagesHistogram[120],'Gray')
    # plt.subplot(4,1,2)
    # plt.plot(jpgRGBImagesHistogram[120][0],'Blue')
    # plt.subplot(4,1,3)
    # plt.plot(jpgRGBImagesHistogram[120][1],'Green')
    # plt.subplot(4,1,4)
    # plt.plot(jpgRGBImagesHistogram[120][2],'Red')
    # plt.subplot(4,1,3)
    
    # plt.subplot(4,1,1)
    # plt.plot(sampleGrayImagesHistogram[0],'k')
    # plt.subplot(4,1,2)
    # plt.plot(sampleRGBImagesHistogram[0][0],'c')
    # plt.subplot(4,1,3)
    # plt.plot(sampleRGBImagesHistogram[0][1],'y')
    # plt.subplot(4,1,4)
    # plt.plot(sampleRGBImagesHistogram[0][2],'m')


    # #plot second image
    # plt.subplot(4,1,1)
    # plt.plot(jpgGrayImagesHistogram[170],'Gray')
    # plt.subplot(4,1,2)
    # plt.plot(jpgRGBImagesHistogram[170][0],'Blue')
    # plt.subplot(4,1,3)
    # plt.plot(jpgRGBImagesHistogram[170][1],'Green')
    # plt.subplot(4,1,4)
    # plt.plot(jpgRGBImagesHistogram[170][2],'Red')
    # plt.subplot(4,1,3)
    
    # plt.subplot(4,1,1)
    # plt.plot(sampleGrayImagesHistogram[1],'k')
    # plt.subplot(4,1,2)
    # plt.plot(sampleRGBImagesHistogram[1][0],'c')
    # plt.subplot(4,1,3)
    # plt.plot(sampleRGBImagesHistogram[1][1],'y')
    # plt.subplot(4,1,4)
    # plt.plot(sampleRGBImagesHistogram[1][2],'m')


    # #plot third image
    plt.subplot(4,1,1)
    plt.plot(jpgGrayImagesHistogram[277],'Gray')
    plt.subplot(4,1,2)
    plt.plot(jpgRGBImagesHistogram[277][0],'Blue')
    plt.subplot(4,1,3)
    plt.plot(jpgRGBImagesHistogram[277][1],'Green')
    plt.subplot(4,1,4)
    plt.plot(jpgRGBImagesHistogram[277][2],'Red')
    plt.subplot(4,1,3)
    
    plt.subplot(4,1,1)
    plt.plot(sampleGrayImagesHistogram[2],'k')
    plt.subplot(4,1,2)
    plt.plot(sampleRGBImagesHistogram[2][0],'c')
    plt.subplot(4,1,3)
    plt.plot(sampleRGBImagesHistogram[2][1],'y')
    plt.subplot(4,1,4)
    plt.plot(sampleRGBImagesHistogram[2][2],'m')    
    
    
    
    
if __name__ == "__main__":
    main()    