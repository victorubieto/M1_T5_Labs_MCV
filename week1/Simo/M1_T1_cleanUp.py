# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:11:36 2020

@author: mohaa
"""
import cv2 as cv
import glob
import pickle
from matplotlib import pyplot as plt
import numpy 
import heapq
from scipy.spatial import distance

from scipy.stats import chisquare

import dictances as ds
import ml_metrics as metrics


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


def alignHistograms(reference, sample, numberOfPoints):
    
    tiny = 1e-15
    
    #get the max position of each histogram
    referenceHistogramMaxPositions = []
    sampleHistogramMaxPositions = []
    
    referenceHistogramWithWindow = []
    sampleHistogramWithWindow = []
    
    referenceHistogramWithWindowPlusTiny = []
    sampleHistogramWithWindowPlusTiny = []
    
    for step in reference:
        referenceHistogramMaxPositions.append(int(numpy.where(step == max(step))[0][0]))
    
    for step in sample:
        sampleHistogramMaxPositions.append(int(numpy.where(step == max(step))[0][0]))

    #Apply a rectangular window centred into the max position wit bandwith of numberOfPoints
    for index in range(0,len(reference)):
        step = reference[index]
        maxPosition = referenceHistogramMaxPositions[index]
        # print('INDEX ' + str(maxPosition))
        if (maxPosition - numberOfPoints) < 0:
            # print('A')
            referenceHistogramWithWindow.append(step[0:2*numberOfPoints])
            # print('AA')
        elif (maxPosition + numberOfPoints) > 255:
            # print('B')
            referenceHistogramWithWindow.append(step[255 - 2*numberOfPoints:255])
            # print('BB')
        
        else:
            # print('C')
            referenceHistogramWithWindow.append(step[maxPosition - numberOfPoints:maxPosition + numberOfPoints])
            # print('CC')

        
    for index in range(0,len(sample)):
        step = sample[index]
        maxPosition = sampleHistogramMaxPositions[index]
        # print('INDEX SAMPLE ' + str(maxPosition))
        if (maxPosition - numberOfPoints) < 0:
            # print('A')
            sampleHistogramWithWindow.append(step[0:2*numberOfPoints])
            # print('AA')
        
        elif (maxPosition + numberOfPoints) > 255:
            # print('B')
            sampleHistogramWithWindow.append(step[255 - 2*numberOfPoints:255])
            # print('BB')
        
        else:
            # print('C')
            sampleHistogramWithWindow.append(step[maxPosition - numberOfPoints:maxPosition + numberOfPoints])
            # print('CC')
            
        # if index == 28:
            # print('POSICION 28: ' + str(step[maxPosition - numberOfPoints:maxPosition + numberOfPoints]))
            # print('MAXPOSICION 28: ' + str(maxPosition))
            # print('NUMBEROFPOINTS 28: ' + str(numberOfPoints))
    # print('LONGITUD = '+str(referenceHistogramWithWindow))
    #Add a little offset to delete the zero value
    for pos in range(0,len(referenceHistogramWithWindow)):
        
        # print('CACA ->' + str(referenceHistogramWithWindow[pos]))
        referenceHistogramWithWindowPlusTiny.append([])
        # print('RR')
        step = referenceHistogramWithWindow[pos]
        for index in range(0,len(step)):
            referenceHistogramWithWindowPlusTiny[pos].append(float(referenceHistogramWithWindow[pos][index] + tiny))
        
    
    for pos in range(0,len(sampleHistogramWithWindow)):
        sampleHistogramWithWindowPlusTiny.append([])
        step = sampleHistogramWithWindow[pos]
        for index in range(0,len(step)):
            sampleHistogramWithWindowPlusTiny[pos].append(float(sampleHistogramWithWindow[pos][index] + tiny))
        
        
    # print('MAX POSITION REF: ' + str(max(referenceHistogramMaxPositions)))  
    # print('MIN POSITION REF: ' + str(min(referenceHistogramMaxPositions)))
    
    # print('MAX POSITION SAMPLE: ' + str(max(sampleHistogramMaxPositions)))  
    # print('MIN POSITION SAMPLE: ' + str(min(sampleHistogramMaxPositions)))
    
    
    return referenceHistogramWithWindowPlusTiny,sampleHistogramWithWindowPlusTiny

    

def getMetrics(metricName, reference, sample):
    metric = []
    bestResults = []
    metricValues = []
    
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
            
    
    
    
    if metricName == 'testingCorrelation':
        metric = []
        for sampleHistIndex in range(0,len(sample)):
            metric.append([])
            
            sampleHist = sample[sampleHistIndex]
            
            for referenceHistIndex in range(0, len(reference)):

                referenceHist = reference[referenceHistIndex]

                corrSampleReference = numpy.corrcoef(sampleHist,referenceHist)[0][1]
                metric[sampleHistIndex].append(corrSampleReference)
    
    if metricName == 'testingChiSquare':
        metric = []
        for sampleHistIndex in range(0,len(sample)):
            metric.append([])
            sampleHist = sample[sampleHistIndex]
            
            for referenceHistIndex in range(0, len(reference)):
                referenceHist = reference[referenceHistIndex]
                chiSquareSampleReference = chisquare(sampleHist,referenceHist)[1]    
                metric[sampleHistIndex].append(chiSquareSampleReference)
   
    if metricName == 'testingEuclideanDistance':
        metric = []
        for sampleHistIndex in range(0,len(sample)):
            metric.append([])
            sampleHist = sample[sampleHistIndex]
            
            for referenceHistIndex in range(0, len(reference)):
                referenceHist = reference[referenceHistIndex]
                euclideanDist = distance.euclidean(sampleHist,referenceHist)
                metric[sampleHistIndex].append(euclideanDist)
    
    # if metricName == 'testingBhattacharyya':
    #     metric = []
    #     for sampleHistIndex in range(0,len(sample)):
    #         metric.append([])
    #         sampleHist = sample[sampleHistIndex]
            
    #         for referenceHistIndex in range(0, len(reference)):
    #             referenceHist = reference[referenceHistIndex]
                
    #             ds.bhattacharyya(a, b)
    

    
        
    for metric_n in metric:
        bestResults.append(heapq.nlargest(10, range(len(metric_n)), metric_n.__getitem__))
    
    for index in range(0,len(bestResults)):
        metricValues.append([])
        bestResultsOfThisIteration = bestResults[index]
        for imageIndex in bestResultsOfThisIteration:
            metricValues[index].append(metric[index][imageIndex])
    
    return bestResults,metricValues

def getResults(metrics,keyWord):
    global data
    with open(r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\qsd1_w1\qsd1_w1\gt_corresps.pkl', 'rb') as f:
        data = pickle.load(f)
    
    detected = 0
    
    # print('**************************************')
    # print('Images detected with ' + str(keyWord) + ' metric')
    for i in range(0,len(metrics)):
        if data[i][0] in metrics[i]:
            detected = detected + 1
            # print('    -> Image number ' +str(i) + ' detected between the 10 best values')
            
    # print('    TOTAL IMAGES DETECTED = ' + str(detected) + "\n\n\n")
    # print('**************************************')
    return detected

def getOptimiziedMetricForHistogramsWithWindow(reference,sample):
    global windowSize,correlationMetricsTestingIDs,chiSquareMetricsTestingIDs,euclideanDistanceMetricsTestingIDs,correlationMetricsTestingValues,chiSquareMetricsTestingValues,euclideanDistanceMetricsTestingValues
    
    windowSize = []
    chiSquareDetectedImages = []
    correlationDetectedImages = []
    euclideanDistanceDetectedImages = []
    
    
    mAPkChiSquareWindow = []
    mAPkCorrelationWindow = []
    mAPkEuclideanDistanceWindow = []
    
    #size in samples
    minWindowSize = 1
    maxWindowSize = 128
    
    for windowSamples in range(minWindowSize,maxWindowSize):
        #align histograms test
        print('Step ' + str(windowSamples))
        referenceHistogramWithWindoww,sampleHistogramWithWindoww = alignHistograms(reference, sample, windowSamples)
        
        #get testing metrics
        correlationMetricsTestingIDs,correlationMetricsTestingValues = getMetrics('testingCorrelation', referenceHistogramWithWindoww, sampleHistogramWithWindoww)
        chiSquareMetricsTestingIDs,chiSquareMetricsTestingValues = getMetrics('testingChiSquare', referenceHistogramWithWindoww, sampleHistogramWithWindoww)
        euclideanDistanceMetricsTestingIDs,euclideanDistanceMetricsTestingValues = getMetrics('testingEuclideanDistance', referenceHistogramWithWindoww, sampleHistogramWithWindoww)
        
        windowSize.append(windowSamples)
        chiSquareDetectedImages.append(getResults(correlationMetricsTestingIDs,'testingChiSquare'))
        correlationDetectedImages.append(getResults(chiSquareMetricsTestingIDs,'testingCorrelation'))
        euclideanDistanceDetectedImages.append(getResults(euclideanDistanceMetricsTestingIDs,'euclideanDistance'))
        
        mAPkChiSquareWindow.append(getMAPk(chiSquareMetricsTestingIDs,10))
        mAPkCorrelationWindow.append(getMAPk(correlationMetricsTestingIDs,10))
        mAPkEuclideanDistanceWindow.append(getMAPk(euclideanDistanceMetricsTestingIDs,10))
        
    return windowSize,chiSquareDetectedImages,correlationDetectedImages,euclideanDistanceDetectedImages,mAPkChiSquareWindow,mAPkCorrelationWindow,mAPkEuclideanDistanceWindow

def getMAPk(query,k):
    global data
    mAPk =  metrics.mapk(data, query ,k)
    return mAPk

def main():
    global correlationMetrics, sampleGrayImagesHistogram, chiSquareMetrics, histogramIntersectionMetrics, bhattacharyyaMetrics,hellingerMetrics, euclideanDistanceMetrics, jpgGrayImagesHistogram, referenceHistogramWithWindoww,sampleHistogramWithWindoww,correlationMetricsTesting,chiSquareMetricsTesting,histogramIntersectionMetricsTesting,bhattacharyyaMetricsTesting,euclideanDistanceMetricsTesting,hellingerMetricsTesting

    
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
    
    #get metrics of the histograms without windowing
    correlationMetricsIDs, correlationMetricsValues = getMetrics('correlation', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    chiSquareMetricsIDs, chiSquareMetricsValues = getMetrics('chiSquare', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    histogramIntersectionMetricsIDs, histogramIntersectionMetricsValues = getMetrics('histogramIntersection', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    bhattacharyyaMetricsIDs,bhattacharyyaMetricsValues = getMetrics('bhattacharyya', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    euclideanDistanceMetricsIDs,euclideanDistanceMetricsValues = getMetrics('euclideanDistance', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
    hellingerMetricsIDs, hellingerMetricsValues = getMetrics('hellinger', jpgGrayImagesHistogram, sampleGrayImagesHistogram)
 
    
    # getResults(correlationMetrics,'correlation')
    # getResults(chiSquareMetrics,'chiSquare')
    # getResults(histogramIntersectionMetrics,'histogramIntersection')
    # getResults(bhattacharyyaMetrics,'bhattacharyya')
    # getResults(euclideanDistanceMetrics,'euclideanDistance')
    # getResults(hellingerMetrics,'hellinger') 
    

    global xAxe,chiSquareAxe,correlationAxe,euDistAxe,mAPkChiSquareWindow,mAPkCorrelationWindow,mAPkEuclideanDistanceWindow
    #get the best metric with the best window size for our gray level histograms
    xAxe,chiSquareAxe,correlationAxe,euDistAxe,mAPkChiSquareWindow,mAPkCorrelationWindow,mAPkEuclideanDistanceWindow = getOptimiziedMetricForHistogramsWithWindow(jpgGrayImagesHistogram,sampleGrayImagesHistogram)
    
    global figure1, figure2
    
    figure1 = plt.figure(1)
    
    plt.plot(xAxe, euDistAxe , color = 'b', label = 'euclideanDistanceWithWindow')
    plt.plot(xAxe, correlationAxe, color = 'r', label = 'correlationWithWindow')
    plt.plot(xAxe, chiSquareAxe, color = 'g', label = 'chiSquareWithWindow')
    plt.legend(loc=4)
    
    figure2 = plt.figure(2)
    plt.plot(xAxe, mAPkEuclideanDistanceWindow , color = 'b', label = 'euclideanDistanceMAPk')
    plt.plot(xAxe, mAPkCorrelationWindow, color = 'r', label = 'correlationWithMAPk')
    plt.plot(xAxe, mAPkChiSquareWindow, color = 'g', label = 'chiSquareWithMAPk')
    plt.legend(loc=4)
    
    
    # for windowSamples in range(3,128):
    # #align histograms test
    #     print('Step ' + str(windowSamples))
    #     referenceHistogramWithWindoww,sampleHistogramWithWindoww = alignHistograms(jpgGrayImagesHistogram, sampleGrayImagesHistogram, windowSamples)
    
    #     #get testing metrics
    #     correlationMetricsTesting = getMetrics('testingCorrelation', referenceHistogramWithWindoww, sampleHistogramWithWindoww)
    #     chiSquareMetricsTesting = getMetrics('testingChiSquare', referenceHistogramWithWindoww, sampleHistogramWithWindoww)
        
    #     # getResults(correlationMetricsTesting,'testingCorrelation')
    #     # getResults(chiSquareMetricsTesting,'testingChiSquare')
    
    #     windowSize.append(windowSamples)
    #     chiSquareDetectedImages.append(getResults(chiSquareMetricsTesting,'testingChiSquare'))
    #     correlationDetectedImages.append(getResults(correlationMetricsTesting,'testingCorrelation'))
    

    
    
    
    # plt.subplot(4,1,1)
    # plt.plot(referenceHistogramWithWindoww[120],'Gray')
    # plt.subplot(4,1,1)
    # plt.plot(sampleHistogramWithWindoww[0],'Red')
    
    # plt.subplot(4,1,2)
    # plt.plot(referenceHistogramWithWindoww[170],'Gray')
    # plt.subplot(4,1,2)
    # plt.plot(sampleHistogramWithWindoww[1],'Red')
    
    # plt.subplot(4,1,3)
    # plt.plot(referenceHistogramWithWindoww[277],'Gray')
    # plt.subplot(4,1,3)
    # plt.plot(sampleHistogramWithWindoww[2],'Red')
    
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
    # plt.subplot(4,1,1)
    # plt.plot(jpgGrayImagesHistogram[277],'Gray')
    # plt.subplot(4,1,2)
    # plt.plot(jpgRGBImagesHistogram[277][0],'Blue')
    # plt.subplot(4,1,3)
    # plt.plot(jpgRGBImagesHistogram[277][1],'Green')
    # plt.subplot(4,1,4)
    # plt.plot(jpgRGBImagesHistogram[277][2],'Red')
    # plt.subplot(4,1,3)
    
    # plt.subplot(4,1,1)
    # plt.plot(sampleGrayImagesHistogram[2],'k')
    # plt.subplot(4,1,2)
    # plt.plot(sampleRGBImagesHistogram[2][0],'c')
    # plt.subplot(4,1,3)
    # plt.plot(sampleRGBImagesHistogram[2][1],'y')
    # plt.subplot(4,1,4)
    # plt.plot(sampleRGBImagesHistogram[2][2],'m')    
    
    
    
    
if __name__ == "__main__":
    main()    