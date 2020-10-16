# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:13:35 2020

@author: mohaa
"""

import cv2 as cv
import glob
from matplotlib import pyplot as plt

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

def getUnchangedImages(imgArray):
    imgUnchangedArray = []
    
    for sample in imgArray:
        imgUnchangedArray.append(cv.imread(sample, cv.IMREAD_UNCHANGED))
    return imgUnchangedArray



def main():
    global jpgGrayImages, jpgRGBImages,unchangedImages
    #define dif paths
    pathDB = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\BBDD\BBDD'
    pathSample = r'C:\Users\mohaa\OneDrive\Escritorio\masterCV\1_M1\1_T1\material\1_images\qsd2_w1\qsd2_w1'
    
    #get path of images
    jpgImagesPaths = getImagesPaths(pathDB,'.jpg')
    pngImagesPaths = getImagesPaths(pathDB,'.png')
    sampleImagesPaths = getImagesPaths(pathSample,'.jpg')
    
    #get images in gray and RGB
    jpgGrayImages, jpgRGBImages = getGrayImages(jpgImagesPaths)
    pngGrayImages, pngRGBImages = getGrayImages(pngImagesPaths)
    sampleGrayImages, sampleRGBImages = getGrayImages(sampleImagesPaths)    
    unchangedImages = getUnchangedImages(sampleImagesPaths)

if __name__ == "__main__":
    main()    