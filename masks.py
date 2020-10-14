import cv2 as cv
import numpy as np

def compute_contours(image):
    # Convert image to HSV an use only saturation channel (has most information)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    img_hsv_blur = cv.GaussianBlur(img_hsv[:, :, 1], (5, 5), 0)
    # img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # img = cv.GaussianBlur(img, (5, 5), 0)

    # # global thresholding
    # ret1,th1 = cv.threshold(img,0,128,cv.THRESH_BINARY)

    # Otsu's thresholding
    ret2,thresh2 = cv.threshold(img_hsv_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Get edges using Canny algorithm
    edged = cv.Canny(thresh2, 0, 1)
    # Apply close transformation to eliminate smaller regions
    kernel = np.ones((5,5),np.uint8)
    edged = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# Subtracts de background from the image and returns cropped_img and mask (mask = rectangle)
def compute_mask(image):

    contours = compute_contours(image)

    # If contours not found, pass whole image
    if not contours:
        mask = np.ones(image.shape)
    else:
        # Find the index of the largest contour
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        # Get smallest rectangle from the contour points
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        # Inicialize mask & activate the pixels inside the rectangle
        mask = np.zeros(image.shape[:2],np.uint8)
        mask[y:y+h, x:x+w] = 1

    # Show image & mask
    # cv.imshow("Mask over image", image)
    # cv.imshow("Mask", mask)
    # cv.waitKey(0)

    return mask


# Subtracts de background from the image and returns cropped_img and mask (mask = rectangle)
def compute_croppedimg(image):

    contours = compute_contours(image)

    # If contours not found, pass whole image
    if not contours:
        cropped_img = image
    else:
        # Find the index of the largest contour
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        # Get smallest rectangle from the contour points and crop img
        x,y,w,h = cv.boundingRect(cnt)
        cropped_img = image[y:y+h, x:x+w, :]

    return cropped_img


# Subtracts de background from a list of images and returns a list of masks
def get_mask_array(imgarray):

    masks = []
    for i in imgarray:
        masks.append(compute_mask(i))

    return masks


# returns cropped images
def crop_imgarray(imgarray):

    cropped_imgs = []
    for i in imgarray:
        crpimg = compute_croppedimg(i)
        cropped_imgs.append(crpimg)
        cv.imshow("CroppedImg", crpimg)
        cv.waitKey(0)

    return cropped_imgs



# from utils import *
#
# # Test dataset path
# path = 'D:\MCV\M1\datasets\qsd2_w1'
# q1Im, q1Ids = load_images(path)
# # for img in q1Im:
# #     mask = getmask(img)
# cropbymask(q1Im)
#
# # # Test black image
# # path = "D:\MCV\M1\datasets\\blackimg.png"
# # img = cv2.imread(path)
# # mask = getmask(img)