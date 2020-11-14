from textDetection import *
from descriptors import *
from masks import *
from rotation import *
from keypoints import *
import numpy as np
import matplotlib.pyplot as plt
import ml_metrics as metrics


# Params that should be modified
pathQ5 = 'qsd1_w5'      # Path to the query images
pathM = 'BBDD'          # Path to the Database
k = 10                  # Number of bests predictions
kpDescriptor = 'ORB'    # ORB or SIFT

# Load the list of images
print("loading images...")
listQImgs, idsQImgs = load_images(pathQ5)
listMImgs, idsMImgs = load_images(pathM)

# Load the list of the authors of the paintings sorted with idsMImgs
listMAuthors, idsMAuthors = load_authors(pathM)

# Load the GT of the text boundary boxes, if exist
# if os.path.exists(os.path.join(pathQ5,'text_boxes.pkl')):
#     with open(os.path.join(pathQ5,'text_boxes.pkl'), 'rb') as f:
#         gtBbox = pickle.load(f)
# else:
gtBbox = None
allIou = []

# For each museum image compute keypoint and descriptor
orb = cv.ORB_create()
sift = cv.SIFT_create()
bf = cv.BFMatcher()
Mkp = []
Mdes = []
print("computing keypoints...")
for i in range(0,len(listMImgs)):
    if kpDescriptor == 'ORB':
        gray = cv.cvtColor(listMImgs[i], cv.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
    elif kpDescriptor == 'SIFT':
        gray = cv.cvtColor(listMImgs[i], cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
    else:
        print('Please entera a valid method name')
        kp = None
        des = None
    Mkp.append(kp)
    Mdes.append(des)

# Rotate the images accordingly
print("correcting rotation...")
rotated_images, angles = rotate_imgs(listQImgs)

# Crop images for each painting
print("cropping images...")
cropped_images, coords_images = crop_imgarray(rotated_images)

# Init lists of the results
results = []
text_bbox = []
frame_bbox = []

# Create a folder to store the name of the authors
dir = 'authors'
if not os.path.exists(dir):
    os.mkdir(dir)

# --- Loop for all images of the Query ---
print("matching  retrieval...")
for i in range(0,len(cropped_images)):

    img = cropped_images[i]
    coords = coords_images[i]
    ang = angles[i]
    print("Finding Best Match for Image  " + str(i))    # counter for debugging purposes

    # Save painter name to txt file
    f = open('authors/000' + str(i) + '.txt', "a+")

    results_painting = []
    painting_bbox = []
    txt_bbox = []
    for j in range(len(img)):  # loop for each individual painting from one image
        painting = img[j]
        coord = coords[j]

        # Compute the denoising filter for each image channel
        denoised_img = np.zeros((np.size(painting, 0), np.size(painting, 1), np.size(painting, 2)))
        denoised_img[:, :, 0] = cv.medianBlur(painting[:, :, 0],3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 1] = cv.medianBlur(painting[:, :, 1],3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 2] = cv.medianBlur(painting[:, :, 2],3)  # Median is the best effective for salt and pepper noise
        denoised_img = denoised_img.astype(np.uint8)

        # Calculate the bbox of the text
        gtBox_i = None
        if gtBbox is not None:
            gtBox_i = gtBbox[i]
        iou_, [x, y, w, h] = calculate_txtbox(denoised_img, gtBox_i)
        if iou_ is not None:
            allIou.append(iou_)

        Bbox = [x, y, x + w, y + h]
        txt_bbox.append(Bbox)  # Add the results list

        # Ignore the text box
        mask = np.ones(denoised_img.shape[:2], dtype="uint8")
        mask[y:y + h, x:x + w] = 0
        mask = mask * 255  # ones have to be 255 to work with ORB
        mask = np.uint8(mask)

        ## UNCOMMENT WHEN USING TEXT DESCRIPTOR
        # Text descriptor, returns the ids sorted by the distance
        idsSorted, max_paintings, name = text_descriptor(denoised_img, Bbox, listMAuthors, idsMImgs)
        # Save painter name to txt file
        f.write("%s\n" % name)  # write the name to a txt

        # ORB descriptor only to the top images, returns the ids sorted by the distance
        best_idsMImgs = idsSorted[:max_paintings]
        best_listMImgs = []
        best_Mkp = []
        best_Mdes = []
        for j in range(0, max_paintings):
            id = idsSorted[j]
            best_Mkp.append(Mkp[id])
            best_Mdes.append(Mdes[id])
            best_listMImgs.append(listMImgs[id])

        ## COMMENT WHEN USING TEXT DESCRIPTOR
        # best_idsMImgs = idsMImgs
        # best_Mkp = Mkp
        # best_Mdes = Mdes
        # best_listMImgs = listMImgs
        #
        # if kpDescriptor == 'ORB':
        #     isValidPainting = 1
        #     gray = cv.cvtColor(denoised_img, cv.COLOR_BGR2GRAY)
        #     best_idsSorted = orb_descriptor(gray, mask, best_Mkp, best_Mdes, listMImgs, best_idsMImgs)
        # elif kpDescriptor == 'SIFT':
        #     isValidPainting = 1
        #     gray = cv.cvtColor(denoised_img, cv.COLOR_BGR2GRAY)
        #     best_idsSorted = sift_descriptor(gray, mask, best_Mkp, best_Mdes, listMImgs, best_idsMImgs)

        ## UNCOMMENT WHEN USING TEXT DESCRIPTOR
        # Complete the list of k bests
        if max_paintings < k:
            best_idsSorted = best_idsSorted + idsSorted[max_paintings:k]
        else:
            best_idsSorted[:k]

        # Only store the k bests
        kList = list(best_idsSorted)
        kList = kList[:k]

        painting_Bbox = [ang, coord]
        painting_bbox.append(painting_Bbox)  # Add the results list
        results_painting.append(kList)  # painting list

    frame_bbox.append(painting_bbox)
    text_bbox.append(txt_bbox)  # final bbox list
    results.append(results_painting)  # final img list

    # Save painters names to txt file
    f.close()

## -----------------------------------------------------------------------------------------------
# Store the bounding boxes, results and find the MAP@K metric

# TEXT BOXES

# pkl_path = 'text_boxes.pkl'
# outfile = open(pkl_path, 'wb')
# pickle.dump(bbox, outfile)
# outfile.close()

# PAINTING BOXES + ANGLES

# gt_file = open("qsd1_w5/frames.pkl",'rb') # it assumes that you are using qsd1_w4 in order to compute the MAP
# gtquery_list = pickle.load(gt_file)
# gt_file.close()
#
# # Correct predictions array to compute correctly the MAPKScore
# for i in range(0,len(frame_bbox)):
#     aux1 = len(gtquery_list[i])
#     aux2 = len(frame_bbox[i])
#     if aux1 < aux2:
#         for j in range(aux2 - aux1):
#             frame_bbox[i].pop(-1)
#     elif aux2 > aux1:
#         for j in range(aux1 - aux2):
#             frame_bbox[i].append(np.zeros[k])
#
# pkl_path = "framesEst.pkl"
# outfile = open(pkl_path, 'wb')
# pickle.dump(frame_bbox, outfile)
# outfile.close()

# MATCHING RESULTS

pkl_path = "result.pkl"
outfile = open(pkl_path, 'wb')
pickle.dump(results, outfile)
outfile.close()

gt_file = open("qsd1_w5/gt_corresps.pkl",'rb') # it assumes that you are using qsd1_w4 in order to compute the MAP
gtquery_list = pickle.load(gt_file)
gt_file.close()
gtquery_list_corrected = transformGT(gtquery_list)

# Correct predictions array to compute correctly the MAPKScore
for i in range(0,len(results)):
    aux1 = len(gtquery_list[i])
    aux2 = len(results[i])
    if aux1 < aux2:
        for j in range(aux2 - aux1):
            results[i].pop(-1)
    elif aux2 > aux1:
        for j in range(aux1 - aux2):
            results[i].append(np.zeros[k])

res = resultMAP(results)
mapkScore = metrics.mapk(gtquery_list_corrected, res, 5)
print('MAP@5: ' + str(mapkScore))

mapkScore = metrics.mapk(gtquery_list_corrected, res, 1)
print('MAP@1: ' + str(mapkScore))
