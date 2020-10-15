import cv2
import os
import numpy as np

# Generic functions
def load_images(folder):
    images = []
    ids = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                for i in range(len(temp1)):
                    id = None
                    if temp1[i] != '0':
                        id = temp1[i:]
                        break
                if id is None:
                    id = '0'
                ids.append(id)
    idx = np.argsort(ids)
    ordIds = np.array(ids)[idx]
    ordImages = np.array(images)[idx]

    return ordImages, ordIds


def sort_ids(metric,idList):
    id = np.argsort(metric)
    sortMetric = np.array(metric)[id]
    sortedIds = np.array(idList)[id]
    return sortMetric, list(map(int, sortedIds))


def load_masks(folder):
    images = []
    ids = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                for i in range(len(temp1)):
                    id = None
                    if temp1[i] != '0':
                        id = temp1[i:]
                        break
                if id is None:
                    id = '0'
                ids.append(id)
    idx = np.argsort(ids)
    ordIds = np.array(ids)[idx]
    ordMasks = np.array(images)[idx]

    return ordMasks, ordIds