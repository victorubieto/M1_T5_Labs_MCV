import cv2
import os


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
                id = temp1.split('0')[-1]
                ids.append(id)

    return images, ids

