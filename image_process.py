import cv2
import numpy as np
import os
import joblib
import timeit
from scipy.ndimage import label

DEFAULT_FOLDER = 'stage1_train'

start = timeit.default_timer()
IMAGES = joblib.load('images.dict')
MASKS = joblib.load('masks.dict')
stop = timeit.default_timer()

print('Slike učitane za ' + str(stop - start) + 's !')


def process_image(image, transformations, thresh=True):
    """
    Obrada slike na temelju zadanih informacija, koristi se Otsu treshold metoda
    """
    for t in transformations:
        image = t.transformation(image, t.kernel)

    if thresh:
        img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, imgf = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = imgf

    return image


def watershed(image, thresh=None, transformations=None):
    if (transformations != None):
        image = process_image(image, transformations, thresh=False)

    if (thresh == None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    all, n1 = label(sure_fg)
    sure_fg = cv2.bitwise_not(sure_fg)
    all, n2 = label(sure_fg)
    print("Ima ih : %d" % max(n1,n2))

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 255]
    return image


def read_img(directory):
    """
    Funkcija za dohvaćanje slike iz direktorija.
    """
    return cv2.imread(os.path.join(directory, os.listdir(directory)[0]))


def read_mask(directory):
    """
    Funkcija za spajanje svih maski unutar jednog direktorija.
    """
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = cv2.imread(mask_path)
        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)
    return mask


def init():
    """
    Preprocesiranje slika, učitavanje u memoriju prije pokretanja programa.
    """
    folders = os.listdir('stage1_train')
    length = len(folders)
    for i, v in enumerate(folders):
        mask_folder = os.path.join(DEFAULT_FOLDER, os.path.join(v, 'masks'))
        img_folder = os.path.join(DEFAULT_FOLDER, os.path.join(v, 'images'))

        IMAGES[v] = read_img(img_folder)
        MASKS[v] = read_mask(mask_folder)

        print('Processed : ' + str((i + 1) / length))

    joblib.dump(IMAGES, 'images.dict', compress=3)
    joblib.dump(MASKS, 'masks.dict', compress=3)


if __name__ == '__main__':
    init()
