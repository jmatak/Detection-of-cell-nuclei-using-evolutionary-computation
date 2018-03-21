import cv2
import numpy as np
import os
import joblib
import timeit

DEFAULT_FOLDER = 'stage1_train'

start = timeit.default_timer()
IMAGES = joblib.load('images.dict')
MASKS = joblib.load('masks.dict')
stop = timeit.default_timer()

print('Slike učitane za ' + str(stop - start) + 's !')


def process_image(image, transformations):
    """
    Obrada slike na temelju zadanih informacija, koristi se Otsu treshold metoda
    """
    for t in transformations:
        image = t.transformation(image, t.kernel)

    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, imgf = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


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