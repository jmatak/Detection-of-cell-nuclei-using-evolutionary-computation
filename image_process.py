import cv2
import numpy as np
import os
import joblib
import timeit
from scipy.ndimage import label

DEFAULT_FOLDER = 'stage1_train'

#################################
# Učitavanje slika iz memorije  #
#################################

start = timeit.default_timer()
IMAGES = joblib.load('images.dict')
MASKS = joblib.load('masks.dict')
stop = timeit.default_timer()

print('Slike učitane za ' + str(stop - start) + 's !')


#################################
def kmeans(image, K):
    """
    Algoritam za određivanje K dominantnih intenziteta boje unutar crno-bijele slike.

    :param image: Predana slika
    :param K: Broj dominantnih boja
    :return: Polje K dominantnih inteziteta
    """
    Z = np.float32(image.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return np.uint8(center)


def process_image(image, transformations, thresh=True):
    """
    Obrada slike na temelju zadanih informacija, koristi se Otsu treshold metoda
    """
    for t in transformations:
        image = t.transformation(image, t.kernel)

    if thresh:
        image = otsu_treshold(image)

    return image


def otsu_treshold(image):
    """
    Primjena Otsu threshold metode na zadanu sliku
    :param image: Slika kao parametar funkcije
    :return: Slika na koju je primijenjena Otsu treshold metoda
    """
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, treshold_image = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return treshold_image


def watershed(image, thresh=None, transformations=None):
    """
    Implementacije Watershed algoritma na siku. Kao parametar može se unaprijed zadati izračunat Otsu treshold
    nad zadanom slikom. Također kao parametar moguće je predati i transformacijske funckije koje se žele primijeniti
    nad slikom.

    :param image: Slika za analizu
    :param thresh: Moguć unaprijed izračunat Otsu treshold
    :param transformations: Transformacijske funkcije
    :return: Slika nad kojoj je primijenjen Watershed algoritam
    """
    if (transformations != None):
        image = process_image(image, transformations, thresh=False)

    if (thresh == None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Izračun dio koji pripada pozadini
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Dio koji pripada unutrašnjosti
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Neodređeni dio (Razlika prethodna dva)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markiranje sigurnog unutrašnjeg dijela
    ret, markers = cv2.connectedComponents(sure_fg)
    # Zbog pozadine
    markers = markers + 1
    markers[unknown == 255] = 0

    #####################################
    # Izračun broja labeliranih stanica #
    #####################################
    # all, n1 = label(sure_fg)
    # sure_fg = cv2.bitwise_not(sure_fg)
    # all, n2 = label(sure_fg)
    # print("Ima ih : %d" % max(n1, n2))

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
