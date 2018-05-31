import cv2
import random
import morpho_util
import image_process
import copy
import structuring_elements as se
from parameters import *
import numpy as np

cv2.useOptimized()

class MorphologyTransformation:
    """
    Klasa koja definira morfološku trasnformaciju koja se sastoji od Transformacije i njoj pripadnog strukturnog
    elementa.
    """

    def __init__(self, transformation, kernel):
        self.transformation = transformation
        self.kernel = kernel

    def __repr__(self):
        return se.to_string(self)


def erode(image, kernel):
    return cv2.erode(image, kernel)


def dilate(image, kernel):
    return cv2.dilate(image, kernel)


def open(image, kernel, newline='\n'):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def close(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def gradient(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def top_hat(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def black_hat(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def simple_compare(tp, tn, fp, fn):
    return ((tp + tn) / (tp + tn + fn + fp)) * 100


def simple_compare_without_tn(tp, tn, fp, fn):
    return ((tp) / (tp + fn + fp)) * 100


def compare_weighted(tp, tn, fp, fn):
    """
    Funkcija za usporedbu procesuirane slike i njoj pripadne maske. Na procjenu više utječe točno predviđeni pixeli kao
    i pozitivni negativni pixeli

    :return: Postotak preklapanja procesuirane slike i pripremljene maske
    """
    tp *= 10
    fp *= 15
    fn *= 5
    return ((tp) / (tp + fn + fp)) * 100


def compare(processed, mask, no_cells):
    return compare_image(processed, mask) * compare_cells(processed, no_cells)


def compare_cells(processed, no_cells):
    detected_cells = image_process.get_number_of_cells(processed)
    if detected_cells > 2 * no_cells: detected_cells = 0

    det = (detected_cells / no_cells) if detected_cells <= no_cells else 2 - (detected_cells / no_cells)
    return det


def compare_image(processed, mask):
    """
    Funkcija za usporedbu procesuirane slike i njoj pripadne maske.

    :param processed: Procesuirana slika
    :param mask: Pripadna maska
    :return: Postotak preklapanja procesuirane slike i pripremljene maske
    """
    background = image_process.kmeans(processed, 1)
    if background > np.array(128):
        processed = (255 - processed)
    tp, tn, fp, fn = 0, 0, 0, 0
    for p, m in zip(processed, mask):
        for i, j in zip(p, m):
            if not i and not j[0]:
                tn += 1
            elif i and j[0]:
                tp += 1
            elif not i and j[0]:
                fn += 1
            elif i and not j[0]:
                fp += 1

    return compareFunction(tp, tn, fp, fn)



def evaluate(individual):
    """
    Evaluacija jedinke, usporedba s pripadnom maskom.

    :param individual: Jedinka za evaluaciju
    :return: Evaluacija jedinke
    """
    sum = 0
    for i, (name, info) in enumerate(image_process.IMAGES_WITH_INFO.items()):
        if i == TRAIN_NO: break

        image, gray_image, mask, no_cells = info
        processed = image_process.process_image(copy.copy(image), individual)
        sum += compare(processed, mask, no_cells)

    return (sum / TRAIN_NO),


def create(Individual, length):
    """
    Kreiranje jedinke, zadaje se jedinka od N transformacijskih funkcija sa pripadnim transformacijskim matricama.

    :param Individual: Jedinka unutar evolucije
    :param length: Duljina novostvorene jedinke
    :return: Novostvorena jedinka
    """
    individual = Individual()
    for i in range(random.randint(1, length)):
        individual.append(
            MorphologyTransformation(morpho_util.get_random_transform(), randomKernel()))
    return individual


def _get_points(ind):
    """
    Dohvaćanje točaka za križanje unutar jedinke

    :param ind: Zadana jedinka
    :return: Dvije točke presjeka
    """
    cxpoint1 = random.randint(0, len(ind) - 1)
    cxpoint2 = random.randint(0, len(ind) - 1)
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    return cxpoint1, cxpoint2


def cross(ind1, ind2):
    """
    Križanje sa dvije zadane točke presjeka te spajanje jedinki.

    :param ind1: Prva jedinka
    :param ind2: Druga jedinka
    :return: Dvije jedinke koje su nastale križanjem prethodnih
    """
    m1, m2 = _get_points(ind1)
    f1, f2 = _get_points(ind2)

    ind1[m1:m2], ind2[f1:f2] = ind2[f1:f2], ind1[m1:m2]

    return ind1, ind2


def mutate(individual, mutation_chance):
    for t in individual:
        if random.random() < mutation_chance:
            _mutate_transformation(t)
        if random.random() < mutation_chance:
            kernelMutationFunction(t)

    return individual,


def _mutate_transformation(morph):
    """
    Mutacija transformacijskog elementa, zamijeni se nasumičnom transformacijom.

    :param morph: Zadana morfološka transformacija
    :return: Nasumično odabrana transformacij
    """
    morph.transformation = morpho_util.get_random_transform()


def _mutate_kernel(morph):
    """
    Mutacija struturnog elementa, zamijeni se nasumičnim elementom.

    :param morph: Zadana morfološka transformacija
    :return: Nasumično odabrani strukturni element
    """
    morph.kernel = morpho_util.get_random_kernel()


def _mutate_kernel_by_value(morph):
    for e in np.nditer(morph.kernel, op_flags=['readwrite']):
        if random.random() < SE_MUTATION_CHANCE:
            e[...] = 1 - e[...]


# defined_transforms = [dilate, erode, open, close, gradient, top_hat, black_hat]
defined_transforms = [dilate, erode, open, close]
compareFunction = simple_compare_without_tn
kernelMutationFunction = _mutate_kernel
randomKernel = morpho_util.get_random_kernel
