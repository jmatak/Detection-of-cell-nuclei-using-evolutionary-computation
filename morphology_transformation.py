import cv2
import random
import morpho_encoder
import image_process
import copy
import structuring_elements as se

TRAIN_NO = 100


class MorphologyTransformation:
    def __init__(self, transformation, kernel):
        self.transformation = transformation
        self.kernel = kernel

    def __repr__(self):
        return se.to_string(self)


def erode(image, kernel):
    return cv2.erode(image, kernel, iterations=1)


def dilate(image, kernel):
    return cv2.dilate(image, kernel, iterations=1)


def open(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def close(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def gradient(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def top_hat(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def black_hat(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def compare(processed, mask):
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

    return ((tp + tn) / (tp + tn + fn + fp)) * 100


def evaluate(individual):
    sum = 0
    for i, (name, img) in enumerate(image_process.IMAGES.items()):
        if i == TRAIN_NO:
            break

        processed = image_process.process_image(copy.copy(img), individual)
        mask = image_process.MASKS[name]

        sum += compare(processed, mask)

    return (sum / TRAIN_NO),


def create(Individual, length):
    individual = Individual()
    for i in range(random.randint(1, length)):
        individual.append(
            MorphologyTransformation(morpho_encoder.get_random_transform(), morpho_encoder.get_random_kernel()))
    return individual


def _get_points(ind):
    cxpoint1 = random.randint(0, len(ind))
    cxpoint2 = random.randint(0, len(ind))
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    return cxpoint1, cxpoint2


def cross(ind1, ind2):
    points1 = _get_points(ind1)
    points2 = _get_points(ind2)

    ind1[points1[0]:points1[1]], ind2[points2[0]:points2[1]] \
        = ind2[points2[0]:points2[1]], ind1[points1[0]:points1[1]]

    return ind1, ind2


def mutate(individual, mutation_chance):
    for t in individual:
        if random.random() < mutation_chance:
            _mutate_transformation(t)
        if random.random() < mutation_chance:
            _mutate_kernel(t)

    return individual,


def _mutate_transformation(morph):
    morph.transformation = morpho_encoder.get_random_transform()


def _mutate_kernel(morph):
    morph.kernel = morpho_encoder.get_random_kernel()


defined_transforms = [dilate, erode, open, close, gradient, top_hat, black_hat]
