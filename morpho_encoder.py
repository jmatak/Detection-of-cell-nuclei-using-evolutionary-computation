import random
import morphology_transformation as mt
import structuring_elements as se


def get_random_transform():
    return random.choice(mt.defined_transforms)


def get_random_kernel():
    return se.elements[random.randint(1, 27)]

