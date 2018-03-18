import random
import morphology_transformation


def get_random_transform():
    return random.choice(morphology_transformation.defined_transforms)


def get_random_kernel():
    return random.randint(0, 27)
