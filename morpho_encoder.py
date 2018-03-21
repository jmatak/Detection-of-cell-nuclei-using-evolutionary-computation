import random
import morphology_transformation as mt
import structuring_elements as se


def get_random_transform():
    """
    Dohvati nasumičnu transformaciju iz liste definiranih
    :return: Nasumična transformacija
    """
    return random.choice(mt.defined_transforms)


def get_random_kernel():
    """
    Dohvati nasumičnu transformacijsku matricu.
    :return: Nasumična transformacijska matrica
    """
    return se.elements[random.randint(1, 27)]

