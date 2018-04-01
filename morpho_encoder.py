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
    return se.elements[random.randint(1, len(se.elements))]


def get_transform(index):
    """
    Dohvaća transformaciju na određenoj poziciji definiranih transformacija
    :param index: Index transformacije
    :return: Transformacija na poziciji index
    """
    return mt.defined_transforms[index]


def get_kernel(index):
    """
    Dohvaća strukturni element na određenoj poziciji definiranih elemenata
    :param index: Index elementa
    :return: Element na poziciji index
    """
    return se.elements[index]
