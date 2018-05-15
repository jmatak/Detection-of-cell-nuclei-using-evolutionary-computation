import numpy as np
import morphology_transformation as mt
import numpy

"""
Predefinirani strukturni elementi.
"""
elements = {1: np.array([[1, 0], [0, 1]], dtype=np.uint8),
            2: np.array([[0, 1], [1, 0]], dtype=np.uint8),
            3: np.ones((2, 2)),
            4: np.ones((3, 3)),
            5: np.ones((3, 1)),
            6: np.ones((1, 3)),
            7: np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8),
            8: np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            9: np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
            10: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),
            11: np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
                         dtype=np.uint8),
            12: np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
                         dtype=np.uint8),
            13: np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype=np.uint8),
            14: np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                         dtype=np.uint8),
            15: np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
                         dtype=np.uint8),
            16: np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                         dtype=np.uint8),
            17: np.ones((5, 5)),
            18: np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
                         dtype=np.uint8),
            19: np.ones((7, 7)),
            20: np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8),
            21: np.array([[1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8),
            22: np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8),
            23: np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8),
            24: np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8),
            25: np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8),
            26: np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]], dtype=np.uint8),
            27: np.array([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
            }


def toBinaryArrayString(kernel):
    string = ""
    for x in np.nditer(kernel):
        string += str(x)
    return string


def to_string(individual):
    string = "("
    found = False
    for i, array in elements.items():
        if numpy.array_equal(array, individual.kernel):
            string += str(i)
            found = True
            break

    if not found:
        string += toBinaryArrayString(individual.kernel)

    string += ","

    if individual.transformation == mt.dilate:
        string += "D"

    elif individual.transformation == mt.erode:
        string += "E"

    elif individual.transformation == mt.open:
        string += "O"

    elif individual.transformation == mt.close:
        string += "C"

    elif individual.transformation == mt.gradient:
        string += "G"

    elif individual.transformation == mt.top_hat:
        string += "TH"

    elif individual.transformation == mt.black_hat:
        string += "BH"

    string += ")"
    return string
