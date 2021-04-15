import numpy as np


def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    assert (
        array.shape == array_grad.shape
    ), """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(
        tuple(array_grad.shape), tuple(array.shape)
    )
    return None