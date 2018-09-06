"""Solve PFHub's benchmark 4
"""

import numpy as np
from toolz.curried import pipe

from module import ElasticFESimulation


def calc_eta(coords, delta=1.0, radius=2.5):
    """Calculate a fake phase field for testing

    Phase field is a circle centered at 0, 0 of radius r, eta = 1 in
    the circle and 0 outside.

    Args:
      coords: x-y coordinates
      delta: interface width
      radius: radius of the circle

    Returns:
      the value of the phase field
    """
    return pipe(
        coords,
        lambda x: np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2),
        lambda x: 0.5 * (1 + np.tanh((-x + radius) * 2 / delta)),
    )


def calc_h(eta):
    """Calculation the interpolation function

    Args:
      eta: the phase field

    Returns:
      the value of h
    """
    return eta ** 3 * (6 * eta ** 2 - 15 * eta + 10)


def stiffness_matrix(c11=250, c12=150, c44=100):
    """Stiffness tensor in the matrix phase

    Args:
      c11: component C_11 = C_22 -> C_1111 = C_2222
      c12: component C_12 = C_21 -> C_1122 = C_2211
      c44: component C_44 = C_55 = C_66 -> C_1212 = C_2121 = C_2112 = C_1221

    Returns:
      3 x 3 stiffness tensor in the matrix phase
    """
    return np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c44]])


def calc_stiffness(coords):
    """Total stiffness tensor

    Args:
      coords: space coordinates

    Returns:
      3 x 3 stiffness tensor
    """
    return (
        stiffness_matrix()[None] * (1 + 0.1 * calc_h(calc_eta(coords)))[:, None, None]
    )


def main(shape):
    """Run the calculation

    Args:
      the shape of the domain

    Returns:
      tuple of strain, displacement and stress
    """
    return ElasticFESimulation(macro_strain=0.1).run(calc_stiffness, shape)


def test():
    """Run some tests
    """
    _, displacement, _ = main((10, 10))
    macro_strain = 0.1
    assert np.allclose(
        displacement[-1, :, 0] - displacement[0, :, 0], 10 * macro_strain
    )
    assert np.allclose(displacement[0, :, 1], displacement[-1, :, 1])
    assert np.allclose(displacement[:, 0], displacement[:, -1])


if __name__ == "__main__":
    test()
