"""Solve PFHub's benchmark 4
"""

# pylint: disable=no-value-for-parameter

import numpy as np
from toolz.curried import pipe, curry, do

from module import solve


@curry
def calc_eta(coords, delta=1.0, radius=2.5):
    """Calculate a fake phase field for testing

    Phase field is a circle centered at 0, 0 of radius r, eta = 1 in
    the circle and 0 outside.

    Args:
      coords: the Sfepy coordinate array
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
    """Calculate the interpolation function

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


@curry
def calc_stiffness(calc_eta_func, coords):
    """Total stiffness tensor

    3 x 3 Stiffness matrix for Sfepy

    Args:
      calc_eta_func: function to calculate the phase field
      coords: the Sfepy coordinate array

    Returns:
      n x 3 x 3 stiffness tensor
    """
    return (
        stiffness_matrix()[None]
        * (1 + 0.1 * calc_h(calc_eta_func(coords)))[:, None, None]
    )


@curry
def calc_prestress(calc_eta_func, coords, epsilon=0.005):
    """Calculate the prestress

    Calculate -h(eta) * [ C_ijkl(eta) * epsilonT_kl ]

    Note that C_1211, C_1222, C_2111 and C_2122 are zero.

    Args:
      calc_eta_func: function to calculate the phase field
      coords: the Sfepy coordinate array
      epsilon: the misfit strain

    Returns:
      n x 3 x 1 stress tensor
    """
    return pipe(
        np.dot(calc_stiffness(calc_eta_func, coords), [epsilon, epsilon, 0]),
        lambda x: -calc_eta_func(coords)[:, None] * x,
        lambda x: np.ascontiguousarray(x[:, :, None]),
    )


def main(shape, dx):
    """Run the calculation

    Args:
      the shape of the domain

    Returns:
      tuple of strain, displacement and stress
    """
    calc_eta_func = calc_eta(delta=dx, radius=shape[0] * dx / 4.)
    return solve(
        calc_stiffness(calc_eta_func), calc_prestress(calc_eta_func), shape, dx=dx
    )


def test():
    """Run some tests
    """
    assert np.allclose(main((10, 10), 1.0)[1][0, 0], [-0.00515589, -0.00515589])
    assert np.allclose(main((10, 10), 0.1)[1][0, 0], [-0.00051559, -0.00051559])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pipe(
        main((50, 50), 0.1)[1],
        lambda x: np.sqrt(np.sum(x ** 2, axis=-1)).swapaxes(0, 1),
        do(plt.imshow),
    )
    plt.colorbar()
    plt.show()
