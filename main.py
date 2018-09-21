"""Solve PFHub's benchmark 4
"""

# pylint: disable=no-value-for-parameter

import numpy as np
from toolz.curried import pipe, curry
from toolz.curried import map as map_

from sfepy_module import solve as sfepy_solve
from fipy_module import solve as fipy_solve


def params():
    """Dictionary of base parameters

    Returns:
      parameter dictionary
    """
    return dict(
        lx=200.0,
        nx=200,
        radius=20.0,
        kappa=0.29,
        mobility=5.0,
        eta0=0.0065,
        max_iter=5,
        C11=250.0,
        C12=150.0,
        C44=100.0,
        delta=0.0,
        misfit_strain=0.005,
    )


def calc_h(eta):
    """Calculate the interpolation function

    Args:
      eta: the phase field

    Returns:
      the value of h

    """
    return eta ** 3 * (6 * eta ** 2 - 15 * eta + 10)


def calc_dh(eta):
    """Calculate the first derivative of the interpolation function

    Args:
      eta: the phase field

    Returns:
      the value of dh
    """
    return 30 * eta ** 2 * (eta - 1) ** 2


def calc_d2h(eta):
    """Calculate the second derivative of the interpolation function

    Args:
      eta: the phase field

    Returns:
      the value of d2h
    """
    return 60 * eta * (2 * eta - 1) * (eta - 1)


def calc_elastic_f():
    """Elastic free energy in the matrix
    """
    c_11 = 250
    c_12 = 150
    epsilon = 0.005
    return 2 * epsilon ** 2 * (c_11 + c_12)


def calc_elastic_d2f(eta):
    """Calculate the second derivative of the elastic free energy
    """
    return (calc_dh(eta) ** 2 + calc_h(eta) * calc_d2h(eta)) * calc_elastic_f()


def calc_chem_d2f(eta):
    """Calculate the second derivative of the chemical free energy
    """

    def a_j(j):
        """The a_j coefficients from the spec
        """
        return [
            0,
            0,
            8.072789087,
            -81.24549382,
            408.0297321,
            -1244.129167,
            2444.046270,
            -3120.635139,
            2506.663551,
            -1151.003178,
            230.2006355,
        ][j]

    def calc_term(j):
        """Calculate a singe term in the free energy sum
        """
        return a_j(j) * j * (j - 1) * eta ** (j - 2)

    return 0.1 * sum(map_(calc_term, range(2, 11)))


def calc_d2f(eta):
    """Calculate the second derivative of the total free energy
    """
    return calc_elastic_d2f(eta) + calc_chem_d2f(eta)


def stiffness_matrix(params_):
    """Stiffness tensor in the matrix phase

    Args:
      c11: component C_11 = C_22 -> C_1111 = C_2222
      c12: component C_12 = C_21 -> C_1122 = C_2211
      c44: component C_44 = C_55 = C_66 -> C_1212 = C_2121 = C_2112 = C_1221

    Returns:
      3 x 3 stiffness tensor in the matrix phase
    """
    return np.array(
        [
            [params_["C11"], params_["C12"], 0],
            [params_["C12"], params_["C11"], 0],
            [0, 0, params_["C44"]],
        ]
    )


@curry
def calc_stiffness(params_, calc_eta_func, coords):
    """Total stiffness tensor

    3 x 3 Stiffness matrix for Sfepy

    Args:
      calc_eta_func: function to calculate the phase field
      coords: the Sfepy coordinate array

    Returns:
      n x 3 x 3 stiffness tensor
    """
    return (
        stiffness_matrix(params_)[None]
        * (1 + params_["delta"] * calc_h(calc_eta_func(coords)))[:, None, None]
    )


@curry
def calc_prestress(params_, calc_eta_func, coords):
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
        params_["misfit_strain"],
        lambda x: np.dot(calc_stiffness(params_, calc_eta_func, coords), [x, x, 0]),
        lambda x: -calc_eta_func(coords)[:, None] * x,
        lambda x: np.ascontiguousarray(x[:, :, None]),
    )


def run_fipy_to_sfepy(params_):
    """Run the calculation

    Returns:
      tuple of strain, displacement and stress
    """
    # eta = fipy_solve(params(), calc_d2f)["eta"]

    # @curry
    # def calc_eta_func(eta, coords):
    #     """Calculate phase field given Sfepy coords
    #     """
    #     return eta(coords.

    @curry
    def interpolate(var, coords):
        """Interpolate from a variable to given coords
        """
        return var(coords.swapaxes(0, 1), order=1)

    return pipe(
        fipy_solve(params_, calc_d2f)["eta"],
        interpolate,
        lambda x: sfepy_solve(
            calc_stiffness(params_, x),
            calc_prestress(params_, x),
            (params_["nx"], params_["nx"]),
            params_["lx"] / params_["nx"],
        ),
    )
