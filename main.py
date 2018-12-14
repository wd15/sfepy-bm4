"""Solve PFHub's benchmark 4
"""

# pylint: disable=no-value-for-parameter

import numpy as np
from toolz.curried import pipe, curry, valmap, dissoc, assoc, get
from toolz.curried import map as map_

from sfepy_module import solve as sfepy_solve
from fipy_module import solve as fipy_solve
from fipy_module import to_face_value, iterate_, view
from elastic import calc_elastic_d2f_


def get_params():
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
        fipy_iter=5,
        C11=250.0,
        C12=150.0,
        C44=100.0,
        delta=0.0,
        misfit_strain=0.005,
        iterations=2,
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


def calc_elastic_d2f(params, total_strain, eta):
    """Calculate the second derivative of the elastic free energy
    """
    return pipe(
        eta,
        np.array,
        lambda x: calc_elastic_d2f_(
            params,
            valmap(to_face_value(eta.mesh), total_strain),
            calc_h(x),
            calc_dh(x),
            calc_d2h(x),
        ),
    )


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


@curry
def calc_d2f(params, total_strain, eta):
    """Calculate the second derivative of the total free energy
    """
    return calc_elastic_d2f(params, total_strain, eta) + calc_chem_d2f(eta)


def stiffness_matrix(params):
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
            [params["C11"], params["C12"], 0],
            [params["C12"], params["C11"], 0],
            [0, 0, params["C44"]],
        ]
    )


@curry
def calc_stiffness(params, calc_eta_func, coords):
    """Total stiffness tensor

    3 x 3 Stiffness matrix for Sfepy

    Args:
      calc_eta_func: function to calculate the phase field
      coords: the Sfepy coordinate array

    Returns:
      n x 3 x 3 stiffness tensor
    """
    return (
        stiffness_matrix(params)[None]
        * (1 + params["delta"] * calc_h(calc_eta_func(coords)))[:, None, None]
    )


@curry
def calc_prestress(params, calc_eta_func, coords):
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
        params["misfit_strain"],
        lambda x: np.dot(calc_stiffness(params, calc_eta_func, coords), [x, x, 0]),
        lambda x: -calc_eta_func(coords)[:, None] * x,
        lambda x: np.ascontiguousarray(x[:, :, None]),
    )


@curry
def sfepy_iter(params, eta):
    """One Sfepy iteration interpolating from FiPy

    Args:
      params: the parameter dictionary
      eta: the phase field

    Returns:
      dictionary with total strain fields and phase field
    """

    @curry
    def interpolate(var, coords):
        """Interpolate from a variable to given coords
        """
        return var(coords.swapaxes(0, 1), order=1)

    return pipe(
        eta,
        interpolate,
        lambda x: sfepy_solve(
            calc_stiffness(params, x),
            calc_prestress(params, x),
            (params["nx"], params["nx"]),
            params["lx"] / params["nx"],
        )[0],
        lambda x: x.swapaxes(0, 1),
        lambda x: x.reshape(x.shape[0] * x.shape[1], x.shape[2]),
        lambda x: dict(e11=x[:, 0], e12=x[:, 2], e22=x[:, 1]),
    )


@curry
def set_eta(eta_value, params, mesh, vars_):
    """Set the intial value of the phase field

    Args:
      params: the parameter dictioniary
      mesh: the FiPy mesh
      vars_: the solution variables

    Returns:
      nothing, this is a setter function
    """
    if eta_value is None:
        vars_["eta"].setValue(
            1., where=(mesh.x ** 2 + mesh.y ** 2) < params["radius"] ** 2
        )
    else:
        vars_["eta"].setValue(np.array(eta_value))


@curry
def fipy_iter(params, data):
    """One FiPy iteration

    Args:
      params: the parameter dictionary
      total_strain: dictionary of total strain fields

    Returns:
      the phase field variable
    """
    return pipe(
        dissoc(data, "eta"),
        calc_d2f(params),
        lambda x: fipy_solve(params, set_eta(data["eta"]), x)["eta"],
    )


@curry
def one_iter(params, data):
    """Do one iteration

    Args:
      params: the parameter dictionary
      data: dictionary of the phase field and strain fields

    Returns:
      dictionary of the phase field and strain fields
    """
    return pipe(
        data, fipy_iter(params), lambda x: assoc(sfepy_iter(params, x), "eta", x)
    )


def run_main(params):
    """Run the calculation

    Args:
      params: the list of parameters
      iterations: the number of iterations

    Returns:
      tuple of strain, displacement and stress
    """
    return pipe(
        dict(e11=0.0, e12=0.0, e22=0.0, eta=None),
        iterate_(one_iter(params), params["iterations"]),
    )


if __name__ == "__main__":  # pragma: no cover
    pipe(get_params(), run_main, get("eta"), view)
    input("stopped")
