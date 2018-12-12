"""All FiPy import go in this module
"""

import fipy as fp
import numpy as np
from toolz.curried import pipe, curry, do, iterate, assoc, dissoc


def get_mesh(params):
    """Create the mesh

    Args:
      params: the parameter dict

    Returns:
      the fipy mesh
    """
    return pipe(
        fp.Grid2D(Lx=params["lx"], Ly=params["lx"], nx=params["nx"], ny=params["nx"]),
        lambda x: x - np.array((params["lx"] / 2., params["lx"] / 2.))[:, None],
    )


@curry
def get_vars(params, mesh):
    """Get the variables

    Args:
      params: the parameter dict
      mesh: the FiPy mesh

    Returns:
      a dictionary of the variables (eta, d2f)
    """
    return pipe(
        dict(
            eta=fp.CellVariable(
                mesh=mesh, hasOld=True, value=params["eta0"], name="eta"
            ),
            d2f=fp.FaceVariable(mesh=mesh, name="d2f"),
        ),
        do(
            lambda x: x["eta"].setValue(
                1., where=(mesh.x ** 2 + mesh.y ** 2) < params["radius"] ** 2
            )
        ),
    )


@curry
def get_eq(params, eta, d2f):
    """Get the equation

    Args:
      params: the parameter dictionary
      eta: the phase field variable
      d2f: the free energy double derivative variable

    Returns:
      a dictionary of the equation and variables
    """
    return pipe(
        fp.CellVariable(mesh=eta.mesh, name="psi"),
        lambda x: (
            fp.TransientTerm(var=eta)
            == -fp.DiffusionTerm(coeff=params["mobility"], var=x)
            + fp.DiffusionTerm(coeff=params["mobility"] * d2f, var=eta),
            fp.ImplicitSourceTerm(coeff=1.0, var=x)
            == fp.DiffusionTerm(coeff=params["kappa"], var=eta),
        ),
        lambda x: x[0] & x[1],
    )


@curry
def sweep(calc_d2f, equation, eta, d2f):
    """Do one solve iteration

    Args:
      calc_d2f: function to calculate the free energy second
        derivative
      equation: the equation
      eta: the phase field variable
      d2f: the free energy double derivative variable

    Returns:
      dictionary including the equation, variables and previous
      residuals

    """
    eta.updateOld()
    d2f.setValue(calc_d2f(eta.faceValue))
    res = equation.sweep(dt=1e-1)
    print("fipy residual:", res)
    return res


@curry
def iterate_(func, times, value):
    """Use toolz iterate function to actually iterate

    Args:
      func: function to iteratep
      times: the number of iterations
      value: start value

    Returns:
      the updated value
    """
    iter_ = iterate(func, value)
    for _ in range(times):
        next(iter_)
    return next(iter_)


def solve(params, calc_d2f):
    """Solve the phase field problem with FiPy

    Args:
      params: dictionary of parameter values
      calc_df2: function to calculate the second derivative of the
        free energy function

    Returns:
      dictionary of the equation, variables and residuals
    """

    def sweep_wrapper(kwargs):
        """Wrapper for sweep function

        Ensures that residuals tuple has the residual appended to it.
        """
        return pipe(
            dissoc(kwargs, "residuals"),
            lambda x: sweep(calc_d2f, **x),
            lambda x: kwargs["residuals"] + (x,),
            assoc(kwargs, "residuals"),
        )

    return pipe(
        params,
        get_mesh,
        get_vars(params),
        lambda x: assoc(x, "equation", get_eq(params, **x)),
        lambda x: assoc(x, "residuals", ()),
        iterate_(sweep_wrapper, params["max_iter"]),
    )


def view(var):
    """View the phase field

    data:
      var: variable to view
    """
    fp.Viewer(var).plot()


@curry
def to_face_value(mesh, value):
    """Convert an array over cells to an array over faces
    """
    return pipe(
        value,
        lambda x: fp.CellVariable(mesh=mesh, value=value),
        lambda x: x.faceValue,
        np.array,
    )


def test():
    """Some tests
    """
    assert iterate_(lambda x: x * 2, 3, 1) == 8
