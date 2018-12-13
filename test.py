"""Run test cases
"""
from toolz.curried import pipe, curry, assoc, get
import numpy as np

from main import (
    calc_stiffness,
    calc_prestress,
    sfepy_solve,
    fipy_solve,
    get_params,
    calc_d2f,
    run_main,
    set_eta,
)


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


def run_sfepy_fake(params, shape, delta_x):
    """Run a Sfepy calculation
    """
    calc_eta_func = calc_eta(delta=delta_x, radius=shape[0] * delta_x / 4.)
    return sfepy_solve(
        calc_stiffness(params, calc_eta_func),
        calc_prestress(params, calc_eta_func),
        shape,
        delta_x,
    )


def test_sfepy():
    """Run some tests
    """
    assert np.allclose(
        run_sfepy_fake(assoc(get_params(), "delta", 0.1), (10, 10), 1.0)[1][0, 0],
        [-0.00515589, -0.00515589],
    )
    assert np.allclose(
        run_sfepy_fake(assoc(get_params(), "delta", 0.1), (10, 10), 0.1)[1][0, 0],
        [-0.00051559, -0.00051559],
    )


def test_fipy():
    """Run the FiPy tests
    """
    assert pipe(
        dict(e11=0.0, e12=0.0, e22=0.0),
        lambda x: np.allclose(
            fipy_solve(
                assoc(get_params(), "max_iter", 2), set_eta, calc_d2f(get_params(), x)
            )["residuals"][-1],
            60.309247734253795,
        ),
    )


def test_combined():
    """Run a combined test
    """
    assert pipe(
        get_params(),
        assoc(key="max_iter", value=2),
        run_main,
        get("eta"),
        np.array,
        np.sum,
        lambda x: np.allclose(x, 1515.784),
    )
