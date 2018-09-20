"""Run test cases
"""
from toolz.curried import pipe, curry, assoc, do
import numpy as np

from fipy_module import view
from main import (
    calc_stiffness,
    calc_prestress,
    sfepy_solve,
    fipy_solve,
    params,
    calc_d2f,
    run,
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


def run_sfepy_fake(shape, delta_x):
    """Run a Sfepy calculation
    """
    calc_eta_func = calc_eta(delta=delta_x, radius=shape[0] * delta_x / 4.)
    return sfepy_solve(
        calc_stiffness(calc_eta_func), calc_prestress(calc_eta_func), shape, delta_x
    )


def test_sfepy():
    """Run some tests
    """
    assert np.allclose(
        run_sfepy_fake((10, 10), 1.0)[1][0, 0], [-0.00515589, -0.00515589]
    )
    assert np.allclose(
        run_sfepy_fake((10, 10), 0.1)[1][0, 0], [-0.00051559, -0.00051559]
    )


def test_fipy():
    """Run the FiPy tests
    """
    assert np.allclose(
        fipy_solve(assoc(params(), "max_iter", 2), calc_d2f)["residuals"][-1],
        60.73614562846711,
    )


def run_sfepy():
    """Run the Sfepy example
    """
    import matplotlib.pyplot as plt

    pipe(
        run()[1],
        lambda x: np.sqrt(np.sum(x ** 2, axis=-1)).swapaxes(0, 1),
        do(plt.imshow),
    )
    plt.colorbar()
    plt.show()
    input("stopped")


def run_fipy():
    """Run the fipy example
    """
    view(fipy_solve(params(), calc_d2f)["eta"])
    input("stopped")


if __name__ == "__main__":
    run_sfepy()
