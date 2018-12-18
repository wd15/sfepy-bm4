"""Calculate the elastic energy density

See index.ipynb for details of the derivation
"""
from toolz.curried import pipe


def term1(params, strain, d2h_value):
    """First term in expression
    """
    return (
        params["delta"]
        * d2h_value
        * (
            params["c11"] * (strain["e11"] ** 2 + strain["e22"] ** 2)
            + 2 * params["c12"] * strain["e11"] * strain["e22"]
            + 4 * params["c44"] * strain["e12"] ** 2
        )
    )


def term2(params, strain, dh_value):
    """Second term in expression
    """

    return params["delta"] * dh_value ** 2 * params["misfit_strain"] * (
        params["c11"] + params["c12"]
    ) * (strain["e11"] + strain["e22"])


def term3(params, h_value, dh_value):
    """Third term in expression
    """
    return (
        2 * (1 + params["delta"] * h_value)
        * dh_value ** 2
        * params["misfit_strain"] ** 2
        * (params["c11"] + params["c12"])
    )


def term4(params, strain, h_value, d2h):
    """Fourth term in expression
    """
    return (
        (1 + params["delta"] * h_value)
        * d2h
        * params["misfit_strain"]
        * (params["c11"] + params["c12"])
        * (strain["e11"] + strain["e22"])
    )


def calc_elastic_d2f_(params, total_strain, h_value, dh_value, d2h_value):
    """Calculate the second derivative of the elastic energy density
    """
    return pipe(
        dict(
            e11=total_strain["e11"] - h_value * params["misfit_strain"],
            e22=total_strain["e22"] - h_value * params["misfit_strain"],
            e12=total_strain["e12"],
        ),
        lambda x: (
            0.5 * term1(params, x, d2h_value)
            - 2 * term2(params, x, dh_value)
            + term3(params, h_value, dh_value)
            - term4(params, x, h_value, d2h_value)
        ),
    )
