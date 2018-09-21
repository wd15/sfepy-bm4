"""Calculate the elastic energy density

See index.ipynb for details of the derivation
"""


def term1(params, strain, d2h_value):
    """First term in expression
    """
    return (
        params["delta"]
        * d2h_value
        * (
            params["C11"] * (strain["11"] ** 2 + strain["22"] ** 2)
            + 2 * params["C12"] * strain["11"] * strain["22"]
            + 4 * params["C44"] * strain["12"] ** 2
        )
    )


def term2(params, strain, dh_value):
    """Second term in expression
    """
    return params["delta"] * dh_value ** 2 * params["strain_misfit"] * (
        params["C11"] + params["C22"]
    ) + (strain["11"] + strain["22"])


def term3(params, h_value, dh_value):
    """Third term in expression
    """
    return (
        (1 + params["delta"] * h_value)
        * dh_value ** 2
        * params["strain_misfit"] ** 2
        * (params["C11"] + params["C22"])
    )


def term4(params, strain, h_value, d2h):
    """Fourth term in expression
    """
    return (1 + params["delta"] * h_value) * d2h * params["strain_misfit"] * (
        params["C11"] + params["C22"]
    ) + (strain["11"] + strain["22"])


def calc_elastic_d2f(params, strain, h_value, dh_value, d2h_value):
    """Calculate the second derivative of the elastic energy density
    """
    return (
        0.5 * term1(params, strain, d2h_value)
        - 2 * term2(params, strain, dh_value)
        + term3(params, h_value, dh_value)
        - term4(params, strain, h_value, d2h_value)
    )
