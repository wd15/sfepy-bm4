import fipy as fp
import numpy as np
from toolz.curried import pipe, curry, do, iterate, get


def h(eta):
    return eta**3 * (6 * eta**2 - 15 * eta + 10)


def dh(eta):
    return 30 * eta**2 * (eta - 1)**2


def d2h(eta):
    return 60 * eta  * (2 * eta - 1) * (eta - 1)


def elastic():
    C_11 = 250
    C_12 = 150
    epsilon = 0.005
    return 2 * epsilon**2 * (C_11 + C_12)


def df2_elastic(eta):
    return (dh(eta)**2 + h(eta) * d2h(eta)) * elastic()


def df2_chem(eta):
    a_j = [0, 0, 8.072789087, -81.24549382, 408.0297321, -1244.129167,
           2444.046270, -3120.635139, 2506.663551, -1151.003178,
           230.2006355]
    eta_array = np.array(eta)

    def calc(j):
        return a_j[j] * j * (j - 1) * eta_array**(j - 2)

    out = 0.0
    for i in np.arange(2, 11):
        out += calc(i)

    out = 0.1 * out #np.sum(list(map(calc, np.arange(2, 11))))

    return out


def df2_(eta):
    return df2_elastic(eta) + df2_chem(eta)


def get_mesh(params):
    return pipe(
        fp.Grid2D(Lx=params['lx'], Ly=params['lx'], nx=params['nx'], ny=params['nx']),
        lambda x: x - np.array((params['lx'] / 2., params['lx'] / 2.))[:, None]
    )


@curry
def get_vars(params, mesh):
    return pipe(
        dict(
            eta=fp.CellVariable(mesh=mesh, hasOld=True, value=params['eta0'], name='eta'),
            psi=fp.CellVariable(mesh=mesh, hasOld=True, name='psi'),
            df2=fp.FaceVariable(mesh=mesh, name='df2')
        ),
        do(lambda x: x['eta'].setValue(1., where=(mesh.x**2 + mesh.y**2) < params['radius']**2))
    )


@curry
def get_eq(params, eta, psi, df2):
    return pipe(
        (
            fp.TransientTerm(var=eta) == \
            -fp.DiffusionTerm(coeff=params['mobility'], var=psi) + \
            fp.DiffusionTerm(coeff=params['mobility'] * df2, var=eta),
            fp.ImplicitSourceTerm(coeff=1.0, var=psi) == \
            fp.DiffusionTerm(coeff=params['kappa'], var=eta)
        ),
        lambda x: dict(eq=x[0] & x[1], eta=eta, psi=psi, df2=df2)
    )


def sweep(eq, eta, psi, df2, res=()):
    return pipe(
        (eta, psi, df2),
        do(lambda x: (x[0].updateOld(), x[1].updateOld(), x[2].setValue(df2_(x[0].faceValue)))),
        lambda _: eq.sweep(dt=1e-1),
        do(lambda x: print(x)),
        lambda x: dict(eq=eq, eta=eta, psi=psi, df2=df2, res=res + (x,))
    )


@curry
def iterate_(func, times, value):
    """Use toolz iterate function to actually iterate
    """
    iter_ = iterate(func, value)
    for _ in range(times):
        next(iter_)
    return next(iter_)


def solve_(params):
    return pipe(
        params,
        get_mesh,
        get_vars(params),
        lambda x: get_eq(params, **x),
        iterate_(lambda x: sweep(**x), params['max_iter'])
    )


def solve(params):
    return solve_(params)['eta']


def test():
    assert iterate_(lambda x: x * 2, 3, 1) == 8
    assert np.allclose(solve_(
        dict(
            lx=200.0,
            nx=200,
            radius=20.0,
            kappa=0.29,
            mobility=5.0,
            eta0=0.0065,
            max_iter=2,
        ))['res'][-1], 60.73614562846711)


if __name__ == '__main__':
    params = dict(
        lx=200.0,
        nx=200,
        radius=20.0,
        kappa=0.29,
        mobility=5.0,
        eta0=0.0065,
        max_iter=5
    )
    fp.Viewer(solve(params)).plot()
    input('stopped')
