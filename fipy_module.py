import fipy as fp
import numpy as np
from toolz.curried import map


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

def run():
    l = 200.0
    radius = 20.0
    n = 200
    kappa = 0.29

    m = fp.Grid2D(Lx=l, Ly=l, nx=n, ny=n) - np.array((l / 2., l / 2.))[:, None]

    eta = fp.CellVariable(mesh=m, hasOld=True, value=0.0065)
    psi = fp.CellVariable(mesh=m, hasOld=True)
    df2 = fp.FaceVariable(mesh=m)

    eta.setValue(1., where=(m.x**2 + m.y**2) < radius**2)

    mobility = 5.0

    #eq = fp.TransientTerm() == -fp.DiffusionTerm((mobility, kappa)) + fp.DiffusionTerm(mobility * df2)
    eq1 = fp.TransientTerm(var=eta) == -fp.DiffusionTerm(coeff=mobility, var=psi) + fp.DiffusionTerm(coeff=mobility * df2, var=eta)
    eq2 = fp.ImplicitSourceTerm(coeff=1.0, var=psi) == fp.DiffusionTerm(coeff=kappa, var=eta)

    eq = eq1 & eq2

    for i in range(100):
        eta.updateOld()
        psi.updateOld()
        df2[:] = df2_(eta.faceValue)
        # res = eq.sweep(eta, dt=1.0e-1)
        res = eq.sweep(dt=1e-1)
        print(res)

    return eta

if __name__ == '__main__':
    fp.Viewer(run()).plot()
    input('stopped')
