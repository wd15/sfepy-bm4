import fipy as fp
import numpy as np


def run():
    l = 400.0
    n = 100

    m = fp.Grid2D(Lx=l, Ly=l, nx=n, ny=n) - np.array((l / 2., l / 2.))[:, None]

    v = fp.CellVariable(mesh=m)
    v.setValue(1., where=(m.x**2 + m.y**2) < 20**2)
    return v


if __name__ == '__main__':
    fp.Viewer(run()).plot()
    input('stopped')
