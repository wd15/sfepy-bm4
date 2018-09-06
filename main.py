import numpy as np
from module import ElasticFESimulation
from toolz.curried import pipe


def calc_eta(coords, delta=1.0, radius=2.5):
    return pipe(
        coords,
        lambda x: np.sqrt(x[:, 0]**2 + x[:, 1]**2),
        lambda x: 0.5 * (1 + np.tanh((-x + radius) * 2 / delta))
    )


def calc_h(eta):
    return eta**3 * (6 * eta**2 - 15 * eta + 10)


def stiffness_matrix(c11=250, c12=150, c44=100):
    return np.array(
        [[c11, c12, 0  ],
         [c12, c11, 0  ],
         [  0,   0, c44]]
    )


def calc_stiffness(coords):
    return stiffness_matrix()[None] * (1 + 0.1 * calc_h(calc_eta(coords)))[:, None, None]


def main(shape):
    return ElasticFESimulation(macro_strain=0.1).run(calc_stiffness, shape)


strain, displacement, stress = main((10, 10))

u = displacement


print(u[-1,:,0] - u[0,:,0])
macro_strain = 0.1
shape = 10, 10
assert np.allclose(u[-1,:,0] - u[0,:,0], 10 * macro_strain)



assert np.allclose(u[0,:,1], u[-1,:,1])


assert np.allclose(u[:,0], u[:,-1])
