import numpy as np
from module import ElasticFESimulation

# X = np.zeros((1, 3, 3), dtype=int)
# X[0, :, 1] = 1

# sim = ElasticFESimulation(elastic_modulus=(1.0, 10.0),
#                           poissons_ratio=(0., 0.))
# sim.run(X)
# y = sim.strain

# exx = y[..., 0]
# eyy = y[..., 1]
# exy = y[..., 2]

# assert np.allclose(exx, 1)
# assert np.allclose(eyy, 0)
# assert np.allclose(exy, 0)

X = np.array([[[1, 0, 0, 1],
               [0, 1, 1, 1],
               [0, 0, 1, 1],
               [1, 0, 0, 1]]])
n_samples, N, N = X.shape
macro_strain = 0.1
sim = ElasticFESimulation((10.0,1.0), (0.3,0.3), macro_strain=0.1)
sim.run(X)
u = sim.displacement[0]


print(u[-1,:,0] - u[0,:,0])
assert np.allclose(u[-1,:,0] - u[0,:,0], N * macro_strain)


assert np.allclose(u[0,:,1], u[-1,:,1])


assert np.allclose(u[:,0], u[:,-1])
