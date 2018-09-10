try:
    import sfepy
except ImportError:
    import pytest
    pytest.importorskip('sfepy')
    raise

import numpy as np
from sfepy.base.goptions import goptions
from sfepy.discrete.fem import Field
try:
    from sfepy.discrete.fem import FEDomain as Domain
except ImportError:
    from sfepy.discrete.fem import Domain
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.terms import Term, Terms
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete import Functions
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.base.base import output

from toolz.curried import pipe, curry


goptions['verbose'] = False
output.set_output(quiet=True)


@curry
def check(coords, ids):
    if len(ids) !=3 :
        raise RuntimeError("length of ids is incorrect")
    return ids


@curry
def subdomain(i_x, domain_, eps, coords, domain=None):
    def i_y():
        return (i_x + 1) % 2
    return pipe(
            (coords[:, i_x] > -eps) & (coords[:, i_x] < eps),
            lambda x: (coords[:, i_x] < domain_.get_mesh_bounding_box()[0][i_x] + eps) | x,
            lambda x: (coords[:, i_x] > domain_.get_mesh_bounding_box()[1][i_x] - eps) | x,
            lambda x: (coords[:, i_y()] < eps) & (coords[:, i_y()] > -eps) & x,
            lambda x: np.where(x)[0],
            check(coords)
        )


def get_bc(domain, dx, index, cond):
    return pipe(
        Function('fix_points', subdomain(index, domain, dx * 1e-3)),
        lambda x: domain.create_region(
            'region_fix_points',
            'vertices by fix_points',
            'vertex',
            functions=Functions([x])
        ),
        lambda x: EssentialBC('fix_points_BC', x, cond)
    )


def get_bcs(domain, dx):
    return Conditions([get_bc(domain, dx, 1, {'u.0': 0.0}),
                       get_bc(domain, dx, 0, {'u.1': 0.0})])


class ElasticFESimulation(object):
    def run(self, calc_stiffness, calc_prestress, shape, dx=1.0):
        domain = pipe(
            np.array(shape),
            lambda x: gen_block_mesh(x * dx, x + 1, np.zeros_like(shape), verbose=False),
            lambda x: Domain('domain', x)
        )

        region_all = domain.create_region('region_all', 'all')

        field = Field.from_args('fu', np.float64, 'vector', region_all, # pylint: disable=no-member
                                approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        def _material_func_(_, coors, mode=None, **kwargs):
            if mode == 'qp':
                return dict(
                    D=calc_stiffness(coors),
                    stress=calc_prestress(coors)
                )
            else:
                return

        m = Material(
            'm',
            function=Function('material_func', _material_func_)
        )

        integral = Integral('i', order=4)

        t1 = Term.new('dw_lin_elastic(m.D, v, u)',
                      integral, region_all, m=m, v=v, u=u)

        t2 = Term.new('dw_lin_prestress(m.stress, v)',
                      integral, region_all, m=m, v=v)

        eq = Equation('balance_of_forces', Terms([t1, t2]))
        eqs = Equations([eq])

        ebcs = get_bcs(domain, dx)

        ls = ScipyDirect({})

        pb = Problem('elasticity', equations=eqs)

        pb.time_update(ebcs=ebcs)

        ev = pb.get_evaluator()
        nls = Newton({}, lin_solver=ls,
                     fun=ev.eval_residual, fun_grad=ev.eval_tangent_matrix)

        pb.set_solver(nls)

        vec = pb.solve()

        u = vec.create_output_dict()['u'].data

        u_reshape = np.reshape(u, (tuple(x + 1 for x in shape) + u.shape[-1:]))

        dims = domain.get_mesh_bounding_box().shape[1]
        strain = np.squeeze(
            pb.evaluate(
                'ev_cauchy_strain.{dim}.region_all(u)'.format(
                    dim=dims),
                mode='el_avg',
                copy_materials=False))
        strain_reshape = np.reshape(strain, (shape + strain.shape[-1:]))

        stress = np.squeeze(
            pb.evaluate(
                'ev_cauchy_stress.{dim}.region_all(m.D, u)'.format(
                    dim=dims),
                mode='el_avg',
                copy_materials=False))
        stress_reshape = np.reshape(stress, (shape + stress.shape[-1:]))

        return strain_reshape, u_reshape, stress_reshape
