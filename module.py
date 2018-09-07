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
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
import sfepy.discrete.fem.periodic as per
from sfepy.discrete import Functions
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.base.base import output
from sfepy.discrete.conditions import LinearCombinationBC

goptions['verbose'] = False
output.set_output(quiet=True)


class ElasticFESimulation(object):
    def __init__(self, macro_strain=1.,):
        self.macro_strain = macro_strain
        self.dx = 1.0


    def _subdomain_func(self, x=(), y=(), z=(), max_x=None):
        """
        Creates a function to mask subdomains in Sfepy.

        Args:
          x: tuple of lines or points to be masked in the x-plane
          y: tuple of lines or points to be masked in the y-plane
          z: tuple of lines or points to be masked in the z-plane

        Returns:
          array of masked location indices

        """
        eps = 1e-3 * self.dx

        def _func(coords, domain=None):
            flag_x = len(x) == 0
            flag_y = len(y) == 0
            flag_z = len(z) == 0
            for x_ in x:
                flag = (coords[:, 0] < (x_ + eps)) & \
                       (coords[:, 0] > (x_ - eps))
                flag_x = flag_x | flag
            for y_ in y:
                flag = (coords[:, 1] < (y_ + eps)) & \
                       (coords[:, 1] > (y_ - eps))
                flag_y = flag_y | flag
            for z_ in z:
                flag = (coords[:, 2] < (z_ + eps)) & \
                       (coords[:, 2] > (z_ - eps))
                flag_z = flag_z | flag
            flag = flag_x & flag_y & flag_z
            if max_x is not None:
                flag = flag & (coords[:, 0] < (max_x - eps))
            return np.where(flag)[0]

        return _func

    def _get_mesh(self, shape):
        """
        Generate an Sfepy rectangular mesh

        Args:
          shape: proposed shape of domain (vertex shape) (n_x, n_y)

        Returns:
          Sfepy mesh

        """
        center = np.zeros_like(shape)
        return gen_block_mesh(shape, np.array(shape) + 1, center,
                              verbose=False)

    def _get_fixed_displacementsBCs(self, domain):
        """
        Fix the left top and bottom points in x, y and z

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]

        kwargs = {}
        fix_points_dict = {'u.0': 0.0, 'u.1': 0.0}
        if len(min_xyz) == 3:
            kwargs = {'z': (max_xyz[2], min_xyz[2])}
            fix_points_dict['u.2'] = 0.0
        fix_x_points_ = self._subdomain_func(x=(min_xyz[0],),
                                             y=(max_xyz[1], min_xyz[1]),
                                             **kwargs)

        fix_x_points = Function('fix_x_points', fix_x_points_)
        region_fix_points = domain.create_region(
            'region_fix_points',
            'vertices by fix_x_points',
            'vertex',
            functions=Functions([fix_x_points]))
        return EssentialBC('fix_points_BC', region_fix_points, fix_points_dict)

    def _get_shift_displacementsBCs(self, domain):
        """
        Fix the right top and bottom points in x, y and z

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]
        kwargs = {}
        if len(min_xyz) == 3:
            kwargs = {'z': (max_xyz[2], min_xyz[2])}

        displacement = self.macro_strain * (max_xyz[0] - min_xyz[0])
        shift_points_dict = {'u.0': displacement}

        shift_x_points_ = self._subdomain_func(x=(max_xyz[0],),
                                               y=(max_xyz[1], min_xyz[1]),
                                               **kwargs)

        shift_x_points = Function('shift_x_points', shift_x_points_)
        region_shift_points = domain.create_region(
            'region_shift_points',
            'vertices by shift_x_points',
            'vertex',
            functions=Functions([shift_x_points]))
        return EssentialBC('shift_points_BC',
                           region_shift_points, shift_points_dict)

    def _get_displacementBCs(self, domain):
        shift_points_BC = self._get_shift_displacementsBCs(domain)
        fix_points_BC = self._get_fixed_displacementsBCs(domain)
        return Conditions([fix_points_BC, shift_points_BC])

    def _get_linear_combinationBCs(self, domain):
        """
        The right nodes are periodic with the left nodes but also displaced.

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]
        xplus_ = self._subdomain_func(x=(max_xyz[0],))
        xminus_ = self._subdomain_func(x=(min_xyz[0],))

        xplus = Function('xplus', xplus_)
        xminus = Function('xminus', xminus_)
        region_x_plus = domain.create_region('region_x_plus',
                                             'vertices by xplus',
                                             'facet',
                                             functions=Functions([xplus]))
        region_x_minus = domain.create_region('region_x_minus',
                                              'vertices by xminus',
                                              'facet',
                                              functions=Functions([xminus]))
        match_x_plane = Function('match_x_plane', per.match_x_plane)

        def shift_(ts, coors, region):
            return np.ones_like(coors[:, 0]) * \
                self.macro_strain * (max_xyz[0] - min_xyz[0])
        shift = Function('shift', shift_)
        lcbc = LinearCombinationBC(
            'lcbc', [region_x_plus, region_x_minus], {
                'u.0': 'u.0'}, match_x_plane, 'shifted_periodic',
            arguments=(shift,))

        return Conditions([lcbc])

    def _get_periodicBC_X(self, domain, dim):
        dim_dict = {1: ('y', per.match_y_plane),
                    2: ('z', per.match_z_plane)}
        dim_string = dim_dict[dim][0]
        match_plane = dim_dict[dim][1]
        min_, max_ = domain.get_mesh_bounding_box()[:, dim]
        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        plus_ = self._subdomain_func(max_x=max_x, **{dim_string: (max_,)})
        minus_ = self._subdomain_func(max_x=max_x, **{dim_string: (min_,)})
        plus_string = dim_string + 'plus'
        minus_string = dim_string + 'minus'
        plus = Function(plus_string, plus_)
        minus = Function(minus_string, minus_)
        region_plus = domain.create_region(
            'region_{0}_plus'.format(dim_string),
            'vertices by {0}'.format(
                plus_string),
            'facet',
            functions=Functions([plus]))
        region_minus = domain.create_region(
            'region_{0}_minus'.format(dim_string),
            'vertices by {0}'.format(
                minus_string),
            'facet',
            functions=Functions([minus]))
        match_plane = Function(
            'match_{0}_plane'.format(dim_string), match_plane)

        bc_dict = {'u.0': 'u.0'}

        bc = PeriodicBC('periodic_{0}'.format(dim_string),
                        [region_plus, region_minus],
                        bc_dict,
                        match='match_{0}_plane'.format(dim_string))
        return bc, match_plane

    def _get_periodicBC_YZ(self, domain, dim):
        dims = domain.get_mesh_bounding_box().shape[1]
        dim_dict = {0: ('x', per.match_x_plane),
                    1: ('y', per.match_y_plane),
                    2: ('z', per.match_z_plane)}
        dim_string = dim_dict[dim][0]
        match_plane = dim_dict[dim][1]
        min_, max_ = domain.get_mesh_bounding_box()[:, dim]
        plus_ = self._subdomain_func(**{dim_string: (max_,)})
        minus_ = self._subdomain_func(**{dim_string: (min_,)})
        plus_string = dim_string + 'plus'
        minus_string = dim_string + 'minus'
        plus = Function(plus_string, plus_)
        minus = Function(minus_string, minus_)
        region_plus = domain.create_region(
            'region_{0}_plus'.format(dim_string),
            'vertices by {0}'.format(
                plus_string),
            'facet',
            functions=Functions([plus]))
        region_minus = domain.create_region(
            'region_{0}_minus'.format(dim_string),
            'vertices by {0}'.format(
                minus_string),
            'facet',
            functions=Functions([minus]))
        match_plane = Function(
            'match_{0}_plane'.format(dim_string), match_plane)

        bc_dict = {'u.1': 'u.1'}
        if dims == 3:
            bc_dict['u.2'] = 'u.2'

        bc = PeriodicBC('periodic_{0}'.format(dim_string),
                        [region_plus, region_minus],
                        bc_dict,
                        match='match_{0}_plane'.format(dim_string))
        return bc, match_plane

    def run(self, calc_stiffness, calc_prestress, shape):
        mesh = self._get_mesh(shape)
        domain = Domain('domain', mesh)

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

        epbcs, functions = self._get_periodicBCs(domain)
        ebcs = self._get_displacementBCs(domain)
        lcbcs = self._get_linear_combinationBCs(domain)

        ls = ScipyDirect({})

        pb = Problem('elasticity', equations=eqs, functions=functions)

        pb.time_update(ebcs=ebcs, epbcs=epbcs, lcbcs=lcbcs)

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

    def _get_periodicBCs(self, domain):
        dims = domain.get_mesh_bounding_box().shape[1]

        bc_list_YZ, func_list_YZ = list(
            zip(*[self._get_periodicBC_YZ(domain, i) for i in range(0, dims)]))
        bc_list_X, func_list_X = list(
            zip(*[self._get_periodicBC_X(domain, i) for i in range(1, dims)]))
        return Conditions(
            bc_list_YZ + bc_list_X), Functions(func_list_YZ + func_list_X)
