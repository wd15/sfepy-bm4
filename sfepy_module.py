try:
    import sfepy
except ImportError:
    import pytest

    pytest.importorskip("sfepy")
    raise

import numpy as np
from sfepy.base.goptions import goptions
from sfepy.discrete.fem import Field

try:
    from sfepy.discrete.fem import FEDomain as Domain
except ImportError:
    from sfepy.discrete.fem import Domain
from sfepy.discrete import (
    FieldVariable,
    Material,
    Integral,
    Function,
    Equation,
    Equations,
    Problem,
)
from sfepy.terms import Term, Terms
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete import Functions
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.base.base import output

from toolz.curried import pipe, curry, do


goptions["verbose"] = False
output.set_output(quiet=True)


@curry
def check(coords, ids):
    if len(ids) != 3:
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
        check(coords),
    )


def get_bc(domain, dx, index, cond):
    return pipe(
        Function("fix_points", subdomain(index, domain, dx * 1e-3)),
        lambda x: domain.create_region(
            "region_fix_points",
            "vertices by fix_points",
            "vertex",
            functions=Functions([x]),
        ),
        lambda x: EssentialBC("fix_points_BC", x, cond),
    )


def get_bcs(domain, dx):
    return Conditions(
        [get_bc(domain, dx, 1, {"u.0": 0.0}), get_bc(domain, dx, 0, {"u.1": 0.0})]
    )


def get_material(calc_stiffness, calc_prestress):
    def _material_func_(_, coors, mode=None, **kwargs):
        if mode == "qp":
            return dict(D=calc_stiffness(coors), stress=calc_prestress(coors))
        else:
            return

    return Material("m", function=Function("material_func", _material_func_))


def get_uv(shape, dx):
    return pipe(
        np.array(shape),
        lambda x: gen_block_mesh(x * dx, x + 1, np.zeros_like(shape), verbose=False),
        lambda x: Domain("domain", x),
        lambda x: x.create_region("region_all", "all"),
        lambda x: Field.from_args("fu", np.float64, "vector", x, approx_order=2),
        lambda x: (
            FieldVariable("u", "unknown", x),
            FieldVariable("v", "test", x, primary_var_name="u"),
        ),
    )
    # field = Field.from_args('fu', np.float64, 'vector', region_all, # pylint: disable=no-member
    #                         approx_order=2)


def get_terms(u, v, calc_stiffness, calc_prestress):
    return (
        Term.new(
            "dw_lin_elastic(m.D, v, u)",
            Integral("i", order=4),
            v.field.region,
            m=get_material(calc_stiffness, calc_prestress),
            v=v,
            u=u,
        ),
        Term.new(
            "dw_lin_prestress(m.stress, v)",
            Integral("i", order=4),
            v.field.region,
            m=get_material(calc_stiffness, calc_prestress),
            v=v
        )
    )


def get_nls(ev):
    return Newton(
        {}, lin_solver=ScipyDirect({}), fun=ev.eval_residual, fun_grad=ev.eval_tangent_matrix
    )


def get_problem(u, v, calc_stiffness, calc_prestress, dx):
    return pipe(
        get_terms(u, v, calc_stiffness, calc_prestress),
        lambda x: Equation("balance_of_forces", Terms([x[0], x[1]])),
        lambda x: Problem("elasticity", equations=Equations([x])),
        do(lambda x: x.time_update(ebcs=get_bcs(v.field.region.domain, dx))),
        do(lambda x: x.set_solver(get_nls(x.get_evaluator())))
    )


def solve(calc_stiffness, calc_prestress, shape, dx=1.0):
    pb = pipe(
        get_uv(shape, dx),
        lambda x: get_problem(x[0], x[1], calc_stiffness, calc_prestress, dx)
    )

    vec = pb.solve()

    u = vec.create_output_dict()["u"].data

    u_reshape = np.reshape(u, (tuple(x + 1 for x in shape) + u.shape[-1:]))

    dims = len(shape) #v.field.domain.get_mesh_bounding_box().shape[1]

    strain = np.squeeze(
        pb.evaluate(
            "ev_cauchy_strain.{dim}.region_all(u)".format(dim=dims),
            mode="el_avg",
            copy_materials=False,
        )
    )
    strain_reshape = np.reshape(strain, (shape + strain.shape[-1:]))

    stress = np.squeeze(
        pb.evaluate(
            "ev_cauchy_stress.{dim}.region_all(m.D, u)".format(dim=dims),
            mode="el_avg",
            copy_materials=False,
        )
    )
    stress_reshape = np.reshape(stress, (shape + stress.shape[-1:]))

    return strain_reshape, u_reshape, stress_reshape
