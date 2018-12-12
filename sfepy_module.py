"""All Sfepy imports go in this module
"""
import numpy as np
from sfepy.base.goptions import goptions
from sfepy.discrete.fem import Field
from sfepy.discrete.fem import FEDomain as Domain
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


def check(ids):
    """Check that the fixed displacement nodes have been isolated

    Args:
      ids: the isolated IDs

    Returns the unchanged IDs

    >>> check([1, 2, 3, 4])
    Traceback (most recent call last):
    ...
    RuntimeError: length of ids is incorrect
    """
    if len(ids) != 3:
        raise RuntimeError("length of ids is incorrect")
    return ids


@curry
def subdomain(i_x, domain_, eps, coords, **_):
    """Find the node IDs that will be fixed

    Args:
      i_x: the index (either 0 or 1) depends on direction of axes
      domain_: the Sfepy domain
      eps: a small value
      coords: the coordinates of the nodes

    Returns:
      the isolated node IDs
    """

    def i_y():
        """Switch the index from 0 -> 1 or from 1 -> 0
        """
        return (i_x + 1) % 2

    return pipe(
        (coords[:, i_x] > -eps) & (coords[:, i_x] < eps),
        lambda x: (coords[:, i_x] < domain_.get_mesh_bounding_box()[0][i_x] + eps) | x,
        lambda x: (coords[:, i_x] > domain_.get_mesh_bounding_box()[1][i_x] - eps) | x,
        lambda x: (coords[:, i_y()] < eps) & (coords[:, i_y()] > -eps) & x,
        lambda x: np.where(x)[0],
        check,
    )


def get_bc(domain, delta_x, index, cond):
    """Make a displacement boundary condition

    Args:
      domain: the Sfepy domain
      delta_x: the mesh spacing
      index: the index (either 0 or 1) depends on direction of axes
      cond: the BC dictionary

    Returns:
      the Sfepy boundary condition
    """
    return pipe(
        Function("fix_points", subdomain(index, domain, delta_x * 1e-3)),
        lambda x: domain.create_region(
            "region_fix_points",
            "vertices by fix_points",
            "vertex",
            functions=Functions([x]),
        ),
        lambda x: EssentialBC("fix_points_BC", x, cond),
    )


def get_bcs(domain, delta_x):
    """Get the boundary conditions

    Args:
      domain: the Sfepy domain
      delta_x: the mesh spacing

    Returns:
      the boundary conditions
    """
    return Conditions(
        [
            get_bc(domain, delta_x, 1, {"u.0": 0.0}),
            get_bc(domain, delta_x, 0, {"u.1": 0.0}),
        ]
    )


def get_material(calc_stiffness, calc_prestress):
    """Get the material

    Args:
      calc_stiffness: the function for calculating the stiffness tensor
      calc_prestress: the function for calculating the prestress

    Returns:
      the material
    """

    def _material_func_(_, coors, mode=None, **__):
        if mode == "qp":
            return dict(D=calc_stiffness(coors), stress=calc_prestress(coors))
        return None

    return Material("m", function=Function("material_func", _material_func_))


def get_uv(shape, delta_x):
    """Get the fields for the displacement and test function

    Args:
      shape: the shape of the domain
      delta_x: the mesh spacing

    Returns:
      tuple of field variables
    """
    return pipe(
        np.array(shape),
        lambda x: gen_block_mesh(
            x * delta_x, x + 1, np.zeros_like(shape), verbose=False
        ),
        lambda x: Domain("domain", x),
        lambda x: x.create_region("region_all", "all"),
        lambda x: Field.from_args("fu", np.float64, "vector", x, approx_order=2),
        lambda x: (
            FieldVariable("u", "unknown", x),
            FieldVariable("v", "test", x, primary_var_name="u"),
        ),
    )
    # field = Field.from_args('fu', np.float64, 'vector', region_all,
    # pylint: disable=no-member
    #                         approx_order=2)


def get_terms(u_field, v_field, calc_stiffness, calc_prestress):
    """Get the terms for the equation

    Args:
      u_field: the displacement field
      v_field: the test function field
      calc_stiffness: a function to calculate the stiffness tensor
      calc_prestress: a function to calculate the prestress tensor

    Returns:
      a tuple of terms for the equation
    """
    return (
        Term.new(
            "dw_lin_elastic(m.D, v, u)",
            Integral("i", order=4),
            v_field.field.region,
            m=get_material(calc_stiffness, calc_prestress),
            v=v_field,
            u=u_field,
        ),
        Term.new(
            "dw_lin_prestress(m.stress, v)",
            Integral("i", order=4),
            v_field.field.region,
            m=get_material(calc_stiffness, calc_prestress),
            v=v_field,
        ),
    )


def get_nls(evaluator):
    """Get the non-linear solver

    Args:
      evaluator: the problem evaluator

    Returns:
      the non-linear solver
    """
    return Newton(
        {},
        lin_solver=ScipyDirect({}),
        fun=evaluator.eval_residual,
        fun_grad=evaluator.eval_tangent_matrix,
    )


def get_problem(u_field, v_field, calc_stiffness, calc_prestress, delta_x):
    """Get the problem

    Args:
      u_field: the displacement field
      v_field: the test function field
      calc_stiffness: a functioin to calcuate the stiffness tensor
      calc_prestress: a function to calculate the prestress tensor
      delta_x: the mesh spacing

    Returns:
      the Sfepy problem
    """
    return pipe(
        get_terms(u_field, v_field, calc_stiffness, calc_prestress),
        lambda x: Equation("balance_of_forces", Terms([x[0], x[1]])),
        lambda x: Problem("elasticity", equations=Equations([x])),
        do(lambda x: x.time_update(ebcs=get_bcs(v_field.field.region.domain, delta_x))),
        do(lambda x: x.set_solver(get_nls(x.get_evaluator()))),
    )


def get_displacement(vec, shape):
    """Extract reshaped displacement from output vector

    Args:
      vec: the output vector obtained from problem.solve()
      shape: the shape of the mesh

    Returns:
      reshaped displacement field
    """
    return pipe(
        vec.create_output_dict()["u"].data,
        lambda x: np.reshape(x, (tuple(y + 1 for y in shape) + x.shape[-1:])),
    )


def get_strain(problem, shape):
    """Calculate the strain field

    Args:
      problem: the Sfepy problem
      hape: the shape of the mesh

    Returns:
      the reshaped strain field
    """
    return get_stress_strain(problem, shape, "ev_cauchy_strain.{dim}.region_all(u)")


def get_stress(problem, shape):
    """Calculate the strain field

    Args:
      problem: the Sfepy problem
      shape: the shape of the mesh

    Returns:
      the reshaped stress field
    """
    return get_stress_strain(
        problem, shape, "ev_cauchy_stress.{dim}.region_all(m.D, u)"
    )


def get_stress_strain(problem, shape, str_):
    """Get the stress or strain field depending on the str_ argument

    Args:
      problem: the Sfepy problem
      shape: the shape of the domain
      str_: string passed to problem.evaluate to extract the stress or
        strain

    Returns
      the reshaped stress or strain field
    """
    return pipe(
        np.squeeze(
            problem.evaluate(
                str_.format(dim=len(shape)), mode="el_avg", copy_materials=False
            )
        ),
        lambda x: np.reshape(x, (shape + x.shape[-1:])),
    )


def get_data(shape, problem, vec):
    """Extract the displacement, strain and stress fields

    Args:
      shape: the shape of the mesh
      problem: the Sfepy problem
      vec: the output vector from problem.solve

    Returns:
      a tuple of arrays for the strain, displacement and stress fields

    """
    return (
        get_strain(problem, shape),
        get_displacement(vec, shape),
        get_stress(problem, shape),
    )


def solve(calc_stiffness, calc_prestress, shape, delta_x):
    """Solve the linear elasticity problem

    Args:
      calc_stiffness: the function to calculate the stiffness tensor
      calc_prestress: the funtion to calculate the prestress tensor
      shape: the shape of the mesh
      delta_x: the mesh spacing

    Returns:
      a tuple of arrays for the displacement, strain and stress fields
    """
    return pipe(
        get_uv(shape, delta_x),
        lambda x: get_problem(x[0], x[1], calc_stiffness, calc_prestress, delta_x),
        lambda x: get_data(shape, x, x.solve()),
    )
