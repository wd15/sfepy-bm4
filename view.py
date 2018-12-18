"""View results for benchmark problem 4

To plot the radius of the precipitate in the x direction
    $ python view.py --folder=data_xxx a_10

To plot the radius of the precipitate in the y direction
    $ python view.py --folder=data_xxx a_10

To plot the radius of the precipitate in the diagonal direction
    $ python view.py --folder=data_xxx a_d

# To plot the solid fraction from directory "data_xxx"
#     $ python view.py --folder data_xxx solid_fraction

# To plot the free energy
#     $ python view.py --folder data_xxx free_energy

To plot the radius of the precipitate in the xx direction
    $ python view.py --folder=data_xxx a10

# To view the latest time step
#     $ python view.py --folder data_xxx view_step --latest

# To view a particular time step
#     $ python view.py --folder data_xxx view_step --step 500

# To save the data to CSV files for upload to the website
#     $ python view.py --folder data_xxx save_data
"""
# pylint: disable=no-value-for-parameter

import glob
import os

from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import click
from fipy import Viewer, CellVariable, numerix
from toolz.curried import curry, pipe, do, assoc, juxt, get
import matplotlib.pyplot as plt
import progressbar
import pandas

from main import sequence, map_
# from main import get_mesh, map_, sequence, get_phase, w_func, get_coupling


@click.group()
@click.option('--folder',
              default='data',
              help="the name of the data directory to parse")
@click.pass_context
def cli(ctx, folder):
    """Use a group function to allow subcommands.
    """
    ctx.params['folder'] = folder


# def get_filename(step, latest, folder):
#     """Get the filename given the step
#     """
#     if latest:
#         return sorted(glob.glob(os.path.join(folder, 'data*.npz')))[-1]
#     return os.path.join(folder, 'data{0}.npz'.format(str(step).zfill(7)))


# @curry
# def get_var(data, name):
#     """Make a variale
#     """
#     return CellVariable(
#         get_mesh(data['params'].item()),
#         value=data[name],
#         name=name
#     )


# def get_vars(ctx, step, latest):
#     return pipe(
#         get_filename(step, latest, ctx.parent.params['folder']),
#         np.load,
#         lambda x: map_(get_var(x), ['heat', 'phase'])
#     )


# @curry
# def save_vars(save, vars):
#     if save:
#         pipe(
#             vars,
#             get(1),
#             lambda x: dict(phase=np.array(x),
#                            x=np.array(x.mesh.cellCenters[0]),
#                            y=np.array(x.mesh.cellCenters[1])),
#             pandas.DataFrame,
#             lambda x: x.to_csv('space.csv', index=False)
#         )








# @curry
# def save_contour(save, vars):
#     def reshape(arr, nx):
#         return np.array(arr).reshape((nx, nx))

#     if save:
#         pipe(
#             vars,
#             get(1),
#             lambda x: dict(phase=reshape(x, x.mesh.nx),
#                            x=reshape(x.mesh.cellCenters[0], x.mesh.nx),
#                            y=reshape(x.mesh.cellCenters[1], x.mesh.nx)),
#             lambda x: plt.contour(x['x'], x['y'], x['phase'], (0.0,)),
#             lambda x: x.collections[0].get_paths()[0].vertices,
#             lambda x: dict(x=x[:, 0], y=x[:, 1]),
#             pandas.DataFrame,
#             lambda x: x.to_csv('contour.csv', index=False)
#         )


# @cli.command()
# @click.option('--step', default=0, help='step to view')
# @click.option('--latest/--no-latest',
#               default=False,
#               help='view the latest result available')
# @click.option('--save/--no-save',
#               default=False,
#               help='save the spatial data to a CSV')
# @click.pass_context
# def view_step(ctx, step, latest, save):
#     """Use Matplotlib to view the phase and heat variables
#     """
#     return pipe(
#         get_vars(ctx, step, latest),
#         do(save_vars(save)),
#         do(save_contour(save)),
#         map_(lambda x: Viewer(x, title=x.name).plot()),
#         do(lambda _: click.pause())
#     )


def plot2d(arr):
    """Plot a 2D graph
    """
    plt.plot(arr[:, 0], arr[:, 1])
    plt.show()


# @curry
# def save2d(filename, column_names, data):
#     pandas.DataFrame(
#         dict(zip(column_names, data.transpose()))
#     ).to_csv(filename, index=False)
#     click.echo('CSV file written to {0}'.format(filename))


def pbar(items):
    pbar_ = progressbar.ProgressBar(
        widgets=[progressbar.Percentage(),
                 ' ',
                 progressbar.Bar(marker=progressbar.RotatingMarker()),
                 ' ',
                 progressbar.ETA()],
        maxval=len(items)).start()
    for counter, item in enumerate(items):
        yield item
        pbar_.update(counter + 1)
    pbar_.finish()


def read_and_calc(f_calc):
    """Read in the data and return a result
    """
    return sequence(
        lambda x: x.parent.params['folder'],
        lambda x: os.path.join(x, 'data*.npz'),
        glob.glob,
        sorted,
        pbar,
        map_(
            sequence(
                np.load,
                f_calc
            )
        ),
        list,
        np.array
    )


def read_and_plot(f_calc):
    """Read in a file and plot using f_calc
    """
    return sequence(
        read_and_calc(juxt(get('step_counter'), f_calc)),
        plot2d
    )


# def read_and_save(filename, column_names, f_calc):
#     """Read in a file and save data to CSV
#     """
#     return sequence(
#         read_and_calc(f_calc),
#         save2d(filename, column_names)
#     )


# def calc_solid_fraction(ctx):
#     return np.sum((1 + ctx['phase']) / 2) / ctx['params'].item()['nx']**2


# @cli.command()
# @click.pass_context
# def solid_fraction(ctx):
#     """Command to plot the solid fraction
#     """
#     read_and_plot(calc_solid_fraction)(ctx)



def calc_contour(arr, lx, nx):
    return pipe(
        lx / nx,
        lambda dx: np.linspace(-dx * (nx / 2 - 0.5), dx * (nx / 2 - 0.5), nx),
        lambda x: np.meshgrid(x, x),
        lambda x: plt.contour(x[0], x[1], arr.reshape(nx, nx), (0.5,)).collections[0].get_paths()[0].vertices,
        do(lambda x: plt.close())
    )


calc_position_ = lambda x: calc_contour(x['eta'], x['params'].item()['lx'], x['params'].item()['nx'])


calc_position_10 = sequence(
    calc_position_,
    lambda x: np.amax(x[:,0]),
)


calc_position_01 = sequence(
    calc_position_,
    lambda x: np.amax(x[:,1]),
)


calc_position_d = sequence(
    calc_position_,
    lambda x: np.amax((x[:,0] + x[:, 1]) / np.sqrt(2.))
)


# def f_chem(phase, heat, params):
#     return -phase**2 / 2 \
#         + phase**4 / 4 \
#         + get_coupling(params) * heat * phase * \
#         (1 - 2 * phase**2 / 3 + phase**4 / 5)


# def w_func_(phase, params):
#     return w_func(
#         numerix.arctan2(phase.grad[1], phase.grad[0]),
#         params
#     )


# def grad_mag(phase):
#     return phase.grad[1]**2 + phase.grad[0]**2


# def free_energy_func(phase, heat, params):
#     return np.array(
#         (
#             0.5 * w_func_(phase, params) * grad_mag(phase)
#             + f_chem(phase, heat, params)
#         ) * params['dx']**2
#     ).sum()


# calc_free_energy = sequence(
#     lambda x: assoc(x, 'params', x['params'].item()),
#     lambda x: assoc(x, 'mesh', get_mesh(x['params'])),
#     lambda x: assoc(x,
#                     'phase',
#                     get_phase(x['mesh'], x['params'], value=x['phase'])),
#     lambda x: assoc(x,
#                     'heat',
#                     CellVariable(mesh=x['mesh'], value=x['heat'])),
#     lambda x: free_energy_func(x['phase'], x['heat'], x['params'])
# )


@cli.command()
@click.pass_context
def a_10(ctx):
    """Command to plot the radius of the preciptate in the x direction
    """
    read_and_plot(calc_position_10)(ctx)


@cli.command()
@click.pass_context
def a_01(ctx):
    """Command to plot the radius of the preciptate in the y direction
    """
    read_and_plot(calc_position_01)(ctx)


@cli.command()
@click.pass_context
def a_d(ctx):
    """Command to plot the radius of the preciptate in the diagonal direction
    """
    read_and_plot(calc_position_d)(ctx)


# @cli.command()
# @click.option('--step', default=0, help='step to view')
# @click.option('--latest/--no-latest',
#               default=False,
#               help='view the latest result available')
# @click.option('--save/--no-save',
#               default=False,
#               help='save the spatial data to a CSV')
# @click.pass_context
# def view_step(ctx, step, latest, save):
#     """Use Matplotlib to view the phase and heat variables
#     """
#     return pipe(
#         get_vars(ctx, step, latest),
#         do(save_vars(save)),
#         do(save_contour(save)),
#         map_(lambda x: Viewer(x, title=x.name).plot()),
#         do(lambda _: click.pause())
#     )

# @cli.command()
# @click.pass_context
# def plot_contour

# @cli.command()
# @click.pass_context
# def free_energy(ctx):
#     """Command to calculate the free energy
#     """
#     read_and_plot(calc_free_energy)(ctx)


# @cli.command()
# @click.pass_context
# def save_time_data(ctx):
#     read_and_save(
#         'time.csv',
#         ['time', 'solid_fraction', 'tip_position', 'free_energy'],
#         juxt(get('elapsed_time'),
#              calc_solid_fraction,
#              calc_tip_position,
#              calc_free_energy)
#     )(ctx)


if __name__ == '__main__':
    cli()
