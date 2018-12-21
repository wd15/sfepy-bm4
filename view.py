"""View results for benchmark problem 4

Usage: view.py [OPTIONS] COMMAND [ARGS]...

  Use a group function to allow subcommands.

Options:
  --folder TEXT  the name of the data directory to parse
  --help         Show this message and exit.

Commands:
  a_01                 Command to plot the radius of the preciptate...
  a_10                 Command to plot the radius of the preciptate...
  a_d                  Command to plot the radius of the preciptate...
  bulk_free_energy     Command to plot the elastic free energy
  elastic_free_energy  Command to plot the elastic free energy
  params               Print out the parameters used for the...

"""
# pylint: disable=no-value-for-parameter
# pylint: disable=invalid-name

import glob
import os
import pprint

import numpy as np
import click
from toolz.curried import pipe, do, assoc, juxt, get
import matplotlib.pyplot as plt
import progressbar

from main import sequence, map_, calc_elastic_f, calc_bulk_f

# from main import get_mesh, map_, sequence, get_phase, w_func, get_coupling


@click.group()
@click.option(
    "--folder", default="data", help="the name of the data directory to parse"
)
@click.pass_context
def cli(ctx, folder):
    """Use a group function to allow subcommands.
    """
    ctx.params["folder"] = folder


def get_filename(step, latest, folder):
    """Get the filename given the step
    """
    if latest:
        return sorted(glob.glob(os.path.join(folder, "data*.npz")))[-1]
    return os.path.join(folder, "data{0}.npz".format(str(step).zfill(7)))


def plot2d(arr):
    """Plot a 2D graph
    """
    plt.plot(arr[:, 0], arr[:, 1])
    plt.show()


def pbar(items):
    """Add a progress bar to iterate over items
    """
    pbar_ = progressbar.ProgressBar(
        widgets=[
            progressbar.Percentage(),
            " ",
            progressbar.Bar(marker=progressbar.RotatingMarker()),
            " ",
            progressbar.ETA(),
        ],
        maxval=len(items),
    ).start()
    for counter, item in enumerate(items):
        yield item
        pbar_.update(counter + 1)
    pbar_.finish()


def read_and_calc(f_calc):
    """Read in the data and return a result
    """
    return sequence(
        lambda x: x.parent.params["folder"],
        lambda x: os.path.join(x, "data*.npz"),
        glob.glob,
        sorted,
        pbar,
        map_(sequence(np.load, f_calc)),
        list,
        np.array,
    )


def read_and_plot(f_calc):
    """Read in a file and plot using f_calc
    """
    return sequence(read_and_calc(juxt(get("step_counter"), f_calc)), plot2d)


def calc_contour(arr, param_lx, param_nx):
    """Get contour points for 0.5 contour
    """
    return pipe(
        param_lx / param_nx,
        lambda dx: np.linspace(
            -dx * (param_nx / 2 - 0.5), dx * (param_nx / 2 - 0.5), param_nx
        ),
        lambda x: np.meshgrid(x, x),
        lambda x: plt.contour(x[0], x[1], arr.reshape(param_nx, param_nx), (0.5,))
        .collections[0]
        .get_paths()[0]
        .vertices,
        do(lambda x: plt.close()),
    )


def calc_position_(data):
    """Calculate the postion of the interface

    Args:
      data: data dictionary

    Returns:
      the contour points as a numpy array
    """
    return calc_contour(
        data["eta"], data["params"].item()["lx"], data["params"].item()["nx"]
    )


calc_position_10 = sequence(calc_position_, lambda x: np.amax(x[:, 0]))


calc_position_01 = sequence(calc_position_, lambda x: np.amax(x[:, 1]))


calc_position_d = sequence(
    calc_position_, lambda x: np.amax((x[:, 0] + x[:, 1]) / np.sqrt(2.))
)


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


@cli.command()
@click.pass_context
def elastic_free_energy(ctx):
    """Command to plot the elastic free energy
    """
    read_and_plot(calc_elastic_free_energy)(ctx)


calc_elastic_free_energy = sequence(
    lambda x: assoc(x, "params", x["params"].item()),
    lambda x: assoc(x, "dx", x["params"]["lx"] / x["params"]["nx"]),
    lambda x: assoc(x, "total_strain", dict(e11=x["e11"], e22=x["e22"], e12=x["e12"])),
    lambda x: calc_elastic_f(x["params"], x["total_strain"], x["eta"]) * x["dx"] ** 2,
    np.sum,
)


@cli.command()
@click.pass_context
def bulk_free_energy(ctx):
    """Command to plot the elastic free energy
    """
    read_and_plot(calc_bulk_free_energy)(ctx)


calc_bulk_free_energy = sequence(
    lambda x: assoc(x, "params", x["params"].item()),
    lambda x: assoc(x, "dx", x["params"]["lx"] / x["params"]["nx"]),
    lambda x: calc_bulk_f(x["eta"]) * x["dx"] ** 2,
    np.sum,
)


@cli.command()
@click.pass_context
def params(ctx):
    """Print out the parameters used for the simulation
    """
    pipe(
        get_filename(1, False, ctx.parent.params["folder"]),
        np.load,
        lambda x: pprint.PrettyPrinter(indent=2).pprint(x["params"].item()),
    )


if __name__ == "__main__":
    cli()
