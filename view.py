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
import warnings

import numpy as np
import click
from toolz.curried import pipe, do, assoc, juxt, get, take_nth, curry
import progressbar
import pandas


warnings.simplefilter("ignore")
# noqa: E402
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position;  # noqa: E402
from main import (  # pylint: disable=wrong-import-position;  # noqa: E402
    sequence,
    map_,
    calc_elastic_f,
    calc_bulk_f,
    set_eta,
)

# pylint: disable=wrong-import-position
from fipy_module import get_mesh, get_vars  # noqa: E402


@click.group()
@click.option(
    "--folder", default="data", help="the name of the data directory to parse"
)
@click.option("--frequency", default=1, help="the display frequency")
@click.pass_context
def cli(ctx, folder, frequency):
    """Use a group function to allow subcommands.
    """
    ctx.params["folder"] = folder
    ctx.params["frequency"] = frequency


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


@curry
def read_and_calc(f_calc, ctx):
    """Read in the data and return a result
    """
    return pipe(
        ctx.parent.params["folder"],
        lambda x: os.path.join(x, "data*.npz"),
        glob.glob,
        sorted,
        take_nth(ctx.parent.params["frequency"]),
        list,
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


calc_dx2 = lambda x: (x["lx"] / x["nx"]) ** 2


calc_elastic_free_energy = sequence(
    lambda x: assoc(x, "params", x["params"].item()),
    lambda x: assoc(x, "dx", x["params"]["lx"] / x["params"]["nx"]),
    lambda x: assoc(x, "total_strain", dict(e11=x["e11"], e22=x["e22"], e12=x["e12"])),
    lambda x: calc_elastic_f(x["params"], x["total_strain"], x["eta"])
    * calc_dx2(x["params"]),
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
    lambda x: calc_bulk_f(x["eta"]) * calc_dx2(x["params"]),
    np.sum,
)


@cli.command()
@click.pass_context
def gradient_free_energy(ctx):
    """Command to plot the gradient free energy
    """
    read_and_plot(calc_gradient_free_energy)(ctx)


def calc_gradient_free_energy(data):
    """Calculate the gradient free energy for one time step

    Args:
      data: dictionary of data from a output file for given time step

    Returns:
      a float representing the gradient free energy for a given time
      step
    """
    func = sequence(
        lambda x: get_vars(x, set_eta(data["eta"]), get_mesh(x)),
        get("eta"),
        lambda x: x.grad.mag ** 2,
    )
    return pipe(
        data["params"].item(),
        lambda x: assoc(x, "dx", x["lx"] / x["nx"]),
        lambda x: func(x) * (x["kappa"] / 2) * calc_dx2(x),
        np.array,
        np.sum,
    )


@cli.command()
@click.pass_context
def total_free_energy(ctx):
    """Command to plot the gradient free energy
    """
    read_and_plot(calc_total_free_energy)(ctx)


calc_total_free_energy = sequence(
    juxt(calc_bulk_free_energy, calc_gradient_free_energy, calc_elastic_free_energy),
    sum,
)


@cli.command()
@click.pass_context
def total_area(ctx):
    """Command to plot the gradient free energy
    """
    read_and_plot(calc_total_area)(ctx)


calc_total_area = sequence(lambda x: x["eta"] * calc_dx2(x["params"].item()), np.sum)


@cli.command()
@click.pass_context
def g_el(ctx):
    """Command to plot the gradient free energy
    """
    read_and_plot(calc_g_el)(ctx)


calc_g_el = sequence(
    juxt(calc_elastic_free_energy, calc_total_area), lambda x: x[1] / x[0]
)


@cli.command()
@click.pass_context
def g_grad(ctx):
    """Command to plot the gradient free energy
    """
    read_and_plot(calc_g_grad)(ctx)


calc_g_grad = sequence(
    juxt(calc_gradient_free_energy, calc_total_area), lambda x: x[1] / x[0]
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


@cli.command()
@click.option("--step", default=0, help="step to view")
@click.option(
    "--latest/--no-latest", default=False, help="view the latest result available"
)
@click.pass_context
def contour(ctx, step, latest):
    """Plot the 0.5 contour
    """
    pipe(
        get_filename(step, latest, ctx.parent.params["folder"]),
        np.load,
        calc_position_,
        plot2d,
    )


@cli.command()
@click.pass_context
def save_time_data(ctx):
    """Dump all the time data to a CSV dat file

    All the time data with each column a differnt qunatity and each
    row a different time step.

    Args:
      ctx: the Click context from the base command
    """
    read_and_save(
        "time.csv",
        [
            "time",
            "a_01",
            "a_10",
            "a_d",
            "elastic_free_energy",
            "gradient_free_energy",
            "bulk_free_energy",
            "precipitate_area",
        ],
        juxt(
            calc_elapsed_time,
            calc_position_01,
            calc_position_10,
            calc_position_d,
            calc_elastic_free_energy,
            calc_gradient_free_energy,
            calc_bulk_free_energy,
            calc_total_area,
        ),
    )(ctx)


def calc_elapsed_time(data):
    """Calculate the elapsed time

    Given the data dictionary from on time step, use the step count and time step size
    to calculate the elapsed time

    Args:
      data: the data dictionary from one output file

    Retuns:
      the elapsed timeimport ipdb; ipdb.set_trace()
    """
    return data["step_counter"] * data["params"].item()["dt"]


def read_and_save(filename, column_names, f_calc):
    """Read in a file and save data to CSV
    """
    return sequence(read_and_calc(f_calc), save2d(filename, column_names))


@curry
def save2d(filename, column_names, data):
    """Save data to a CSV file

    Args:
      filename: the namve of the CSV dile
      column_names: names of the data columns
      data: 2D data arrys with columns in same oreder as column_names
    """
    pandas.DataFrame(dict(zip(column_names, data.transpose()))).to_csv(
        filename, index=False
    )
    click.echo("CSV file written to {0}".format(filename))


if __name__ == "__main__":
    cli()
