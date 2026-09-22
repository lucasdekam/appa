"""
Interactive viewing of trajectories. Not meant to be scripted: this opens the
ASE GUI so a person can look at a run.
"""

from pathlib import Path

import click
import numpy as np


TOPOLOGY_FILENAME = "system.data"


def symbols_from_masses(masses, tolerance: float = 0.5):
    """
    Guess chemical symbols from a list of atomic masses.

    An XTC file stores neither elements nor types, and a LAMMPS data file
    stores types and masses but no element names, so the masses are the only
    thing left to identify the species by.

    Parameters
    ----------
    masses : array_like
        Atomic masses in amu.
    tolerance : float
        Largest accepted deviation from a tabulated mass, in amu. A larger
        deviation means the topology is probably not what it is assumed to be,
        and is reported rather than silently rounded to the nearest element.

    Returns
    -------
    list[str]
        Chemical symbols, one per mass.
    """
    from ase.data import atomic_masses, chemical_symbols

    # atomic_masses[0] is NaN (the X dummy species), so search from hydrogen
    table = np.asarray(atomic_masses[1:])
    symbols = []
    worst = 0.0
    worst_mass = None

    for mass in np.asarray(masses, dtype=float):
        index = int(np.argmin(np.abs(table - mass)))
        deviation = abs(table[index] - mass)
        if deviation > worst:
            worst, worst_mass = deviation, mass
        symbols.append(chemical_symbols[index + 1])

    if worst > tolerance:
        raise click.ClickException(
            f"Could not identify an element with mass {worst_mass:.4f} amu "
            f"(nearest tabulated mass is {worst:.4f} amu away). "
            f"Is the topology file the right one for this trajectory?"
        )

    return symbols


def read_xtc(trajectory, topology, start: int = 0, stop=None, every: int = 1):
    """
    Read an XTC trajectory into a list of ASE Atoms.

    Parameters
    ----------
    trajectory : os.PathLike
        Path to the .xtc file.
    topology : os.PathLike
        Path to the LAMMPS ``system.data`` file that gives the species.
    start, stop, every : int
        Frame slice, as in ``range``.

    Returns
    -------
    list[ase.Atoms]
    """
    import MDAnalysis as mda
    from ase import Atoms
    from ase.cell import Cell

    universe = mda.Universe(
        str(topology),
        str(trajectory),
        topology_format="DATA",
        format="XTC",
        atom_style="id type x y z",
    )

    symbols = symbols_from_masses(universe.atoms.masses)

    images = []
    for _ in universe.trajectory[start:stop:every]:
        images.append(
            Atoms(
                symbols=symbols,
                positions=universe.atoms.positions.copy(),
                cell=Cell.fromcellpar(universe.dimensions),
                pbc=True,
            )
        )

    return images


@click.command("view")
@click.argument(
    "trajectory",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--topology",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"LAMMPS data file with the species. Default: {TOPOLOGY_FILENAME} "
    "next to the trajectory.",
)
@click.option(
    "--start",
    type=int,
    default=0,
    show_default=True,
    help="First frame to load.",
)
@click.option(
    "--stop",
    type=int,
    default=None,
    help="Stop before this frame. Default: the end of the trajectory.",
)
@click.option(
    "--every",
    type=int,
    default=1,
    show_default=True,
    help="Load every Nth frame.",
)
@click.option(
    "--max-frames",
    type=int,
    default=2000,
    show_default=True,
    help="Refuse to load more frames than this, since the GUI becomes "
    "unusable and the images are held in memory. Use 0 for no limit.",
)
def view(trajectory, topology, start, stop, every, max_frames):
    """
    View an XTC trajectory in the ASE GUI.

    An XTC file holds only positions and the box, so the species are taken
    from a LAMMPS data file, by default the system.data written next to the
    trajectory by `appa convert xtc`.
    """
    from ase.visualize import view as ase_view

    if topology is None:
        topology = trajectory.parent / TOPOLOGY_FILENAME
        if not topology.exists():
            raise click.ClickException(
                f"No {TOPOLOGY_FILENAME} next to {trajectory}. "
                "Pass one with --topology."
            )

    if every < 1:
        raise click.ClickException("--every must be at least 1.")

    click.echo(f"Reading {trajectory} with topology {topology}")
    images = read_xtc(trajectory, topology, start=start, stop=stop, every=every)

    if not images:
        raise click.ClickException(
            "The requested frame range is empty; check --start, --stop and --every."
        )

    if max_frames and len(images) > max_frames:
        raise click.ClickException(
            f"{len(images)} frames selected, more than --max-frames {max_frames}. "
            f"Thin the trajectory with --every {int(np.ceil(len(images) / max_frames))} "
            "or raise the limit."
        )

    formula = images[0].get_chemical_formula()
    click.echo(f"Loaded {len(images)} frames of {formula}")
    ase_view(images)
