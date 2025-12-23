from typing import List, Optional
import glob
import os

import numpy as np
import click
from ase.io import read, write
from ase.cell import Cell

from quests.descriptor import get_descriptors_multicomponent
from quests.entropy import entropy
from quests.compression.fps import msc


def load_dataset(data_dir: str) -> List:
    dataset = []
    for pattern in ["*.extxyz", "*.xyz", "*.traj"]:
        for fname in glob.glob(os.path.join(data_dir, pattern)):
            loaded = read(fname, ":")
            if not isinstance(loaded, list):
                loaded = [loaded]
            dataset.extend(loaded)
    return dataset


def filter_by_species(dataset, allowed_species: Optional[List[str]]):
    if allowed_species is None:
        return dataset

    allowed = set(allowed_species)

    filtered = []
    for atoms in dataset:
        symbols = set(atoms.symbols)
        if symbols.issubset(allowed):
            filtered.append(atoms)

    return filtered


def filter_by_vacuum(dataset, d_vacuum: float):
    """
    Keep only structures where the top d_vacuum Å of the cell is empty.
    """
    if d_vacuum is None:
        return dataset

    filtered = []
    for atoms in dataset:
        cell: Cell = atoms.cell
        a, b, c, _, _, _ = cell.cellpar()
        z_positions = atoms.positions[:, 2]

        if np.isclose(a, b, rtol=0.05) and np.isclose(a, c):
            filtered.append(atoms)
        elif z_positions.max() <= c - d_vacuum:
            filtered.append(atoms)

    return filtered


@click.command()
@click.option(
    "--data-dir",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Directory with .extxyz/.xyz/.traj files",
)
@click.option(
    "--size",
    type=int,
    default=100,
    help="Number of structures to select",
)
@click.option(
    "--bw",
    type=float,
    default=0.065,
    help="Bandwidth for entropy estimation kernel",
)
@click.option(
    "--out",
    "-o",
    default="selected.xyz",
)
@click.option(
    "--species",
    "-s",
    multiple=True,
    type=str,
    help="Allowed species (e.g. -s O -s H -s Pt)",
)
@click.option(
    "--d-vacuum",
    type=float,
    default=None,
    help="Require the top d_vacuum (Å) of the cell to be empty (slab vacuum check)",
)
def select(data_dir, size, bw, out, species, d_vacuum):
    dataset = load_dataset(data_dir)

    if not dataset:
        click.echo("No structures found in data directory")
        return

    if species:
        dataset = filter_by_species(dataset, list(species))
        click.echo(
            f"Dataset filtered to {len(dataset)} structures "
            f"using species={list(species)}"
        )

    if d_vacuum is not None:
        dataset = filter_by_vacuum(dataset, d_vacuum)
        click.echo(
            f"Dataset filtered to {len(dataset)} structures "
            f"with ≥ {d_vacuum:.2f} Å vacuum on top"
        )

    if not dataset:
        click.echo("No structures left after filtering")
        return

    frames = len(dataset)
    n_atoms = len(dataset[0])

    # Determine species present (after filtering)
    species_list = sorted(set(dataset[0].symbols))

    # Compute descriptors for all frames
    x = [
        get_descriptors_multicomponent([s], species=species_list).reshape(len(s), -1)
        for s in dataset
    ]
    # x = x.reshape(frames, n_atoms, -1)

    # Compute entropies (one value per structure)
    entropies = [entropy(element, h=bw) for element in x]

    # Run selection
    selected_indices = msc(
        descriptors=[ele for ele in x],
        entropies=np.array(entropies),
        size=size,
        h=bw,
    )

    selected_structures = [dataset[i] for i in selected_indices]

    # Write output
    write(out, images=selected_structures)
    click.echo(f"Wrote {len(selected_structures)} structures to {out}")
