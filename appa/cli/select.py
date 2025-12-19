from typing import List
import glob
import os

import numpy as np
import click
from ase.io import read, write

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
    help="Output xyz file for selected structures",
)
def select(data_dir, size, bw, out):
    """Select diverse structures with MSC"""
    dataset = load_dataset(data_dir)

    if not dataset:
        click.echo("No structures found in data directory")
        return

    frames = len(dataset)
    n_atoms = len(dataset[0])

    # Determine species present in the first frame
    species_list = sorted(set(dataset[0].symbols))

    # Compute descriptors for all frames
    x = get_descriptors_multicomponent(dataset[:frames], species=species_list)
    x = x.reshape(frames, n_atoms, -1)

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
