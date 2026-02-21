from pathlib import Path
from glob import glob
from multiprocessing import Pool, cpu_count
import os

import pandas as pd
import click
from ase.io import read, write
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysis.transformations.boxdimensions import set_dimensions


@click.group()
def convert():
    """File conversion utilities."""
    pass


@convert.command("xyz2grace")
@click.argument(
    "xyz_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "out_file",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--subtract-reference",
    is_flag=True,
    help="Subtract reference energies from isolated atom configs",
)
def xyz2grace(xyz_file, out_file, subtract_reference):
    """Convert an extxyz file to a GRACE-compatible DataFrame."""
    atoms_list = read(xyz_file, index=":")
    records = []

    isolated = {}
    elements = set()

    # Collect isolated-atom reference energies
    if subtract_reference:
        for atoms in atoms_list:
            symbols = atoms.get_chemical_symbols()
            elements.update(symbols)

            is_isolated = len(atoms) == 1
            energy = atoms.info.get("DFT_energy")

            if is_isolated:
                if energy is None:
                    raise click.ClickException("Isolated atom is missing DFT_energy")

                isolated[symbols[0]] = energy

        missing = elements - isolated.keys()
        if missing:
            raise click.ClickException(
                "Missing isolated-atom reference energies for elements: "
                + ", ".join(sorted(missing))
            )

        click.echo("INFO: Found isolated-atom reference energies:")
        for symbol in sorted(isolated):
            click.echo(f"  {symbol:<2} : {isolated[symbol]: .6f} eV")

    # Build GRACE dataframe records (skip isolated atoms)
    for atoms in atoms_list:
        if len(atoms) == 1:
            continue

        energy = atoms.info.get("DFT_energy")
        forces = atoms.arrays.get("DFT_forces")

        if energy is None:
            raise click.ClickException("Missing DFT_energy in configuration")
        if forces is None:
            raise click.ClickException("Missing DFT_forces in configuration")

        if subtract_reference:
            atom_refs = sum(isolated[symbol] for symbol in atoms.get_chemical_symbols())
            energy_corrected = energy - atom_refs
        else:
            energy_corrected = energy

        records.append(
            dict(
                ase_atoms=atoms,
                energy=energy,
                energy_corrected=energy_corrected,
                forces=forces,
            )
        )

    df = pd.DataFrame(records)
    df.to_pickle(out_file, compression="gzip")

    click.echo(f"Wrote {len(df)} configurations to {out_file}")


@convert.command("extract-isolated")
@click.argument(
    "xyz_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "out_xyz",
    type=click.Path(dir_okay=False, path_type=Path),
)
def extract_isolated(xyz_file, out_xyz):
    """
    Extract isolated-atom reference energies from an XYZ file and
    write a new XYZ file without the isolated-atom configurations.
    """
    atoms_list = read(xyz_file, index=":")

    isolated = {}
    kept_atoms = []

    for atoms in atoms_list:
        is_isolated = len(atoms) == 1
        energy = atoms.info.get("DFT_energy")

        if is_isolated:
            if energy is None:
                raise click.ClickException("Isolated atom is missing DFT_energy")

            symbol = atoms.get_chemical_symbols()[0]
            isolated[symbol] = energy
        else:
            kept_atoms.append(atoms)

    # Print dictionary summary
    if isolated:
        click.echo("Isolated-atom reference energies:")
        click.echo(
            "{"
            + ", ".join(
                f"'{sym}': {energy:.6f}" for sym, energy in sorted(isolated.items())
            )
            + "}"
        )
    else:
        click.echo("No isolated-atom configurations found.")

    # Write filtered XYZ
    if not kept_atoms:
        raise click.ClickException(
            "All configurations were isolated atoms; nothing to write."
        )

    from ase.io import write

    write(out_xyz, kept_atoms, format="extxyz", write_results=False)
    click.echo(
        f"Wrote {len(kept_atoms)} configurations (isolated atoms removed) to {out_xyz}"
    )


def process_xyz(file):
    """Convert XYZ trajectory to XTC."""
    directory = os.path.dirname(file)
    click.echo(f"Processing XYZ: {file}")

    # Generate system.data once for consistency
    atoms = read(file, index=0)
    lammps_data = os.path.join(directory, "system.data")

    write(lammps_data, atoms, format="lammps-data", masses=True)
    click.echo(f"  Wrote system.data")

    u = mda.Universe(
        file,
        file,
        topology_format="XYZ",
        format="XYZ",
        transformations=[set_dimensions(atoms.cell.cellpar())],
    )

    xtc_path = os.path.splitext(file)[0] + ".xtc"

    with XTCWriter(xtc_path, n_atoms=u.atoms.n_atoms, precision=5) as W:
        for frame, _ in enumerate(u.trajectory, 1):
            W.write(u.atoms)
            if frame % 10000 == 0 or frame == 1:
                click.echo(f"    Wrote frame {frame}")

    click.echo(f"  Finished writing {xtc_path}")


def process_lammpsdump(file):
    """Convert LAMMPS dump trajectory to XTC."""
    directory = os.path.dirname(file)
    click.echo(f"Processing LAMMPS dump: {file}")

    system_data = os.path.join(directory, "system.data")
    if not os.path.exists(system_data):
        raise FileNotFoundError(
            f"system.data not found in {directory}. Required for topology."
        )

    u = mda.Universe(
        system_data,
        file,
        topology_format="DATA",
        format="LAMMPSDUMP",
        atom_style="id type x y z",
    )

    xtc_path = os.path.splitext(file)[0] + ".xtc"

    with XTCWriter(xtc_path, n_atoms=u.atoms.n_atoms, precision=5) as W:
        for frame, _ in enumerate(u.trajectory, 1):
            W.write(u.atoms)
            if frame % 10000 == 0 or frame == 1:
                click.echo(f"    Wrote frame {frame}")

    click.echo(f"  Finished writing {xtc_path}")


def worker(args):
    file, fmt = args
    if fmt == "xyz":
        process_xyz(file)
    elif fmt == "lammpsdump":
        process_lammpsdump(file)


@convert.command("xtc")
@click.option(
    "--pattern",
    required=True,
    help="Glob pattern for input files (e.g., './*/*.xyz').",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["xyz", "lammpsdump"]),
    required=True,
    help="Input format.",
)
@click.option(
    "--nprocs",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel processes.",
)
def convert_to_xtc(pattern, fmt, nprocs):
    """Convert XYZ or LAMMPS dump trajectories to XTC."""
    files = sorted(glob(pattern))

    if not files:
        click.echo("No files matched pattern.", err=True)
        return

    click.echo(f"Found {len(files)} {fmt} files.")

    jobs = [(file, fmt) for file in files]

    if nprocs > 1:
        with Pool(processes=min(nprocs, cpu_count())) as pool:
            pool.map(worker, jobs)
    else:
        for job in jobs:
            worker(job)
