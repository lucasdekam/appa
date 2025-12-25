from ase.io import read
import pandas as pd
import click
from pathlib import Path


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
    help="Subtract reference energies from structures with config_type='IsolatedAtom'",
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

            is_isolated = atoms.info.get("config_type") == "IsolatedAtom"
            energy = atoms.info.get("DFT_energy")

            if is_isolated:
                if len(atoms) != 1:
                    raise click.ClickException(
                        "IsolatedAtom config must contain exactly one atom"
                    )
                if energy is None:
                    raise click.ClickException(
                        "IsolatedAtom config is missing DFT_energy"
                    )

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
        if atoms.info.get("config_type") == "IsolatedAtom":
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
        is_isolated = atoms.info.get("config_type") == "IsolatedAtom"
        energy = atoms.info.get("DFT_energy")

        if is_isolated:
            if len(atoms) != 1:
                raise click.ClickException(
                    "IsolatedAtom config must contain exactly one atom"
                )
            if energy is None:
                raise click.ClickException("IsolatedAtom config is missing DFT_energy")

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
