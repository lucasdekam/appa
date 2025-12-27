"""
VASP-related CLI commands.
"""

from pathlib import Path
import os
import yaml
import click
import numpy as np

from ase import Atoms
from ase.io import write, read
from ase.calculators.vasp import Vasp
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor


def print_percentiles(name, data, percentiles=(0, 1, 25, 50, 75, 99, 100)):
    vals = np.percentile(data, percentiles)
    click.echo(f"\n{name} percentiles:")
    for p, v in zip(percentiles, vals):
        click.echo(f"  {p:>3}% : {v:7.2f}")


def print_histogram(name, data, bins=20, width=40):
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / hist.max()  # normalize for bar width

    click.echo(f"\n{name} histogram:")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "█" * int(np.ceil(h * width))
        click.echo(f"{lo:7.2f} - {hi:7.2f} | {bar}")


def read_and_check_results(directory: Path) -> list[Atoms]:
    configs = []

    # Collect configurations
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue

        vasprun_path = subdir / "vasprun.xml"
        if not vasprun_path.exists():
            click.echo(f"INFO: Missing vasprun.xml in {subdir}, skipping.")
            continue

        try:
            vasprun = Vasprun(str(vasprun_path))
        except Exception as e:
            click.echo(f"INFO: Failed to parse {vasprun_path}: {e}, skipping.")
            continue

        if not vasprun.converged_electronic:
            click.echo(f"WARNING: SCF not converged in {subdir}, skipping.")
            continue

        try:
            atoms = AseAtomsAdaptor.get_atoms(vasprun.final_structure)
        except Exception as e:
            click.echo(f"INFO: Failed to convert structure in {subdir}: {e}")
            continue

        atoms.arrays["DFT_forces"] = vasprun.ionic_steps[-1]["forces"]
        atoms.info["DFT_energy"] = float(vasprun.final_energy)

        click.echo(f"{subdir}: E={atoms.info['DFT_energy']:.3f} eV")

        configs.append(atoms)

    return configs


def filter_energy_force(
    configs: list[Atoms], fmax: float, emax_fraction: float
) -> list[Atoms]:
    energies = np.array([a.info["DFT_energy"] for a in configs])
    max_forces = np.array(
        [np.linalg.norm(a.arrays["DFT_forces"], axis=1).max() for a in configs]
    )

    print_percentiles("Energy (eV)", energies)
    print_histogram("Energy (eV)", energies)

    print_percentiles("Max |force| (eV/Å)", max_forces)
    print_histogram("Max |force| (eV/Å)", max_forces)

    # Filtering criteria
    mean_energy = energies.mean()
    e_max = mean_energy * (1 - emax_fraction)  # energy always negative

    click.echo("\nFiltering criteria:")
    click.echo(f"  Energy <= {e_max:.3f} eV")
    click.echo(f"  Max |force| <= {fmax:.1f} eV/Å")

    filtered = []
    for a in configs:
        e = a.info["DFT_energy"]
        f = np.abs(a.arrays["DFT_forces"]).max()

        if not (e <= e_max):
            continue
        if f > fmax:
            continue

        filtered.append(a)

    return filtered


@click.group()
def vasp():
    """VASP batch job utilities."""
    pass


@vasp.command("collect")
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    default="collected.xyz",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output extxyz file.",
)
@click.option(
    "--fmax",
    default=10,
    show_default=True,
    type=float,
    help="Maximum allows force magnitude in eV/Å",
)
@click.option(
    "--emax",
    default=0.1,
    show_default=True,
    type=float,
    help="Maximum deviation from mean energy as a fraction of mean energy",
)
def collect(directory: Path, output: Path, fmax: float, emax: float):
    """
    Collect VASP outputs, analyze energy/force distributions,
    filter outliers, and write to extxyz.
    """
    configs = read_and_check_results(directory)
    if not configs:
        click.echo("No converged configurations found.")
        return
    click.echo(f"\nCollected {len(configs)} configurations")
    filtered = filter_energy_force(configs, fmax, emax)
    click.echo(
        f"Kept {len(filtered)} / {len(configs)} configurations "
        f"({len(configs) - len(filtered)} filtered)"
    )

    if not filtered:
        click.echo("No configurations left after filtering.")
        return

    # Write output
    write(
        output,
        filtered,
        format="extxyz",
        columns=["symbols", "positions", "DFT_forces"],
        write_results=False,
    )

    click.echo(f"Wrote {len(filtered)} configurations to {output}")


@vasp.command("input")
@click.option(
    "--xyz",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="XYZ file with configurations to be labelled.",
)
@click.option(
    "--index",
    required=True,
    type=int,
    help="Index of the configuration (e.g. SLURM array index).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for VASP input files.",
)
@click.option(
    "--params",
    default="params.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML file with VASP parameters.",
)
@click.option(
    "--ncore",
    default=8,
    show_default=True,
    type=int,
    help="VASP NCORE.",
)
@click.option(
    "--kpar",
    default=1,
    show_default=True,
    type=int,
    help="VASP KPAR.",
)
def input(
    xyz: Path,
    index: int,
    output_dir: Path,
    params: Path,
    ncore: int,
    kpar: int,
):
    """
    Generate VASP input files for a single structure (specified by xyz and
    index). Requires a config .yaml file with sections specified by config_type,
    and an ASE-readable XYZ extended file where all structures have a
    config_type in their info section.
    """
    # Set pseudopotential path for ASE
    os.environ.setdefault("VASP_PP_PATH", os.environ["HOME"] + "/vasp/pps")

    atoms = read(xyz, index=index)
    with open(params, "r", encoding="utf-8") as f:
        yaml_params: dict = yaml.safe_load(f)
        vasp_params: dict = yaml_params[atoms.info["config_type"]].copy()

    # Reciprocal lattice lengths including 2pi (Å⁻¹)
    rec_lengths = np.linalg.norm(atoms.cell.reciprocal(), axis=1) * 2 * np.pi

    kspacing = vasp_params.get("kspacing", None)
    is_slab = vasp_params.get("ldipol", False)
    click.echo(f"Slab geometry? {is_slab}")

    if kspacing is not None:
        click.echo(
            "Reciprocal lengths (Å⁻¹), incl. 2pi: "
            f"a={rec_lengths[0]:.3f}, "
            f"b={rec_lengths[1]:.3f}, "
            f"c={rec_lengths[2]:.3f}"
        )
        click.echo(f"KSPACING = {kspacing:.3f}")

        # Generate k-point grid explicitly
        kpts = np.maximum(1, np.ceil(rec_lengths / kspacing).astype(int))

        if is_slab:
            click.echo("Detected slab (ldipol=True): forcing Nkz = 1 and gamma=True")
            kpts[2] = 1
            vasp_params["gamma"] = True

        vasp_params["kpts"] = tuple(int(k) for k in kpts)

        # Remove kspacing so VASP does not regenerate k-points
        vasp_params.pop("kspacing", None)

        click.echo(f"Using explicit k-points: {vasp_params['kpts']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_slab:
        click.echo("Calculating dipole position...")
        vasp_params["dipol"] = [
            0.5,
            0.5,
            (atoms.positions[:, 2].mean() / atoms.cell[2, 2]).item(),
        ]

    calc = Vasp(
        directory=str(output_dir),
        ncore=ncore,
        kpar=kpar,
        **vasp_params,
    )

    click.echo(
        f"Writing input for config {index} of {xyz} to {output_dir}\n"
        f"params={vasp_params}"
    )

    calc.write_input(atoms)
