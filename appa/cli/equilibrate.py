import click
from ase.io.trajectory import Trajectory
from ase.calculators.mixing import SumCalculator
import torch
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from pet_mad.calculator import PETMADCalculator

from appa.md import run_langevin_md
from appa.io import write_with_fixatoms, read_with_fixatoms


@click.command("equilibrate")
@click.argument("structure")
@click.option(
    "--output",
    default="equilibrated.xyz",
    help="Output name of equilibrated structure",
)
@click.option(
    "--temperature",
    default=200.0,
    help="MD temperature in K. Low temperature helps relax structure",
)
@click.option("--steps", default=1000)
@click.option(
    "--z-buffer",
    default=2.0,
    help="Wall position from maximum position of input structure",
)
@click.option(
    "--k-wall",
    default=1.0,
    help="Wall spring force constant (eV/Å^2)",
)
@click.option(
    "--traj",
    default=None,
    help="Output trajectory name",
)
def equilibrate(
    structure,
    output,
    temperature,
    steps,
    z_buffer,
    k_wall,
    traj,
):
    """Equilibrate a structure using ASE MD with PET-MAD. Uses a harmonic wall
    to make sure water molecules don't escape into the vacuum region."""
    atoms = read_with_fixatoms(structure)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    calc_MAD = PETMADCalculator(version="latest", device=device)
    dft_d3 = TorchDFTD3Calculator(device=device, xc="pbesol", damping="bj")

    combined_calc = SumCalculator([calc_MAD, dft_d3])
    atoms.calc = combined_calc

    zmax = atoms.positions[:, 2].max() + z_buffer

    atoms = run_langevin_md(
        atoms=atoms,
        temperature_K=temperature,
        steps=steps,
        zmax=zmax,
        k=k_wall,
        traj=Trajectory(traj, "w", atoms) if traj is not None else None,
        log_interval=20,
    )
    write_with_fixatoms(output, atoms)
    click.echo(f"MD finished, trajectory written to {traj}")
