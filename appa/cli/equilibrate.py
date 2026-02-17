import click
from ase.io.trajectory import Trajectory
from ase.calculators.mixing import SumCalculator
from ase.io import write, read

from appa.md import run_langevin_md


@click.command("equilibrate")
@click.argument("structure")
@click.argument("model")
@click.option(
    "--output",
    default="equilibrated.xyz",
    help="Output name of equilibrated structure",
)
@click.option(
    "--temperature",
    default=300.0,
    help="MD temperature in K.",
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
    model,
    output,
    temperature,
    steps,
    z_buffer,
    k_wall,
    traj,
):
    """Equilibrate a structure using ASE MD with a given compiled NequIP model.
    Uses a harmonic wall to make sure water molecules don't escape into
    the vacuum region."""
    atoms = read(structure)
    click.echo(f"Loaded structure: {structure}")

    try:
        import torch
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        from nequip.ase import NequIPCalculator

    except ImportError as e:
        raise ImportError(
            "Please install nequip and torch-dftd to run this script."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    calc_mlp = NequIPCalculator.from_compiled_model(compile_path=model, device=device)
    dft_d3 = TorchDFTD3Calculator(device=device, xc="pbe", damping="bj", cutoff=10)
    combined_calc = SumCalculator([calc_mlp, dft_d3])
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
    write(output, atoms)
    click.echo(f"MD finished, equilibrated structure written to {output}")
