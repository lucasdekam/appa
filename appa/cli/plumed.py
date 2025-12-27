import click
from ase.io import read

from appa.plumed import (
    DEFAULT_KAPPA,
    DEFAULT_STRIDE,
    DEFAULT_WARMUP,
    generate_plumed_volmer,
)


@click.group()
def plumed():
    """PLUMED input writing."""
    pass


@plumed.command("volmer")
@click.argument("xyz", type=str)
@click.option("--oxygen_id", "-o", type=int)
@click.option("--hydrogen_id", "-h", type=int)
@click.option("--surface_id", "-s", type=int)
@click.option(
    "--cv_target",
    "-t",
    type=float,
    help="Target value of the reaction coordinate ξ = d(O-H) - d(Pt-H) in Å.",
)
@click.option(
    "--kappa",
    type=float,
    default=DEFAULT_KAPPA,
    help="Umbrella strength (eV/Å^2)",
    show_default=True,
)
@click.option(
    "--stride",
    type=int,
    default=DEFAULT_STRIDE,
    help="How often to print to COLVAR",
    show_default=True,
)
@click.option(
    "--warmup",
    type=float,
    default=DEFAULT_WARMUP,
    help="Warmup period in which the CV is shifted from initial to target value",
    show_default=True,
)
def volmer(
    xyz,
    oxygen_id,
    hydrogen_id,
    surface_id,
    cv_target,
    kappa,
    stride,
    max_dist_mh,
    warmup,
):
    """Write plumed.dat input file for a Volmer step calculation."""
    generate_plumed_volmer(
        atoms=read(xyz),
        oxygen_id=oxygen_id,
        hydrogen_id=hydrogen_id,
        surface_id=surface_id,
        cv_target=cv_target,
        kappa=kappa,
        stride=stride,
        warmup=warmup,
        colvar_file=f"COLVAR_{cv_target}",
        outfile="plumed.dat",
    )
