import click

from appa.plumed import generate_plumed_volmer


@click.group()
def plumed():
    """PLUMED input writing."""
    pass


@plumed.command("volmer")
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
    default=4.0,
    help="Umbrella strength (eV/Å^2)",
)
@click.option(
    "--stride",
    type=int,
    default=10,
    help="How often to print to COLVAR",
)
@click.option(
    "--max-dist-mh",
    type=float,
    default=2.8,
    help="Maximum M-H distance, enforced by upper wall constraint",
)
def volmer(
    oxygen_id,
    hydrogen_id,
    surface_id,
    cv_target,
    kappa,
    stride,
    max_dist_mh,
):
    """Write plumed.dat input file for a Volmer step calculation."""
    generate_plumed_volmer(
        oxygen_id,
        hydrogen_id,
        surface_id,
        cv_target,
        kappa,
        stride,
        max_dist_mh,
        colvar_file=f"COLVAR_{cv_target}",
        outfile="plumed.dat",
    )
