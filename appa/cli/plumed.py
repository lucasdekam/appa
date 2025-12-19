import click

from appa.plumed import generate_plumed_volmer


@click.command("plumed")
@click.argument("setup")
@click.option("--oxygen_id", "-o", type=int)
@click.option("--hydrogen_id", "-h", type=int)
@click.option("--surface_id", "-s", type=int)
@click.option(
    "--cv_target",
    "-t",
    type=float,
    help="Target value of the reaction coordinate ξ = d(O-H) - d(Pt-H) in Å.",
)
def plumed(
    setup,
    oxygen_id,
    hydrogen_id,
    surface_id,
    cv_target,
    kappa,
    stride,
):
    """Write plumed.dat input file for a Volmer step calculation."""
    if setup == "volmer":
        generate_plumed_volmer(
            oxygen_id,
            hydrogen_id,
            surface_id,
            cv_target,
            kappa,
            stride,
            colvar_file=f"COLVAR_{cv_target}",
            outfile="plumed.dat",
        )
