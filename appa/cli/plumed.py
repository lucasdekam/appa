import click

from appa.plumed import DEFAULT_STRIDE, generate_plumed_volmer


@click.group()
def plumed():
    """PLUMED input writing."""
    pass


@plumed.command("volmer")
@click.option("--oxygen_id", "-o", type=int)
@click.option("--hydrogen_id", "-h", type=int)
@click.option("--surface_id", "-s", type=int)
@click.option(
    "--stride",
    type=int,
    default=DEFAULT_STRIDE,
    help="How often to print to COLVAR",
    show_default=True,
)
def volmer(
    oxygen_id,
    hydrogen_id,
    surface_id,
    stride,
):
    """Write plumed.dat input file for a Volmer step calculation."""
    generate_plumed_volmer(
        oxygen_id=oxygen_id,
        hydrogen_id=hydrogen_id,
        surface_id=surface_id,
        stride=stride,
        colvar_file=f"COLVAR",
        outfile="plumed.dat",
    )
