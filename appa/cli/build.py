import click
from ase.io import write

from appa.build import Electrode, Interface


@click.command("build")
@click.option(
    "-o",
    "--output",
    default="interface.xyz",
    show_default=True,
    help="Output structure file",
)
@click.option(
    "--material",
    default="Au",
    show_default=True,
    help="Chemical element of the electrode material",
)
@click.option(
    "--size",
    nargs=3,
    type=int,
    default=(4, 4, 4),
    show_default=True,
    help="Slab dimensions (nx ny nz); ny must be even",
)
@click.option(
    "--d-water",
    type=float,
    default=10,
    show_default=True,
    help="Water layer thickness",
)
@click.option(
    "--d-vacuum",
    type=float,
    default=20,
    show_default=True,
    help="Vacuum layer thickness",
)
@click.option(
    "--a",
    type=float,
    default=None,
    show_default=True,
    help="Lattice parameter; if None takes ASE default",
)
@click.option(
    "--fix-layers",
    type=int,
    default=2,
    show_default=True,
    help="Number of layers to fix at bottom of slab",
)
@click.option(
    "--ion",
    type=str,
    default=None,
    show_default=True,
    help="Chemical element of the ion",
)
@click.option(
    "--n-ions",
    type=int,
    default=0,
    show_default=True,
    help="Number of ions to add",
)
@click.option(
    "--coverage",
    type=float,
    default=0,
    show_default=True,
    help="Hydrogen coverage",
)
@click.option(
    "--ptop",
    type=float,
    default=0.5,
    show_default=True,
    help="Probability for top site occupation; P(fcc) is 1-P(top)",
)
def build(
    output,
    material,
    size,
    d_water,
    d_vacuum,
    a,
    fix_layers,
    ion,
    n_ions,
    coverage,
    ptop,
):
    """Build an electrode-electrolyte interface."""
    electrode = Electrode(material, size, a, fix_layers)
    electrode.add_hydrogens(coverage, ptop)
    ion_dict = None
    if ion is not None and n_ions > 0:
        ion_dict = {ion: 4 for _ in range(n_ions)}
    interface = Interface(
        electrode.atoms,
        d_water=d_water,
        d_vacuum=d_vacuum,
        rho=0.95,
        ions=ion_dict,
        ion_delta_z=1.5,
    )
    atoms = interface.add_electrolyte(seed=1, verbose=True)
    write(output, atoms)
    click.echo(f"Structure written to {output}")
