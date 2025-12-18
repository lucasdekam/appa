import click

from appa.build import Electrode, Interface
from appa.io import write_with_fixatoms


@click.command("build")
@click.option(
    "-o",
    "--output",
    default="interface.xyz",
    help="Output structure file",
)
@click.option(
    "--material",
    default="Au",
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
    help="Water layer thickness",
)
@click.option(
    "--d-vacuum",
    type=float,
    default=20,
    help="Vacuum layer thickness",
)
@click.option(
    "--a",
    default=None,
    help="Lattice parameter",
)
@click.option(
    "--fix-layers",
    type=int,
    default=2,
    help="Number of layers to fix at bottom of slab",
)
@click.option(
    "--ion",
    type=str,
    default=None,
    help="Chemical element of the ion",
)
@click.option(
    "--n-ions",
    type=int,
    default=0,
    help="Number of ions to add",
)
def build(output, material, size, d_water, d_vacuum, a, fix_layers, ion, n_ions):
    """Build an electrode-electrolyte interface."""
    electrode = Electrode(material, size, a, fix_layers)
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
    write_with_fixatoms(output, atoms)
    click.echo(f"Structure written to {output}")
