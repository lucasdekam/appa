import click
from ase.io import read
from appa.lammps import AtomisticSimulation
from ase.constraints import FixAtoms


@click.command("lammps")
@click.argument(
    "initial",
    type=str,
)
@click.option(
    "--architecture",
    type=str,
    required=True,
    help="appa-supported architecture (mace-mliap, grace, mtt, nequip...)",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Path to model",
)
@click.option(
    "--steps",
    type=int,
    default=1000,
    show_default=True,
    help="Number of steps to run",
)
@click.option(
    "--temperature",
    type=float,
    default=300,
    show_default=True,
    help="MD temperature (K)",
)
@click.option(
    "--timestep",
    type=float,
    default=0.0005,
    show_default=True,
    help="MD timestep (ps)",
)
@click.option(
    "--dump-freq",
    type=int,
    default=20,
    show_default=True,
    help="How many steps between saving frames to the dump file",
)
@click.option(
    "--plumed-file",
    type=str,
    default=None,
    show_default=True,
    help="Path to PLUMED input file",
)
@click.option(
    "--charge",
    type=float,
    default=None,
    help=(
        "Total charge in electrons (negative = excess electrons) for a "
        "charge-conditioned GRACE model. Also logs the work function dE/dq."
    ),
)
@click.option(
    "--padding",
    type=float,
    default=None,
    help=(
        "GRACE fake-atom padding fraction (LAMMPS default: 0.01). Use 0 for an "
        "exact work function in a single point or rerun; keep the default for MD."
    ),
)
@click.option(
    "--boundary",
    type=str,
    default=None,
    help=(
        'LAMMPS boundary, e.g. "p p f". Default: from the structure\'s pbc '
        "(p where periodic, f where not). --wall-distance needs a non-periodic z."
    ),
)
@click.option(
    "--thermostat",
    type=click.Choice(["nose-hoover", "csvr"]),
    default="nose-hoover",
    show_default=True,
    help="fix nvt, or Bussi's stochastic velocity rescaling (fix temp/csvr + nve).",
)
@click.option(
    "--damping",
    type=float,
    default=None,
    help="Thermostat relaxation time (ps). Default: 100 x timestep.",
)
@click.option(
    "--wall-distance",
    type=float,
    default=None,
    help=(
        "Add a one-sided harmonic wall this far (Å) above the top electrode "
        "atom, acting on --wall-species. Default: no wall."
    ),
)
@click.option(
    "--wall-species",
    type=str,
    default="O",
    show_default=True,
    help="Element the wall acts on.",
)
@click.option(
    "--wall-k",
    type=float,
    default=1.0,
    show_default=True,
    help="Wall spring constant (eV/Å^2), as for `appa equilibrate`.",
)
@click.option(
    "--surface-species",
    type=str,
    default=None,
    help="Electrode element the wall is measured from. Default: the frozen atoms' element.",
)
def lammps(
    model,
    architecture,
    initial,
    steps,
    temperature,
    timestep,
    dump_freq,
    plumed_file,
    charge,
    padding,
    boundary,
    thermostat,
    damping,
    wall_distance,
    wall_species,
    wall_k,
    surface_species,
):
    """Write LAMMPS simulation inputs."""
    atoms = read(initial)
    click.echo(f"Loaded initial configuration from: {initial}")

    fixed_indices = []
    if atoms.constraints:
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_indices = constraint.index.tolist()
                break
    click.echo(f"Fixed atom indices: {fixed_indices}")

    try:
        sim = AtomisticSimulation(atoms, boundary=boundary)
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    click.echo(f"Boundary: {' '.join(sim.boundary)}")
    sim.set_potential(
        model,
        architecture=architecture,
        total_charge=charge,
        padding=padding,
    )
    if charge is not None:
        click.echo(f"Total charge: {charge} e; logging the work function dE/dq")

    md_kwargs = {} if damping is None else {"damping": damping}
    sim.set_molecular_dynamics(
        temperature=temperature,
        timestep=timestep,
        fixed_atoms=fixed_indices,
        thermostat=thermostat,
        **md_kwargs,
    )
    click.echo(f"Thermostat: {thermostat}")
    if wall_distance is not None:
        try:
            z0 = sim.set_harmonic_wall(
                wall_distance,
                species=wall_species,
                surface_species=surface_species,
                k=wall_k,
            )
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        click.echo(
            f"Harmonic wall on {wall_species} at z = {z0:.3f} Å "
            f"({wall_distance} Å above the electrode), k = {wall_k} eV/Å^2"
        )
    if plumed_file is not None:
        sim.set_plumed(plumed_file)
    sim.set_log()
    sim.set_dump(dump_freq=dump_freq)
    sim.set_run(n_steps=steps)

    sim.write_inputs()
    click.echo("Inputs written to current working directory")
