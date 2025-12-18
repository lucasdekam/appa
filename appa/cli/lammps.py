import click
from ase.io import read
from appa.lammps import AtomisticSimulation
from ase.constraints import FixAtoms


@click.command("lammps")
@click.option(
    "--model",
    type=str,
    required=True,
    help="Path to model",
)
@click.option(
    "--initial",
    type=str,
    required=False,
    help="Path to initial configuration (extxyz)",
)
@click.option(
    "--steps",
    type=int,
    default=1000,
    help="Number of steps to run",
)
@click.option(
    "--temperature",
    type=float,
    default=300,
    help="MD temperature (K)",
)
@click.option(
    "--timestep",
    type=float,
    default=0.0005,
    help="MD timestep (ps)",
)
@click.option(
    "--architecture",
    type=str,
    required=True,
    help="appa-supported architecture (mace-mliap, grace, mtt)",
)
@click.option(
    "--dump-freq",
    type=int,
    default=20,
    help="How many steps between saving frames to the dump file",
)
def lammps(
    model,
    initial,
    steps,
    temperature,
    timestep,
    architecture,
    dump_freq,
):
    """Write LAMMPS simulation inputs."""
    atoms = read(initial)
    fixed_indices = []
    if atoms.constraints:
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_indices = constraint.index.tolist()
                break

    sim = AtomisticSimulation(atoms)
    sim.set_potential(model, architecture=architecture)

    sim.set_molecular_dynamics(
        temperature=temperature,
        timestep=timestep,
        fixed_atoms=fixed_indices,
    )
    sim.set_log()
    sim.set_dump(dump_freq=dump_freq)
    sim.set_run(n_steps=steps)

    sim.write_inputs()
