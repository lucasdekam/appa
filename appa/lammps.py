"""
LAMMPS I/O for atomistic machine learning simulations
"""

from typing import Literal, Optional
import os
import numpy as np
from ase import Atoms, io
from pymatgen.io.lammps.inputs import LammpsInputFile

SYSTEM_DATA_FILENAME = "system.data"
INIT_STAGENAME = "Initialization"
READ_STAGENAME = "Define simulation box"
POTL_STAGENAME = "Define interatomic potential"
MDYN_STAGENAME = "Molecular dynamics setup"
LOGS_STAGENAME = "Optional logging settings"
RUNN_STAGENAME = "Running"


class AtomisticSimulation(LammpsInputFile):
    """
    Setting up an atomistic simulation

    Examples
    --------
    >>> from ase.io import read
    >>> atoms = read('myatoms.xyz')
    >>> sim = AtomisticSimulation(atoms)
    >>> sim.set_potential(model_file="my_potential.lammps.pt")
    >>> sim.set_molecular_dynamics(temperature=300, timestep=0.001)
    >>> sim.set_run(n_steps=10000)
    >>> sim.write_file(filename="input.lmp")
    """

    def __init__(self, atoms: Atoms):
        """
        Initialize the atomistic simulation with given atoms.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object containing the atomic configuration.
        """
        self.atoms = atoms
        self.species = np.unique(self.atoms.get_chemical_symbols()).tolist()

        super().__init__(stages=None)
        self.add_stage(
            stage_name=INIT_STAGENAME,
            commands=[
                "units metal",
                "boundary p p p",
                "atom_style atomic",
            ],
        )
        self.add_stage(
            stage_name=READ_STAGENAME,
            commands=[
                f"read_data {SYSTEM_DATA_FILENAME}",
            ],
        )

    def set_potential(
        self,
        model_file: os.PathLike,
        architecture: Literal["mlmace", "mace-mliap", "grace"] = "mace-mliap",
    ):
        """
        Define commands for the interatomic potential (force field).

        Parameters
        ----------
        model_file : os.PathLike
            Path to the potential model file.
        architecture : {'mlmace', 'mace-mliap', 'grace'}
            Type of architecture for the potential. Default is 'mace-mliap'.

        Examples
        --------
        >>> sim.set_potential(model_file="my_potential.lammps.pt")
        """
        formatted_symbols = " ".join(self.species)

        if architecture == "mlmace":
            self.add_commands(
                stage_name=INIT_STAGENAME,
                commands=["atom_modify map yes", "newton on"],
            )
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    "pair_style mace no_domain_decomposition",
                    f"pair_coeff * * {model_file} {formatted_symbols}",
                ],
            )
        elif architecture == "mace-mliap":
            self.add_commands(
                stage_name=INIT_STAGENAME,
                commands=["atom_modify map yes", "newton on"],
            )
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    f"pair_style mliap unified {model_file} 0",
                    f"pair_coeff * * {formatted_symbols}",
                ],
            )
        elif architecture == "grace":
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    f"pair_style grace pad_verbose",
                    f"pair_coeff * * {model_file} {formatted_symbols}",
                ],
            )
        else:
            raise NotImplementedError(f"Architecture {architecture} is not recognized.")

    def set_molecular_dynamics(
        self,
        temperature: int = 300,
        timestep: float = 0.0005,
        fixed_atoms: Optional[list[int]] = None,
        **kwargs,
    ):
        """
        Set up the molecular dynamics simulation.

        Parameters
        ----------
        temperature : int, optional
            Temperature in Kelvin. Default is 300.
        timestep : float, optional
            Timestep for integration in picoseconds. Default is 0.0005.

        Other Parameters
        ----------------
        damping : float, optional
            Damping factor for NVT thermostat. Default is ``100 * timestep``.
        skin : float, optional
            Skin distance for neighbor list construction. Default is 2.0.
        seed : int, optional
            Seed for velocity initialization. Default is 1.
        neigh_modify : int, optional
            After how many steps to re-calculate the neighbor list. Default: 10

        Examples
        --------
        >>> sim.set_molecular_dynamics(temperature=500, timestep=0.001, seed=42)
        """
        damping = kwargs.get("damping", 100 * timestep)
        skin = kwargs.get("skin", 2.0)
        seed = kwargs.get("seed", 1)
        neigh_modify = kwargs.get("neigh_modify", 10)

        commands = [
            f"neighbor {skin:.1f} bin",
            f"neigh_modify every {neigh_modify}",
            f"timestep {timestep}",
        ]

        if fixed_atoms is not None:
            fixed_atoms_one_based = [i + 1 for i in fixed_atoms]
            fixed_group_cmd = "group fixed_group id " + " ".join(
                map(str, fixed_atoms_one_based)
            )
            commands += [
                fixed_group_cmd,
                "fix freeze_fix fixed_group setforce 0.0 0.0 0.0",
                "velocity fixed_group set 0.0 0.0 0.0",
                "group mobile subtract all fixed_group",
                f"velocity mobile create {temperature} {seed} mom yes rot no",
                f"fix nvt_fix mobile nvt temp {temperature} {temperature} {damping}",
            ]
        else:
            commands += [
                f"velocity all create {temperature} {seed} mom yes rot no",
                f"fix nvt_fix all nvt temp {temperature} {temperature} {damping}",
            ]

        self.add_stage(
            stage_name=MDYN_STAGENAME,
            commands=commands,
        )

    def set_output(
        self,
        log_freq: int = 20,
        dump_freq: int = 20,
        dump_name: str = "lammps.dump",
        log_forces: bool = False,
    ):
        """
        Parameters
        ----------
        log_freq : int, optional
            How often to print thermo information to the log file. Default: 20
        dump_freq : int = 20
            Frequency of dump file output
        dump_name : str = "lammps.dump"
            Name of the dump file
        """
        formatted_symbols = " ".join(self.species)
        commands = [
            f"thermo {log_freq}",
            "thermo_style custom step pe ke etotal temp",
            "thermo_modify format float %15.5f",
        ]
        if log_forces:
            dump_spec = f"dump dump_1 all custom {dump_freq} {dump_name} id type element xu yu zu vx vy vz fx fy fz"
        else:
            dump_spec = f"dump dump_1 all custom {dump_freq} {dump_name} id type element xu yu zu vx vy vz"
        commands.append(dump_spec)
        commands.append(f"dump_modify dump_1 element {formatted_symbols} sort id")
        self.add_stage(
            stage_name=LOGS_STAGENAME,
            commands=commands,
        )

    def set_run(
        self,
        n_steps: int,
        restart_freq: Optional[int] = None,
    ):
        """
        Define run command and saving restart files.

        Parameters
        ----------
        n_steps : int
            Number of steps to run the simulation.
        restart_freq : int, optional
            Frequency of saving restart files. Default is None, in which case no
            restart files are saved.

        Examples
        --------
        >>> sim.set_run(n_steps=1000, restart_freq=500)
        """
        commands = []
        if restart_freq is not None:
            commands.append(f"restart {restart_freq} restart.*")
        commands.append(f"run {n_steps}")

        self.add_stage(
            stage_name=RUNN_STAGENAME,
            commands=commands,
        )

    def write_inputs(
        self,
        working_directory: os.PathLike = ".",
        input_filename: str = "input.lmp",
    ):
        """
        Write the LAMMPS input file and system.data file.

        Parameters
        ----------
        working_directory : os.PathLike
            Directory to write the input files.
        input_filename : str, optional
            Name of the input file. Default is "input.lmp".

        Examples
        --------
        >>> sim.write_inputs(working_directory="results")
        """
        os.makedirs(working_directory, exist_ok=True)

        io.write(
            os.path.join(working_directory, SYSTEM_DATA_FILENAME),
            self.atoms,
            format="lammps-data",
            specorder=self.species,
            masses=True,
        )

        input_path = os.path.join(working_directory, input_filename)
        self.write_file(input_path)


def write_array_job_inputs(
    directory: os.PathLike,
    simulations: list[AtomisticSimulation],
    folder_name: str = "task",
):
    for identifier, sim in enumerate(simulations):
        results_dir = os.path.join(directory, f"{folder_name}_{identifier:03d}")
        sim.write_inputs(results_dir)
