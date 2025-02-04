"""
LAMMPS I/O for atomistic machine learning simulations
"""

from typing import Literal, Optional
import warnings
import os
import numpy as np
from ase import Atoms, io
from pymatgen.io.lammps.inputs import LammpsInputFile

INIT_STAGENAME = "Initialization"
READ_STAGENAME = "Reading atomic data"
POTL_STAGENAME = "Setting up interatomic potential"
MDYN_STAGENAME = "Molecular dynamics setup"
LOGS_STAGENAME = "Optional logging settings"
RUNN_STAGENAME = "Running"


class AtomisticSimulation(LammpsInputFile):
    """
    Setting up an atomistic simulation

    Examples
    --------
    >>> from ase.io import read
    >>> sim = AtomisticSimulation(working_directory="results")
    >>> atoms = read('myatoms.xyz')
    >>> sim.set_atoms(atoms)
    >>> sim.set_potential(model_file="my_potential.lammps.pt")
    >>> sim.set_molecular_dynamics(temperature=300, timestep=0.001)
    >>> sim.set_run(n_steps=1000)
    >>> sim.write_file(filename="input.lmp")
    """

    def __init__(self, working_directory: os.PathLike):
        """
        Parameters
        ----------
        working_directory : os.PathLike
            Directory where simulation files will be stored.
        """
        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)

        self.atoms = None
        self.species = None
        self.data_file = None

        super().__init__(stages=None)
        self.add_stage(
            stage_name=INIT_STAGENAME,
            commands=[
                "units metal",
                "boundary p p p",
                "atom_style atomic",
            ],
        )

    def set_atoms(self, atoms: Atoms):
        """
        Define the initial structure for the simulation

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object representing the atomic structure.

        Examples
        --------
        >>> from ase.io import read
        >>> atoms = read('myatoms.xyz')
        >>> sim.set_atoms(atoms)
        """
        data_file_name = "system.data"
        self.atoms = atoms
        self.species = np.unique(self.atoms.get_chemical_symbols()).tolist()
        self.data_file = os.path.join(self.working_directory, data_file_name)
        self.add_stage(
            stage_name=READ_STAGENAME,
            commands=[f"read_data {self.data_file}"],
        )
        io.write(
            self.data_file,
            self.atoms,
            format="lammps-data",
            specorder=self.species,
            masses=True,
        )

    def set_potential(
        self, model_file: os.PathLike, architecture: Literal["mace"] = "mace"
    ):
        """
        Define commands for the interatomic potential (force field)

        Parameters
        ----------
        model_file : os.PathLike
            Path to the coefficients file for the potential.
        architecture : {'mace'}, optional
            Type of architecture for the potential. Default is 'mace'.

        Examples
        --------
        >>> sim.set_potential(model_file="my_potential.lammps.pt")
        """
        if architecture == "mace":
            formatted_symbols = " ".join(self.species)
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    "pair_style mace no_domain_decomposition",
                    f"pair_coeff * * {model_file} {formatted_symbols}",
                ],
                after_stage=READ_STAGENAME,
            )
        else:
            raise NotImplementedError(
                f"Architecture {architecture} has not been implemented: only 'mace' is supported."
            )

    def set_molecular_dynamics(
        self,
        temperature: int = 300,
        timestep: float = 0.0005,
        **kwargs,
    ):
        """
        This method sets up the molecular dynamics simulation by defining the
        initial velocities, neighbor list construction, timestep, and thermostat
        settings. It also configures the output of simulation data to a dump file.

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
        dump_freq : int, optional
            Frequency of dump file output. Default is 10.
        dump_name : str, optional
            Name of the dump file. Default is "lammps.dump".

        Examples
        --------
        >>> sim.set_molecular_dynamics(temperature=500, timestep=0.001, seed=42)
        """
        damping = kwargs.get("damping", 100 * timestep)
        skin = kwargs.get("skin", 2.0)
        seed = kwargs.get("seed", 1)
        dump_freq = kwargs.get("dump_freq", 10)
        dump_name = kwargs.get("dump_name", "lammps.dump")

        formatted_symbols = " ".join(self.species)

        self.add_stage(
            stage_name=MDYN_STAGENAME,
            commands=[
                f"velocity all create {temperature} {seed}",
                f"velocity all scale {temperature}",
                f"neighbor {skin:.1f} bin",
                "neigh_modify every 1",
                f"timestep {timestep}",
                f"fix thermo_fix all nvt temp {temperature} {temperature} {damping}",
                (
                    f"dump dump_1 all custom {dump_freq} {dump_name} "
                    "id element xu yu zu fx fy fz vx vy vz"
                ),
                f"dump_modify dump_1 element {formatted_symbols} sort id",
            ],
            after_stage=POTL_STAGENAME,
        )

    def set_run(
        self,
        n_steps: int,
        restart_freq: Optional[int] = None,
    ):
        """
        Define run command and saving restart files

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
            after_stage=MDYN_STAGENAME,
        )

    def write_file(
        self,
        filename: str = "input.lmp",
        ignore_comments: bool = False,
        keep_stages: bool = True,
    ):
        """
        Write the LAMMPS input file.

        Parameters
        ----------
        filename : str, optional
            The filename to output to, including path. Default is "input.lmp".
        ignore_comments : bool, optional
            True if only the commands should be kept from the InputFile. Default is False.
        keep_stages : bool, optional
            True if the block structure from the InputFile should be kept according to the stages.
            Default is True.
        """
        valid_stages = [
            INIT_STAGENAME,
            READ_STAGENAME,
            POTL_STAGENAME,
            MDYN_STAGENAME,
            RUNN_STAGENAME,
        ]

        if not all(stage in self.stages_names for stage in valid_stages):
            warnings.warn(
                category=Warning,
                message="WARNING: Not all parts of the simulation may be present. "
                "You need to run set_atoms, set_potential, set_molecular_dynamics and set_run, "
                "with set_run last.",
            )
        super().write_file(filename, ignore_comments, keep_stages)
