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

    def set_potential(
        self, model_file: os.PathLike, architecture: Literal["mace"] = "mace"
    ):
        """
        Define commands for the interatomic potential (force field).

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
                after_stage=INIT_STAGENAME,
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
            after_stage=INIT_STAGENAME,
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
        data_filename: str = "system.data",
        input_filename: str = "input.lmp",
    ):
        """
        Write the LAMMPS input file and system.data file.

        Parameters
        ----------
        working_directory : os.PathLike
            Directory to write the input files.
        data_filename : str, optional
            Name of the data file. Default is "system.data".
        input_filename : str, optional
            Name of the input file. Default is "input.lmp".

        Examples
        --------
        >>> sim.write_inputs(working_directory="results")
        """
        os.makedirs(working_directory, exist_ok=True)
        data_path = os.path.join(working_directory, data_filename)

        self.add_stage(
            stage_name=READ_STAGENAME,
            commands=[f"read_data {data_path}"],
            after_stage=INIT_STAGENAME,
        )

        io.write(
            data_path,
            self.atoms,
            format="lammps-data",
            specorder=self.species,
            masses=True,
        )

        input_path = os.path.join(working_directory, input_filename)
        self.write_file(input_path)


class BatchJob:
    def __init__(self, working_directory: os.PathLike):
        self.working_directory = working_directory

    def write_jobfile(
        self,
        modules: Optional[list] = None,
        run_command="srun ~/lammps/build-a100/lmp -k on g 1 -sf kk -in input.lmp",
        **kwargs,
    ):
        # Parse kwargs for SBATCH arguments
        job_name = kwargs.get("job_name", "batch_job")
        output = kwargs.get("output", "logs/job_%A_%a.out")
        partition = kwargs.get("partition", "gpu")
        nodes = kwargs.get("nodes", 1)
        ntasks = kwargs.get("ntasks", 1)
        cpus_per_task = kwargs.get("cpus_per_task", 18)
        gpus = kwargs.get("gpus", 1)
        time = kwargs.get("time", "3-00:00:00")

        jobfile_content = f"""#!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={output}
        #SBATCH --partition={partition}
        #SBATCH --nodes={nodes}
        #SBATCH --ntasks={ntasks}
        #SBATCH --cpus-per-task={cpus_per_task}
        #SBATCH --gpus={gpus}
        #SBATCH --time={time}
        """

        # Load necessary modules
        if modules:
            for module in modules:
                jobfile_content += f"module load {module}\n"

        # Find folder corresponding to SLURM_ARRAY_TASK_ID
        jobfile_content += f"FOLDER_LIST=({self.working_directory}/task_$(printf '%06d' $SLURM_ARRAY_TASK_ID))\n"
        jobfile_content += "cd $FOLDER\n"
        jobfile_content += f"{run_command}\n"

        with open("jobfile.sh", "w", encoding="utf-8") as jobfile:
            jobfile.write(jobfile_content)
