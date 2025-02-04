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
POTL_STAGENAME = "Setting up interatomic potential"
MDYN_STAGENAME = "Molecular dynamics setup"
LOGS_STAGENAME = "Optional logging settings"
RUNN_STAGENAME = "Running"
SNELLIUS_LAMMPS_MODULES = [
    "2023",
    "CUDA/12.1.1",
    "imkl/2023.1.0",
    "OpenMPI/4.1.5-NVHPC-24.5-CUDA-12.1.1",
]


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
                f"read_data {SYSTEM_DATA_FILENAME}",
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


class BatchJob:
    """
    A class to manage batch jobs for atomistic simulations.

    Parameters
    ----------
    working_directory : os.PathLike
        The directory where the simulation tasks will be executed.
    simulations : list of AtomisticSimulation
        A list of atomistic simulations to be submitted by the batch job.
    """

    def __init__(
        self, working_directory: os.PathLike, simulations: list[AtomisticSimulation]
    ):
        self.working_directory = working_directory
        self.simulations = dict(enumerate(simulations))

    def write_inputs(self):
        """
        Write input files for each simulation in the batch job.
        """
        for identifier, sim in self.simulations.items():
            results_dir = os.path.join(self.working_directory, f"task_{identifier:06d}")
            sim.write_inputs(results_dir)

    def write_jobfile(  # pylint: disable=dangerous-default-value
        self,
        modules: Optional[list] = SNELLIUS_LAMMPS_MODULES,
        lmp_executable="~/lammps/build-a100/lmp -k on g 1 -sf kk",
        **kwargs,
    ):
        """
        Write a SLURM job file for running the simulations.

        This function generates a SLURM job script that can be used to submit a batch job
        for running multiple atomistic simulations. The job script includes necessary
        SBATCH directives, module loading commands, and the execution command for LAMMPS.

        Parameters
        ----------
        modules : list, optional
            List of modules to load in the job script. Default are modules used to run LAMMPS
            with GPU on Snellius.
        lmp_executable : str, optional
            Command to run LAMMPS. Default is "~/lammps/build-a100/lmp -k on g 1 -sf kk".
            The code then adds 'srun' before this command and '-in input.lmp', the default input
            file name, after the command.

        Other Parameters
        ----------------
        Additional SBATCH arguments. Can be disabled by setting to None. Underscores in
        the parameter names are replaced by dashes if necessary.
        job_name: str
            Name of the job. Default: "batch_job"
        output: str
            Name of SLURM output file. Default: "logs/job_%A_%a.out"
            (uses %A for job ID and %a for array job ID).
        partition: str
            Name of partition. Default: "gpu"
        nodes: int
            Number of nodes. Default: 1
        ntasks: int
            Number of parallel tasks. Default: 1
        cpus_per_task: int
            Number of CPUs per task. Default: 18
        gpus: int
            Number of GPUs used. Default: 1
        time: str
            String specifying maximum runtime in (d-)hh:mm:ss. Default: "01:00:00"

        Examples
        --------
        >>> batch_job.write_jobfile(job_name="my_job", time="1-00:00:00", partition="gpu")
        """
        sbatch_args = {
            "job_name": "batch_job",
            "output": "logs/job_%A_%a.out",
            "partition": "gpu",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 18,
            "gpus": 1,
            "time": "01:00:00",
        }
        sbatch_args.update(kwargs)

        jobfile_content = "#!/bin/bash\n"
        for key, value in sbatch_args.items():
            if value is not None:
                sbatch_key = key.replace("_", "-")
                jobfile_content += f"#SBATCH --{sbatch_key}={value}\n"
        jobfile_content += "\n"

        if modules:
            for module in modules:
                jobfile_content += f"module load {module}\n"

        jobfile_content += f"""
FOLDER_LIST=({self.working_directory}/task_$(printf '%06d' $SLURM_ARRAY_TASK_ID))
cd $FOLDER_LIST
srun {lmp_executable} -in input.lmp
"""

        with open("jobfile.sh", "w", encoding="utf-8") as jobfile:
            jobfile.write(jobfile_content)
