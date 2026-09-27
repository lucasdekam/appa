"""
LAMMPS I/O for atomistic machine learning simulations
"""

from typing import Literal, Optional
import os
import numpy as np
from ase import Atoms, io
from ase.constraints import FixAtoms
from pymatgen.io.lammps.inputs import LammpsInputFile

SYSTEM_DATA_FILENAME = "system.data"
INIT_STAGENAME = "Initialization"
READ_STAGENAME = "Define simulation box"
POTL_STAGENAME = "Define interatomic potential"
MDYN_STAGENAME = "Molecular dynamics setup"
WALL_STAGENAME = "Harmonic wall"
PLUMED_STAGENAME = "Plumed input and output"
LOG_STAGENAME = "Logging settings"
DUMP_STAGENAME = "Dump output settings"
RUN_STAGENAME = "Running"
RERUN_STAGENAME = "Rerun"

ALLOWED_ARCHS = ["mace-mliap", "grace", "nequip", "allegro", "mtt"]

# id of the ``compute pair grace`` that exposes dE/dq of a charge-conditioned model
WORK_FUNCTION_COMPUTE = "workfunc"


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

    def __init__(self, atoms: Atoms, boundary: Optional[str] = None):
        """
        Initialize the atomistic simulation with given atoms.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object containing the atomic configuration.
        boundary : str, optional
            LAMMPS ``boundary`` string, e.g. ``"p p f"``. Default: read from
            ``atoms.pbc``, ``p`` where periodic and ``f`` (fixed, non-periodic)
            where not. A slab from an extxyz with ``pbc="T T F"`` therefore
            runs with ``p p f``, matching how slab DFT and its training data
            treat the normal. With ``f`` an atom that leaves the box is lost
            and LAMMPS stops, rather than wrapping onto the far side of the
            slab.
        """
        self.atoms = atoms
        self.species = np.unique(self.atoms.get_chemical_symbols()).tolist()
        self.numbers = np.unique(self.atoms.get_atomic_numbers()).tolist()
        self.has_work_function = False

        if boundary is None:
            boundary = " ".join("p" if p else "f" for p in atoms.pbc)
        self.boundary = boundary.split()
        if len(self.boundary) != 3:
            raise ValueError(f"boundary needs three flags, got '{boundary}'")
        for flag, length in zip(self.boundary, atoms.cell.lengths()):
            # LAMMPS reads the box from the cell in every direction, periodic
            # or not, so a zero-length cell vector would give a zero-size box
            if length <= 0:
                raise ValueError(
                    f"cell has a zero-length vector (boundary '{flag}'); "
                    "LAMMPS needs a box in every direction."
                )

        super().__init__(stages=None)
        self.add_stage(
            stage_name=INIT_STAGENAME,
            commands=[
                "units metal",
                f"boundary {' '.join(self.boundary)}",
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
        architecture: str = "mace-mliap",
        total_charge: Optional[float] = None,
        padding: Optional[float] = None,
    ):
        """
        Define commands for the interatomic potential (force field).

        Parameters
        ----------
        model_file : os.PathLike
            Path to the potential model file.
        architecture : {'mace-mliap', 'grace', 'mtt', ...}
            Type of architecture for the potential. Default is 'mace-mliap'.
        total_charge : float, optional
            Total charge of the system in electrons, negative for excess
            electrons (the GPAW-SJM convention). Only for ``grace`` with a
            charge-conditioned (FiLM) model; the model then also exports the
            work function dE/dq, which is logged through a ``compute pair``.
            Default is None: no charge keyword, so an ordinary model.
        padding : float, optional
            ``grace`` only: fraction of fake atoms padded onto the system so
            that the TensorFlow graph does not have to be retraced every time
            the neighbor count changes. Default is None, which leaves the
            LAMMPS default (0.01).

            Padded atoms are conditioned on the real charge and contribute to
            dE/dq, and unlike their contribution to the energy that is not a
            constant offset. Use ``padding=0`` for a single point or a rerun,
            where an exact work function matters more than speed; keep the
            default for MD, where retracing every step is prohibitive.

        Examples
        --------
        >>> sim.set_potential(model_file="my_potential.lammps.pt")
        >>> sim.set_potential("final_model", architecture="grace", total_charge=-0.5)
        """
        formatted_symbols = " ".join(self.species)

        if architecture != "grace":
            for name, value in (("total_charge", total_charge), ("padding", padding)):
                if value is not None:
                    raise NotImplementedError(
                        f"{name} is only supported for the 'grace' architecture, "
                        f"not for '{architecture}'."
                    )

        if architecture == "mace-mliap":
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
            keywords = ["pad_verbose"]
            if padding is not None:
                keywords += ["padding", f"{padding}"]
            if total_charge is not None:
                keywords += ["q", f"{total_charge}"]

            commands = [
                "pair_style grace " + " ".join(keywords),
                f"pair_coeff * * {model_file} {formatted_symbols}",
            ]
            if total_charge is not None:
                # dE/dq is the pair style's global extra quantity; `compute
                # pair` turns it into an ordinary thermo value c_<id>[1].
                commands.append(f"compute {WORK_FUNCTION_COMPUTE} all pair grace")
                self.has_work_function = True

            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=commands,
            )
        elif architecture == "mtt":
            formatted_numbers = " ".join(self.numbers)
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    f"pair_style metatomic/kk {model_file}",
                    f"pair_coeff * * {formatted_numbers}",
                ],
            )
        elif architecture in ["nequip", "allegro"]:
            self.add_stage(
                stage_name=POTL_STAGENAME,
                commands=[
                    f"pair_style {architecture}",
                    f"pair_coeff * * {model_file} {formatted_symbols}",
                ],
            )
        else:
            raise NotImplementedError(
                f"Architecture {architecture} is not recognized. "
                f"Allowed architectures: {', '.join(ALLOWED_ARCHS)}"
            )

    def set_rerun(
        self,
        input_dump: os.PathLike,
    ):
        """
        Set up a rerun of the dump file `input_dump`.
        """
        commands = [f"rerun {input_dump} dump x y z"]
        self.add_stage(
            stage_name=RERUN_STAGENAME,
            commands=commands,
        )

    def set_molecular_dynamics(
        self,
        temperature: int = 300,
        timestep: float = 0.0005,
        fixed_atoms: Optional[list[int]] = None,
        thermostat: Literal["nose-hoover", "csvr"] = "nose-hoover",
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
        fixed_atoms : list of int, optional
            Zero-based indices of atoms to freeze. The thermostat then acts on
            the remaining (``mobile``) atoms only.
        thermostat : {'nose-hoover', 'csvr'}, optional
            ``'nose-hoover'`` (default) writes ``fix nvt``. ``'csvr'`` writes
            Bussi's stochastic velocity rescaling, ``fix temp/csvr``, together
            with the ``fix nve`` it needs for the time integration. CSVR
            samples the canonical ensemble without the non-ergodicity a single
            Nose-Hoover thermostat can show for small systems. It needs
            LAMMPS's EXTRA-FIX package.

        Other Parameters
        ----------------
        damping : float, optional
            Thermostat relaxation time in picoseconds (``Tdamp``), for either
            thermostat. Default is ``100 * timestep``.
        skin : float, optional
            Skin distance for neighbor list construction. Default is 2.0.
        seed : int, optional
            Seed for velocity initialization. Default is 1.
        neigh_modify : int, optional
            After how many steps to re-calculate the neighbor list. Default: 10

        Examples
        --------
        >>> sim.set_molecular_dynamics(temperature=500, timestep=0.001, seed=42)
        >>> sim.set_molecular_dynamics(temperature=330, thermostat="csvr", damping=0.1)
        """
        if thermostat not in ("nose-hoover", "csvr"):
            raise ValueError(
                f"thermostat must be 'nose-hoover' or 'csvr', got '{thermostat}'"
            )
        damping = kwargs.get("damping", 100 * timestep)
        skin = kwargs.get("skin", 2.0)
        seed = kwargs.get("seed", 1)
        neigh_modify = kwargs.get("neigh_modify", 10)

        commands = [
            f"neighbor {skin:.1f} bin",
            f"neigh_modify every {neigh_modify}",
            f"timestep {timestep}",
        ]

        # an empty list means nothing is fixed: `group fixed_group id` with no
        # ids is an illegal LAMMPS command
        if fixed_atoms:
            fixed_atoms_one_based = [i + 1 for i in fixed_atoms]
            fixed_group_cmd = "group fixed_group id " + " ".join(
                map(str, fixed_atoms_one_based)
            )
            commands += [
                fixed_group_cmd,
                "fix freeze_fix fixed_group setforce 0.0 0.0 0.0",
                "velocity fixed_group set 0.0 0.0 0.0",
                "group mobile subtract all fixed_group",
            ]
            group = "mobile"
        else:
            group = "all"

        commands.append(f"velocity {group} create {temperature} {seed} mom yes rot no")
        if thermostat == "nose-hoover":
            commands.append(
                f"fix nvt_fix {group} nvt temp {temperature} {temperature} {damping}"
            )
        else:
            # temp/csvr only rescales velocities; fix nve does the integration.
            # Both act on the same group, so frozen atoms are neither moved nor
            # counted in the thermostat's temperature.
            commands += [
                f"fix nve_fix {group} nve",
                f"fix csvr_fix {group} temp/csvr {temperature} {temperature} "
                f"{damping} {seed}",
            ]

        self.add_stage(
            stage_name=MDYN_STAGENAME,
            commands=commands,
        )

    def set_harmonic_wall(
        self,
        distance: float,
        species: str = "O",
        surface_species: Optional[str] = None,
        k: float = 1.0,
        cutoff: float = 5.0,
    ):
        """
        One-sided harmonic wall a fixed distance above the electrode surface.

        Every atom of ``species`` above the plane ``z0 = z_surface + distance``
        feels ``F_z = -k (z - z0)`` (energy ``k (z - z0)^2 / 2``); below the
        plane it feels nothing. This is the wall `appa equilibrate` uses, and
        the Hookean plane constraint of the RAZOR MD (Bergmann, Reuter &
        Hoermann, J. Chem. Phys. 164, 174110 (2026), 10 A above Pt). It keeps
        water from evaporating into the vacuum gap, which matters at elevated
        temperature.

        ``z_surface`` is the highest ``surface_species`` atom *of the initial
        structure*, so the plane is fixed in space during the run.

        Written as ``fix wall/harmonic`` on the ``zhi`` face. LAMMPS's
        ``E = eps (r - r_c)^2`` for ``r < r_c`` is repulsive from the wall out
        to ``r_c``, so placing the LAMMPS wall at ``z0 + cutoff`` with
        ``eps = k / 2`` makes it exactly the one-sided spring above: zero below
        ``z0``, ``k (z - z0)^2 / 2`` between ``z0`` and the LAMMPS wall. An atom
        reaching the LAMMPS wall itself (``cutoff`` beyond ``z0``, an energy of
        ``k cutoff^2 / 2``) is a LAMMPS error. The wall energy is not added to
        ``pe`` (no ``fix_modify energy yes``), so the logged energy stays the
        model's own.

        LAMMPS refuses a wall in a periodic dimension, so z must be
        non-periodic: build the simulation from atoms with ``pbc[2] = False``
        (``pbc="T T F"`` in extxyz) or pass ``boundary="p p f"``.

        Parameters
        ----------
        distance : float
            Height of the wall plane above the top ``surface_species`` atom, Å.
        species : str, optional
            Element the wall acts on. Default ``'O'``: holding the oxygens
            holds the molecules, and H bonded to them follows.
        surface_species : str, optional
            Element of the electrode surface. Default: the element of the
            atoms frozen by the structure's ``FixAtoms`` constraint, i.e. the
            slab that `appa build --fix-layers` fixes.
        k : float, optional
            Spring constant in eV/Å², as for `appa equilibrate`. Default 1.0.
        cutoff : float, optional
            How far beyond ``z0`` the spring extends before the hard LAMMPS
            wall, Å. Default 5.0, i.e. 12.5 eV at k = 1: never reached.

        Returns
        -------
        float
            The plane height ``z0`` in Å.

        Examples
        --------
        >>> sim = AtomisticSimulation(atoms)                      # pbc T T F
        >>> sim.set_harmonic_wall(distance=10.0)                  # O, 10 Å above Pt
        >>> sim.set_harmonic_wall(10.0, surface_species="Pt", k=5.0)
        """
        if self.boundary[2].startswith("p"):
            raise ValueError(
                "a wall needs a non-periodic z, but the boundary is "
                f"'{' '.join(self.boundary)}'. Use atoms with pbc[2] = False or "
                "AtomisticSimulation(atoms, boundary='p p f') (CLI: --boundary 'p p f')."
            )
        symbols = np.array(self.atoms.get_chemical_symbols())
        z = self.atoms.get_positions()[:, 2]

        if surface_species is None:
            fixed = [
                i for c in self.atoms.constraints if isinstance(c, FixAtoms)
                for i in c.index
            ]
            if not fixed:
                raise ValueError(
                    "surface_species not given and the structure has no FixAtoms "
                    "constraint to infer the electrode from; pass surface_species."
                )
            fixed_species = np.unique(symbols[fixed])
            if len(fixed_species) != 1:
                raise ValueError(
                    f"frozen atoms contain {fixed_species.tolist()}; pass "
                    "surface_species explicitly."
                )
            surface_species = str(fixed_species[0])

        for name in (species, surface_species):
            if name not in self.species:
                raise ValueError(f"'{name}' is not in this structure ({self.species}).")

        z0 = float(z[symbols == surface_species].max() + distance)
        z_wall = z0 + cutoff
        if z[symbols == species].max() >= z_wall:
            raise ValueError(
                f"a {species} atom already sits at z = {z[symbols == species].max():.2f}, "
                f"at or beyond the LAMMPS wall at {z_wall:.2f}; raise distance or cutoff."
            )
        # same index as the LAMMPS type: specorder in write_inputs is self.species
        atom_type = self.species.index(species) + 1

        commands = [
            f"group wall_group type {atom_type}",
            f"fix wall_fix wall_group wall/harmonic zhi {z_wall:.4f} {k / 2} 1.0 {cutoff}",
        ]
        # before the run stage whatever the call order, so the wall is in
        # force from step 0
        names = self.stages_names
        after = next(
            (s for s in (MDYN_STAGENAME, POTL_STAGENAME, READ_STAGENAME) if s in names),
            None,
        )
        self.add_stage(stage_name=WALL_STAGENAME, commands=commands, after_stage=after)
        return z0

    def set_plumed(
        self,
        plumed_file: str,
        outfile_path: str = "plumed.log",
    ):
        """
        Link to a PLUMED input file

        :param plumed_file: Path to .dat PLUMED input file
        :type plumed_file: str
        :param outfile_path: Path to PLUMED log file, default: plumed.log
        :type outfile_path: str
        """
        self.add_stage(
            stage_name=PLUMED_STAGENAME,
            commands=[
                f"fix pl_fix all plumed plumedfile {plumed_file} outfile {outfile_path}"
            ],
        )

    def set_log(
        self,
        log_freq: int = 20,
        create_energy_log: bool = True,
    ):
        """
        Setup logging
        """
        # see https://docs.lammps.org/fix_print.html
        energy_logfile_commands = [
            "variable time equal step*dt",
            "variable temp equal temp",
            "variable pe equal pe",
            "variable ke equal ke",
            "variable float1 format time %10.4f",
            "variable float2 format temp %.7f",
            "variable float3 format pe %.7f",
            "variable float4 format ke %.7f",
        ]
        columns = ["${float1}", "${float2}", "${float3}", "${float4}"]
        titles = ["time", "temp", "pe", "ke"]

        # logging in the default log.lammps logfile
        thermo_keywords = ["step", "pe", "ke", "etotal", "temp"]

        if self.has_work_function:
            energy_logfile_commands += [
                f"variable wf equal c_{WORK_FUNCTION_COMPUTE}[1]",
                "variable float5 format wf %.7f",
            ]
            columns.append("${float5}")
            titles.append("work_function")
            thermo_keywords.append(f"c_{WORK_FUNCTION_COMPUTE}[1]")

        column_spec = " ".join(columns)
        title_spec = " ".join(titles)
        energy_logfile_commands.append(
            f"fix myinfo all print 1 '{column_spec}' "
            f"title '{title_spec}' file energy.log screen no"
        )

        default_logfile_commands = [
            f"thermo {log_freq}",
            "thermo_style custom " + " ".join(thermo_keywords),
            "thermo_modify format float %15.5f",
        ]

        commands = default_logfile_commands
        if create_energy_log:
            commands += energy_logfile_commands

        self.add_stage(
            stage_name=LOG_STAGENAME,
            commands=commands,
        )

    def set_dump(
        self,
        dump_freq: int = 20,
        dump_name: str = "lammps.dump",
        forces: bool = False,
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
        forces : bool = False
            Whether to include force components in the dump file
        """
        formatted_symbols = " ".join(self.species)
        commands = []
        if forces:
            dump_spec = f"dump dump_1 all custom {dump_freq} {dump_name} id type element xu yu zu vx vy vz fx fy fz"
        else:
            dump_spec = f"dump dump_1 all custom {dump_freq} {dump_name} id type element xu yu zu vx vy vz"
        commands.append(dump_spec)
        commands.append(f"dump_modify dump_1 element {formatted_symbols} sort id")
        self.add_stage(
            stage_name=DUMP_STAGENAME,
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
            stage_name=RUN_STAGENAME,
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
