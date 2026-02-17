<img src="examples/appa.png" alt="appa's lost days" width="190"/>

# Appa

Miscellaneous code I use in my research.

(Appa is a pet flying bison in the series *Avatar: The Last Airbender*. He speeds up the journey of the Avatar by flying him and his friends around.)

## Building interfaces
The `appa.build` module builds simulation cells common in electrocatalysis research. For example, the `Electrode` class builds  fcc(111) electrode surfaces and allows for automated adding of hydrogen on top and fcc sites.

```python
from appa.build import Electrode
electrode = Electrode(material="Pt", size=(3, 2, 4), a=3.94, fix_layers=2)
electrode.add_hydrogens(coverage=0.66, topsite_probability=0.3)
```

The `ase.Atoms` object can be accessed as `electrode.atoms` for futher manipulation.

One can build an `Interface` with an electrode, water and ions as

```python
from appa.build import Interface
interface = Interface(electrode=electrode.atoms, d_water=20, d_vacuum=15, ions={"Na": 4.5}, ion_delta_z=2.5)
interface.write("interface.xyz")
```

The `Interface` class makes use of [mdapackmol](https://github.com/MDAnalysis/MDAPackmol) to pack the ions (here, between 4.5-2.5 A and 4.5+2.5 A from the surface) and water molecules (between the highest surface z-coordinate and d_water from the surface). To use this class you need to install [Packmol](https://m3g.github.io/packmol/) and MDAPackmol (which is most convenient on Linux).

## Writing and loading Atoms with constraints

The `Electrode.atoms` object uses `FixAtoms` constraints. To save and load these `Atoms` objects including their constraints, use

```python
from appa.utils import write_with_fixatoms
write_with_fixatoms("fix.xyz", atoms)
```

and

```python
from appa.utils import read_with_fixatoms
atoms = read_with_fixatoms("fix.xyz")
```

Currently, only writing one structure is supported (todo: make it work for a list of `Atoms` objects).

## Setting up LAMMPS simulations
The `appa.lammps` module sets up LAMMPS simulations with MACE or GRACE (and could be extended for use with other interatomic potentials). Example:

```python
from appa.lammps import AtomisticSimulation, ArrayJob
from ase.io import read

# Define atomic structure
atoms = read("path/to/myatoms.xyz")

# Setup a simulation
sim1 = AtomisticSimulation(atoms)
sim1.set_potential("path/to/my_potential.lammps.pt")
sim1.set_molecular_dynamics(temperature=330, timestep=0.0005)
sim1.set_output()
sim1.set_run(n_steps=2000000)

# Setup another simulation
sim2 = AtomisticSimulation(atoms)
sim2.set_potential("path/to/another_potential.lammps.pt")
sim2.set_molecular_dynamics(temperature=330, timestep=0.0005)
sim2.set_output(forces=True)
sim2.set_run(n_steps=2000000)

sims = [sim1, sim2]

# Write input files to submit as a batch job
write_array_job_inputs(directory="results", simulations=sims, folder_name="mytask")
```

This will write LAMMPS input and LAMMPS `.data` to `results/mytask_000` and `results/mytask_001`. 

Then you can submit the two simulations (indices 0 to 1) using an appropriate array job as in `examples/lammps-array-job`:

```bash
sbatch jobfile --array=0-1
```

## Writing CP2K input files
The `appa.cp2k` module contains tools to write CP2K input files. The DFT settings are usually fixed for a given project, and are written in a `params.yaml` file. The DFT section in this file has a similar structure to the dictionaries that can be read and written by [cp2k-input-tools](https://github.com/cp2k/cp2k-input-tools). The `params.yaml` file also specifies what basis sets and pseudopotentials are used for the different elements that might appear in configurations. The function `read_params` reads the YAML parameter file, and `write_input` writes a `coord.xyz` and `input.inp` file to the specified directory.

```python
from appa.cp2k import read_params, write_input
from ase.io import read

params = read_params("examples/cp2k_params.yaml")
params['dft']['+mgrid']['cutoff'] = 500
atoms = read('coords.xyz')
write_input("results", atoms, params)
```

## Todo

Writing CLI utilities, such as 

* Building structures 
* Setting up a (modular) active learning MD with ASE 
* Converting `lammps.dump` into a binary format like XTC
* Performing analysis of trained models, like parity plots
