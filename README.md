![Appa Icon](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/0a1d2e33-9dcf-47be-8ecd-c0af5458c545/drieup-065047ff-715e-467b-a309-31ff11b7a61a.jpg/v1/fill/w_150,h_150,q_75,strp/appa_icon_by_rufftoon_drieup-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MjAwIiwicGF0aCI6IlwvZlwvMGExZDJlMzMtOWRjZi00N2JlLThlY2QtYzBhZjU0NThjNTQ1XC9kcmlldXAtMDY1MDQ3ZmYtNzE1ZS00NjdiLWEzMDktMzFmZjExYjdhNjFhLmpwZyIsIndpZHRoIjoiPD0yMDAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.nmOkivf8pvdmI2b2LL6Qa_qBiid5-RG7JfypiQdTHZ8)

# Appa

Miscellaneous code I use in my research. 

(Appa is a pet flying bison in the series *Avatar: The Last Airbender*. He speeds up the journey of the Avatar by flying him and his friends around.)

Icon by *rufftoon* on [DeviantArt](https://www.deviantart.com/rufftoon/art/Appa-Icon-46208689)

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

(this should really be integrated into ASE at some point).

## Setting up LAMMPS simulations
The `appa.lammps` module sets up LAMMPS simulations with MACE (and could be extended for use with other interatomic potentials). Example:

```python
from appa.lammps import AtomisticSimulation, ArrayJob
from ase.io import read

# Define atomic structure
atoms = read("path/to/myatoms.xyz")

# Setup a simulation
sim1 = AtomisticSimulation(atoms)
sim1.set_potential("path/to/my_potential.lammps.pt")
sim1.set_molecular_dynamics(temperature=330, timestep=0.0005)
sim1.set_run(n_steps=2000000)

# Setup another simulation
sim2 = AtomisticSimulation(atoms)
sim2.set_potential("path/to/another_potential.lammps.pt")
sim2.set_molecular_dynamics(temperature=330, timestep=0.0005)
sim2.set_run(n_steps=2000000)

sims = [sim1, sim2]

# Write input files to submit as a batch job
job = ArrayJob("results", sims)
job.write_inputs()
job.write_jobfile()
```

Then you can submit the two simulations (indices 0 to 1) as an array job:

```bash
sbatch jobfile.sh --array=0-1
```


## Learning curves
The `appa.learning_curves` module contains tools to plot learning curves for machine learning interatomic potentials. In this context a learning curve plot shows the test error of the model against the training set size. For all data points the model should be fully trained. (These learning curves are not to be confused with test/train error vs. #epochs curves obtained from training one model). See also fig. 2e of [this paper](https://arxiv.org/pdf/2404.12367). Example:

```python
from appa.learning_curves import LearningCurve 
import numpy as np

seeds = range(3)  # define seeds with which different models are trained
# (even better would be to subsample different training sets)
subsets = [10, 110, 610, 1610, 2879]  # define size of each training set

# load 'true' forces and energy on test set
dft_forces = np.load('data/test.f.npy')
dft_energy = np.load('data/test.e.npy')

mace_lc = LearningCurve()
for i, size in enumerate(subsets):
    mace_lc.add_training_set(
        n_training_samples=size,
        force_component_errors=[np.load(f"models/test-{i:d}-{j:d}.f.npy") - dft_forces for j in seeds],
        energy_per_atom_errors=[np.load(f"models/test-{i:d}-{j:d}.e.npy") - dft_energy for j in seeds],
    )
```

Then plotting:

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7,3))
ax_lc = fig.add_subplot(121)
ax_vi = fig.add_subplot(122)

style = dict(
    color="#f52f2f",
    label="mace",
    marker="o",
    markersize=7,
    fillstyle="full",
)

mace_lc.make_violin(
    ax_vi, 
    error_type="force_component",
    face_color=style["color"],
    n_subsampling=10000,
)

mace_lc.make_learningcurve(
    ax_lc,
    error_type="force_component",
    **style,
)

fig.tight_layout()
plt.show()
```


