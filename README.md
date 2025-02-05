# Appa

Some code I use in my research for quickly setting up simulations. Currently includes LAMMPS simulations with MACE. Example:

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
