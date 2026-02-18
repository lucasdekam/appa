# Enhanced sampling

Given a reasonably equilibrated configuration we can try enhanced sampling of the Volmer step. To generate a simple `plumed.dat` file, you can use `appa plumed volmer`. View the structure and note the (zero-based) indices of the hydrogen atom that you want to transfer, the nearest surface atom, and the most accessible oxygen atom. Then use the command as follows:

```sh
Usage: appa plumed volmer [OPTIONS]

  Write plumed.dat input file for a Volmer step calculation.

Options:
  -o, --oxygen_id INTEGER
  -h, --hydrogen_id INTEGER
  -s, --surface_id INTEGER
  --stride INTEGER           How often to print to COLVAR  [default: 10]
  --help                     Show this message and exit.
```

For example:
```sh
appa plumed volmer -o 107 -s 56 -h 64
```

You can then setup a molecular dynamics run with `appa lammps`, such as
```sh
appa lammps initial.xyz --architecture grace --model ~/plumed-test/train/seed/1/final_model --steps 800000 --plumed-file plumed.dat
```

You can also write different PLUMED files, such as umbrella sampling of a reaction coordinate suggested by [Kronberg & Laasonen](https://doi.org/10.1021/acscatal.1c00538), and [Santos et al.](doi.org/10.1016/j.jelechem.2024.118044):

```sh
# Reaction coordinate xi = d(O-H) - d(Pt-H)
xi: COMBINE ARG=d_OH,d_MH COEFFICIENTS=1,-1 PERIODIC=NO

# Harmonic umbrella restraint
restraint: MOVINGRESTRAINT ...
    ARG=xi
    STEP0=0 AT0={xi0:.2f} KAPPA0=0.0
    STEP1={warmup} AT1={cv_target:.2f} KAPPA1={kappa}
...

pos = atoms.positions
d_OH = np.linalg.norm(pos[oxygen_id, :] - pos[hydrogen_id, :])
d_MH = np.linalg.norm(pos[surface_id, :] - pos[hydrogen_id, :])
xi0 = d_OH - d_MH
```

However, in my experience this results in the proton diffusing along the surface, creating a large $d_\mathrm{OH}$ and $d_\mathrm{MH}$, resulting in a $\xi\approx0$ but a structure very different from the expected transition state. Adding more training data around the TS might help.

## Analysis

To get a free-energy surface from metadynamics, you can use 

```sh
module load 2025
module load PLUMED/2.9.4-foss-2025a
plumed sum_hills --hills path/to/my/HILLS
module purge
```

(since I'm not sure how to call the `plumed` that was built upon LAMMPS install).

You can plot it to get a result like this:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/fes.dat', comments='#')
cv1 = data[:,0]
cv2 = data[:,1]
energy = data[:,2]

plt.figure(figsize=(5,4))
plt.tricontourf(cv1, cv2, energy, levels=10, cmap='magma')
plt.colorbar()
plt.xlabel('d_OH')
plt.ylabel('d_MH')
plt.show()
```

Probably limited training data, limited metadynamics sampling time, and limited training time all contributes to this result not being amazing. The proton crossed the energy barrier once in this run. But it still gives an idea of what is possible with MLIPs.
