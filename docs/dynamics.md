# Molecular dynamics

To run molecular dynamics with LAMMPS, `appa` provides automatic input generation through the command ```appa lammps```, which takes an initial `.xyz` file (INITIAL) and generates a LAMMPS input file and LAMMPS `.data` initial structure file.

```sh
Usage: appa lammps [OPTIONS] INITIAL

  Write LAMMPS simulation inputs.

Options:
  --architecture TEXT  appa-supported architecture (mace-mliap, grace, mtt,
                       nequip...)  [required]
  --model TEXT         Path to model  [required]
  --steps INTEGER      Number of steps to run  [default: 1000]
  --temperature FLOAT  MD temperature (K)  [default: 300]
  --timestep FLOAT     MD timestep (ps)  [default: 0.0005]
  --dump-freq INTEGER  How many steps between saving frames to the dump file
                       [default: 20]
  --plumed-file TEXT   Path to PLUMED input file
  --charge FLOAT       Total charge in electrons (negative = excess electrons)
                       for a charge-conditioned GRACE model. Also logs the work
                       function dE/dq.
  --padding FLOAT      GRACE fake-atom padding fraction (LAMMPS default: 0.01).
                       Use 0 for an exact work function in a single point or
                       rerun; keep the default for MD.
  --help               Show this message and exit.
```

The PLUMED file is optional.

To run LAMMPS you can use a job like this:

```sh
#!/bin/bash
#SBATCH --job-name=lmp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00

module purge
module load 2025
module load OpenMPI/5.0.7-GCC-14.2.0
module load CUDA/12.8.0
module load CMake/3.31.3-GCCcore-14.2.0
module load OpenBLAS/0.3.29-GCC-14.2.0
module load FFTW.MPI/3.3.10-gompi-2025a

source ~/.bashrc
conda activate grace
which python

WORK_DIR="${PWD}"
OUTPUT_PATH="${PWD}/${SLURM_JOB_ID}"
LOCAL_PATH="/scratch-local/$USER/${SLURM_JOB_ID}"

# --- Prepare scratch ---
mkdir -p $LOCAL_PATH
cd $LOCAL_PATH || exit 1
cp -r $WORK_DIR/* $LOCAL_PATH/

# --- Generate input files for LAMMPS ---
appa lammps initial.xyz --architecture grace --model ~/train/seed/1/final_model --steps 20000
srun /home/ldkam/lammps/build/lmp -in input.lmp

# --- Copy results back & clean ---
mkdir -p "$OUTPUT_PATH"
cp -r $LOCAL_PATH/* $OUTPUT_PATH
rm -rf $LOCAL_PATH
```

For MACE, you need a different command to run LAMMPS, and you need to convert the `.model` file to a MLIAP-LAMMPS interface model. **The model conversion needs to be run on a GPU with CUDA available.** So better include it in the job:

```sh
mace_create_lammps_model path/to/mymace.model --format=mliap
srun /home/ldkam/lammps/build/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lmp
```

If you installed NequIP without kokkos then you should be able to use the same command as for GRACE. Otherwise see the [NequIP LAMMPS interface repo](https://github.com/mir-group/pair_nequip_allegro).

From the MD you get a `lammps.dump` file which you can analyze further. TODO: add a CLI tool to convert to XTC.

## Charged interfaces with GRACE

A charge-conditioned (FiLM) GRACE model takes the total charge of the system as
an input and exports the work function $\partial E/\partial q$ next to the
energy and forces. Running that in LAMMPS instead of through ASE needs the
[charge-conditioned fork](https://github.com/lucasdekam/lammps/tree/grace),
which adds a `q` keyword to `pair_style grace` and publishes dE/dq as the pair
style's global extra quantity.

`appa` drives both through `--charge`:

```sh
appa lammps initial.xyz --architecture grace --model ~/train/seed/1/final_model \
    --charge -0.5 --steps 20000
```

The charge is in electrons, **negative for excess electrons**, matching
GPAW-SJM and the usual training-data convention. This writes

```
pair_style grace pad_verbose q -0.5
pair_coeff * * /path/to/final_model H O Pt
compute workfunc all pair grace
```

and adds `c_workfunc[1]` to `thermo_style` and a `work_function` column to
`energy.log`, so the work function comes out of the same files as the energy —
no separate output to collect.

To sweep the charge, write one directory per charge and run them as an array
job:

```python
from ase.io import read
from appa.lammps import AtomisticSimulation, write_array_job_inputs

atoms = read("initial.xyz")
sims = []
for q in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    sim = AtomisticSimulation(atoms)
    sim.set_potential("final_model", architecture="grace", total_charge=q)
    sim.set_molecular_dynamics(temperature=300, timestep=0.0005)
    sim.set_log()
    sim.set_dump()
    sim.set_run(n_steps=20000)
    sims.append(sim)

write_array_job_inputs("runs", sims, folder_name="q")
```

Two things to keep in mind:

* Passing `--charge` to a model that is *not* charge-conditioned is an error in
  LAMMPS, not a warning — otherwise every number would silently be the $q = 0$
  answer wearing a charge label. Passing it to a non-GRACE architecture is an
  error in `appa`.
* The reported dE/dq is that of the *padded* system. Padded atoms are
  conditioned on the real charge and contribute, and unlike their contribution
  to the energy that is not a constant offset. Use `--padding 0` when the
  absolute work function has to be exact (a single point or a rerun); for MD,
  keep the default, because retracing the TensorFlow graph every time the
  neighbor count changes is prohibitively slow.

## Output file conversion

`appa` contains a handy tool to convert big LAMMPS dump files and/or XYZ files to the compressed XTC format. The XTC file then only contains the atomic positions in 5-decimal precision (and always needs the corresponding topology file `system.data` to be interpreted). For XYZ trajectories, a `system.data` file is generated.

```sh
Usage: appa convert xtc [OPTIONS]

  Convert XYZ or LAMMPS dump trajectories to XTC.

Options:
  --pattern TEXT             Glob pattern for input files (e.g., './*/*.xyz').
                             [required]
  --format [xyz|lammpsdump]  Input format.  [required]
  --nprocs INTEGER           Number of parallel processes.  [default: 1]
  --help                     Show this message and exit.
```

## Looking at a trajectory

To actually look at an XTC trajectory, `appa view` opens it in the ASE GUI:

```sh
appa view runs/task_000/lammps.xtc
```

An XTC file holds only positions and the box, so the species come from the
`system.data` beside it — the one `appa convert xtc` wrote. Point `--topology`
somewhere else if it lives elsewhere.

```sh
Usage: appa view [OPTIONS] TRAJECTORY

  View an XTC trajectory in the ASE GUI.

Options:
  --topology FILE       LAMMPS data file with the species. Default:
                        system.data next to the trajectory.
  --start INTEGER       First frame to load.  [default: 0]
  --stop INTEGER        Stop before this frame. Default: the end of the
                        trajectory.
  --every INTEGER       Load every Nth frame.  [default: 1]
  --max-frames INTEGER  Refuse to load more frames than this, since the GUI
                        becomes unusable and the images are held in memory.
                        Use 0 for no limit.  [default: 2000]
  --help                Show this message and exit.
```

Every frame is held in memory and the GUI slider gets unusable long before you
run out of it, so a production run wants thinning rather than the whole thing:

```sh
appa view lammps.xtc --every 50
```

The `--max-frames` guard is there to stop you loading a 200k-frame trajectory
by accident; it tells you which `--every` would have fit.
