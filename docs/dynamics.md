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
