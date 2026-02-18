# Training set construction

When you have gathered several dozens of structures describing your system of interest, you can use them to build a training set. `appa` contains tools for structure selection and setting up DFT calculations with VASP.

## Structure selection

The command `appa select` uses the [Maximum Set Coverage algorithm](https://doi.org/10.48550/arXiv.2511.10561) from the `quests` [package](https://github.com/dskoda/quests) to select diverse structures from a dataset. To use this tool you need to have `quests` installed (available with `pip`).

```shell
Usage: appa select [OPTIONS]

  Select the most diverse configurations spanning the configuration space
  using the Maximum Set Coverage algorithm.

Options:
  -d, --data-dir PATH  Directory with .extxyz/.xyz/.traj files  [required]
  --size INTEGER       Number of structures to select  [default: 100]
  --bw FLOAT           Bandwidth for entropy estimation kernel  [default: 0.065]
  -o, --out TEXT       Output XYZ file  [default: selected.xyz]
  -s, --species TEXT   Allowed species (e.g. -s O -s H -s Pt)
  --help               Show this message and exit.
```

The default bandwidth seems okay for water-related atomic environments.

Example:

```shell
appa select -d data --size 1200 -s O -s H -s Pt
```

Running can take a few minutes so it's best to run it on a small CPU node on the HPC.

## Data labeling

When you have a set of structures, you can set up a batch DFT calculation with `appa vasp input`. 

```shell
Usage: appa vasp input [OPTIONS]

  Generate VASP input files for a single structure (specified by xyz and
  index). Requires a config .yaml file with sections specified by config_type,
  and an ASE-readable XYZ extended file where all structures have a
  config_type in their info section.

Options:
  --xyz FILE              XYZ file with configurations to be labelled.
                          [required]
  --index INTEGER         Index of the configuration (e.g. SLURM array index).
                          [required]
  --output-dir DIRECTORY  Output directory for VASP input files.  [required]
  --params FILE           YAML file with VASP parameters.  [default: params.yaml]
  --ncore INTEGER         VASP NCORE.  [default: 8]
  --kpar INTEGER          VASP KPAR.  [default: 1]
  --help                  Show this message and exit.
```

An example of a `params.yaml` file is given below. The tags correspond to the ASE `Vasp` calculator input kwargs.

```yaml
interface:
  encut: 450
  kspacing: 0.18
  xc: rpbe
  ivdw: 11
  ediff: 1.0e-6
  ismear: 0
  sigma: 0.1
  lreal: Auto
  nelm: 200
  algo: Fast
  ldipol: True
  idipol: 3
  lwave: False
  lcharg: False
  lasph: True
```

An example jobscript, which can be submitted by `sbatch job --array=0-123` where `123` is the number of structures minus one in your `to_label.xyz` file.

```shell
#!/bin/bash
#SBATCH --job-name=label
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=1-00:00:00

source ~/.bashrc
conda activate grace  # whatever environment has appa installed
which python

module purge
module load 2025
module load VASP5/5.4.4.pl2-foss-2025a-VASPsol-VTST-CUDA-12.8.0

echo "$(date "+%Y-%m-%d %H:%M:%S"): Job started"
echo "$(date "+%Y-%m-%d %H:%M:%S"): Node $SLURMD_NODENAME"

TASK_DIR=$(printf "$PWD/results/%05d" "$SLURM_ARRAY_TASK_ID")
SCRATCH_BASE="/scratch-local/$USER"
SCRATCH_DIR="$SCRATCH_BASE/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

appa vasp input --xyz="to_label.xyz" --index=$SLURM_ARRAY_TASK_ID --output-dir=$TASK_DIR --ncore=32 --kpar=2

mkdir -p $SCRATCH_DIR
rsync -a "$TASK_DIR/" "$SCRATCH_DIR/"
pushd $SCRATCH_DIR
echo "$(date "+%Y-%m-%d %H:%M:%S"): Running VASP"
srun vasp_std
popd
rsync -a "$SCRATCH_DIR/" "$TASK_DIR/"

# Clean up
rm -rf "$SCRATCH_DIR"
echo "$(date "+%Y-%m-%d %H:%M:%S"): Job finished"
```

To collect the outputs from the DFT calculations, you can use `appa vasp collect`, where `DIRECTORY` is the name of the directory where your VASP calculations are stored, according to the format `DIRECTORY/*/vasprun.xml`.

```shell
Usage: appa vasp collect [OPTIONS] DIRECTORY

  Collect VASP outputs, analyze energy/force distributions, filter outliers,
  and write to extxyz.

Options:
  -o, --output FILE  Output extxyz file.  [default: collected.xyz]
  --fmax FLOAT       Maximum allowed force magnitude in eV/Å  [default: 10]
  --emax FLOAT       Maximum deviation from mean energy as a fraction of mean
                     energy  [default: 0.1]
  --help             Show this message and exit.
```

This script also filters outliers, i.e. structures with large forces or large energy deviations, which can destabilize the ML model training.

This approach is an alternative to selecting structures based on committee disagreement; it does not require running slow molecular dynamics with a committee. It might do a few more DFT calculations than necessary (on outliers), though, but usually CPU hours are cheaper and more readily available than GPU hours.
