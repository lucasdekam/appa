---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Building 

This page describes the `appa` tools to set up initial configurations for MD simulations of electrochemical interfaces.

## Building a structure
To build a model structure of an electrochemical interface, you can use the command-line tool `appa build`. This has several options:

```shell
Usage: appa build [OPTIONS]

  Build an electrode-electrolyte interface.

Options:
  -o, --output TEXT     Output structure file  [default: interface.xyz]
  --material TEXT       Chemical element of the electrode material  [default:
                        Au]
  --size INTEGER...     Slab dimensions (nx ny nz); ny must be even  [default:
                        4, 4, 4]
  --d-water FLOAT       Water layer thickness  [default: 10]
  --d-vacuum FLOAT      Vacuum layer thickness  [default: 20]
  --a FLOAT             Lattice parameter; if None takes ASE default
  --fix-layers INTEGER  Number of layers to fix at bottom of slab  [default:
                        2]
  --ion TEXT            Chemical element of the ion
  --n-ions INTEGER      Number of ions to add  [default: 0]
  --coverage FLOAT      Hydrogen coverage  [default: 0]
  --ptop FLOAT          Probability for top site occupation; P(fcc) is
                        1-P(top)  [default: 0.5]
  --help                Show this message and exit.
```

For example, 

```
appa build -o interface.xyz --material Pt --size 4 4 4 --d-water 8 --d-vacuum 20 --a 3.94 --coverage 0.0833 --ptop 1
```

```{code-cell} python
from ase.io import read
from ase.visualize import view

atoms = read('data/interface.xyz')
view(atoms, viewer='x3d')
```

## Pre-equilibration
`packmol` packs the water molecules in a pretty random way and this is not structurally optimal at all. To equilibrate, you can use `appa equilibrate`, which runs MD at a specified temperature using a NequIP potential you specify. Note that you need to specify a compiled model. Compiling a foundation model was described in the [](#nequip-install) section. The `equilibrate` script uses a harmonic 'wall' to ensure water molecules staying in the water region. The options are:

```shell
Usage: appa equilibrate [OPTIONS] STRUCTURE MODEL

  Equilibrate a structure using ASE MD with a given compiled NequIP model.
  Uses a harmonic wall to make sure water molecules don't escape into the
  vacuum region.

Options:
  --output TEXT        Output name of equilibrated structure  [default:
                       equilibrated.xyz]
  --temperature FLOAT  MD temperature in K.  [default: 300.0]
  --steps INTEGER
  --z-buffer FLOAT     Wall position from maximum position of input structure
                       [default: 2.0]
  --k-wall FLOAT       Wall spring force constant (eV/Å^2)  [default: 1.0]
  --traj TEXT          Output trajectory name
  --help               Show this message and exit.
```

Equilibrating the structure shown above for 2000 steps at a temperature of 300K:

```{code-cell} python
atoms = read('data/equilibrated.xyz')
atoms.wrap()
view(atoms, viewer='x3d')
```

## Example jobscript

```shell
#!/bin/bash
#SBATCH --job-name=build
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00

module purge
module load 2025
module load CUDA/12.8.0

source ~/.bashrc
conda activate nequip

appa build -o interface.xyz --material Pt --size 4 4 4 --d-water 8 --d-vacuum 20 --a 3.94 --coverage 0.0833 --ptop 1
appa equilibrate interface.xyz ../nequip-oam-l.nequip.pt2 --temperature 300 --steps 2000 --traj equi.traj
```