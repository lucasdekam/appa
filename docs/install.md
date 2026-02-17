# Installation

To install `appa`, 

```shell
git clone github.com/lucasdekam/appa.git
cd appa
pip install .
```

You'll also need `mdapackmol` and `packmol`, both of which should be available through `pip`; otherwise you can [install packmol from source](https://m3g.github.io/packmol/download.shtml) and add the `packmol` program to PATH in your `.bashrc`. 

Currently the script for pre-equilibrating newly created structures uses a NequIP foundation model, whereas the scripts for setting up LAMMPS simulations are intended for use with MACE and GRACE. It's most convenient to install these packages in different environments. If you don't you might run into dependency problems.

(nequip-install)=
## NequIP

Find the installation instructions [here](https://nequip.readthedocs.io/en/latest/guide/getting-started/install.html). In short

```shell
pip install nequip
pip install torch-dftd
```

`torch-dftd` is used in the equilibration script to add a D3 correction on top of NequIP's foundation model. To obtain a foundation model, you can run the following job script or similar: 

```shell
#!/bin/bash
#SBATCH --job-name=compile
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
conda activate nequip  # or your own way to load an environment

nequip-compile nequip.net:mir-group/NequIP-OAM-L:0.1 --mode aotinductor --device cuda --target ase
```

If you rerun this script many times, the NequIP server might deny your request and you should [download the model manually](https://www.nequip.net/models/mir-group/NequIP-OAM-L:0.1) and then supply the local path of the model to the compile command. 

It's possible to run NequIP with LAMMPS but I have not tested it yet. I did include the `pair_style` in automatic input generation with `appa`. Find the LAMMPS instructions [here](https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html). Since kokkos does not seem to significantly improve speed for the small systems considered in electrochemistry, I'd use the GRACE LAMMPS setup.

## MACE

The [MACE software stack](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html) is probably the most complicated. You need to `pip install` the following modules in your environment. **At the time of writing Python version 3.11 is recommended.**

```shell
mace_torch
cuequivariance  
cuequivariance-ops-cu12
cuequivariance-ops-torch-cu12
cuequivariance-torch
cupy-cuda12x
```

For LAMMPS, make sure you have the correct Python dependencies installed, then
```shell
git clone https://github.com/lammps/lammps.git
cd lammps
mkdir build
cd build
```
And then in the `build` folder create a jobscript,

```shell
#!/bin/bash
#SBATCH --job-name=build
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=18
#SBATCH --time=2:00:00             
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1

module purge
module load 2025
module load OpenMPI/5.0.7-GCC-14.2.0
module load CUDA/12.8.0
module load CMake/3.31.3-GCCcore-14.2.0

source ~/.bashrc
conda activate mace  # activate the correct environment 

which python
nvcc --version 

pip install cython

cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D CMAKE_CUDA_ARCHITECTURES="80" \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_EXTRA-FIX=ON \
  -D PKG_PLUMED=ON \
  -D DOWNLOAD_PLUMED=ON \
  ../cmake

make -j 18  
make install
make install-python
```

It's also possible to load the `PLUMED` module and turn `DOWNLOAD_PLUMED` off, but the Snellius software stack then loads a newer Python version as a PLUMED module dependency and this gave me issues with running the MACE MLIAP-LAMMPS interface.

Also create a file `kokkos-cuda.cmake`:
```shell
# preset that enables KOKKOS and selects CUDA compilation using the nvcc_wrapper
# enabled as well. The GPU architecture *must* match your hardware (If not manually set, Kokkos will try to autodetect it).
set(PKG_KOKKOS ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA   ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_AVX512XEON ON CACHE BOOL "" FORCE)
get_filename_component(NVCC_WRAPPER_CMD ${CMAKE_CURRENT_SOURCE_DIR}/../lib/kokkos/bin/nvcc_wrapper ABSOLUTE)
set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER_CMD} CACHE FILEPATH "" FORCE)

# If KSPACE is also enabled, use CUFFT for FFTs
set(FFT_KOKKOS "CUFFT" CACHE STRING "" FORCE)

# hide deprecation warnings temporarily for stable release
set(Kokkos_ENABLE_DEPRECATION_WARNINGS OFF CACHE BOOL "" FORCE)
```

This is for the NVIDIA A100, your HPC staff will know the options for other hardware.

Submit the jobscript to build LAMMPS.

## GRACE

Find the installation instructions [here](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/) and install `tensorpotential`:

```shell
pip install tensorpotential
```

The instructions for installing LAMMPS are [here](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/#lammps-with-grace). According to the instructions, be in the correct environment with `tensorpotential` installed and then
```shell
git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git
cd lammps
mkdir build
cd build
```

In the `build` folder, you can run this jobscript:

```shell
#!/bin/bash  
#SBATCH --job-name=build  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1 # Number of tasks per node (adjust based on cores per node)  
#SBATCH --cpus-per-task=18  
#SBATCH --time=2:00:00 # Adjust based on the expected build time  
#SBATCH --partition=gpu_a100  
#SBATCH --gpus-per-node=1  
 
source ~/.bashrc
conda activate grace
which python

module purge
module load 2025
module load OpenMPI/5.0.7-GCC-14.2.0
module load CUDA/12.8.0
module load CMake/3.31.3-GCCcore-14.2.0
module load OpenBLAS/0.3.29-GCC-14.2.0
module load FFTW.MPI/3.3.10-gompi-2025a

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_MC=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_EXTRA-DUMP=ON \
    -D PKG_PLUMED=ON \
    -D DOWNLOAD_PLUMED=ON \
../cmake

cmake --build . -- -j 8
```