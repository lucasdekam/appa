<img src="docs/appa.png" alt="appa's lost days" width="190"/>

# Appa

Miscellaneous code I use in my research.

(Appa is a pet flying bison in the series *Avatar: The Last Airbender*. He speeds up the journey of the Avatar by flying him and his friends around.)

See the [documentation/tutorial](https://lucasdekam.github.io/appa/). 

<!-- ## Writing CP2K input files
The `appa.cp2k` module contains tools to write CP2K input files. The DFT settings are usually fixed for a given project, and are written in a `params.yaml` file. The DFT section in this file has a similar structure to the dictionaries that can be read and written by [cp2k-input-tools](https://github.com/cp2k/cp2k-input-tools). The `params.yaml` file also specifies what basis sets and pseudopotentials are used for the different elements that might appear in configurations. The function `read_params` reads the YAML parameter file, and `write_input` writes a `coord.xyz` and `input.inp` file to the specified directory.

```python
from appa.cp2k import read_params, write_input
from ase.io import read

params = read_params("examples/cp2k_params.yaml")
params['dft']['+mgrid']['cutoff'] = 500
atoms = read('coords.xyz')
write_input("results", atoms, params)
``` -->

## Todo

Writing more CLI utilities, such as 

* Performing analysis of trained models, like parity plots
