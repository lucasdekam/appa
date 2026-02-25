# Training models

Training models is done by the respective packages, not through `appa`. `appa` does provide tools to extract single-atom energies, in case you do not want to include these in the training set but supply them separately.

```sh
Usage: appa convert extract-isolated [OPTIONS] XYZ_FILE OUT_XYZ

  Extract isolated-atom reference energies from an XYZ file and write a new
  XYZ file without the isolated-atom configurations.

Options:
  --help  Show this message and exit.
```

[GRACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/quickstart/#grace-model-parameterization) provides input defaults with the `gracemaker -t` tool. 

If your XYZ training set does not have the default ASE `energy` and `force` fields, it will not work with GRACE. You can convert an XYZ training set with `DFT_energy` and `DFT_forces` fields to a GRACE-compatible `.pkl.gz` file (a pickled and gzipped Pandas dataframe):

```sh
Usage: appa convert xyz2grace [OPTIONS] XYZ_FILE OUT_FILE

  Convert an extxyz file to a GRACE-compatible DataFrame.

Options:
  --subtract-reference  Subtract reference energies from isolated atom configs
  --help                Show this message and exit.
```

For example:

```sh
appa convert xyz2grace my_input.xyz my_output.pkl.gz
```

For MACE, you can use an input yaml such as 

```yaml
name: mace
model: ScaleShiftMACE
train_file: mytrain.xyz
valid_file: myvalid.xyz
seed: 1
r_max: 5.0
num_interactions: 2
hidden_irreps: 64x0e + 64x1o 
energy_weight: 1.0
forces_weight: 10.0
energy_key: DFT_energy
forces_key: DFT_forces
lr: 0.01
scaling: rms_forces_scaling
batch_size: 3
max_num_epochs: 500
patience: 10
eval_interval: 1
ema: true
ema_decay: 0.99
swa: true
start_swa: 300
amsgrad: true
default_dtype: float32
device: cuda
E0s:
  1: -1.20657996  # adapt single-atom energies
  8: -1.59725568
  79: -0.19135585
enable_cueq: true
restart_latest: true
```

Then submit a job with 
```shell
mace_run_train --config input.yaml
```

The NequIP [repo contains an example config](https://github.com/mir-group/nequip/blob/main/configs/tutorial.yaml).
