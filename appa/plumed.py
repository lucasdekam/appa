from typing import Optional
from ase import Atoms
import numpy as np

DEFAULT_STRIDE = 10


def generate_plumed_volmer(
    oxygen_id: int,
    hydrogen_id: int,
    surface_id: int,
    stride: int = DEFAULT_STRIDE,
    colvar_file: str = "COLVAR",
    outfile: Optional[str] = None,
):
    """
    Generate a PLUMED input file for a Volmer-step umbrella sampling window.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object representing the initial simulation structure.
    oxygen_id : int
        Atom ID of the oxygen atom (zero-based, like ASE).
    hydrogen_id : int
        Atom ID of the transferring hydrogen atom (zero-based, like ASE).
    surface_id : int
        Atom ID of the platinum atom (zero-based, like ASE).
    cv_target : float
        Target value of the reaction coordinate ξ = d(O-H) - d(M-H) in Å.
    kappa : float, optional
        Umbrella force constant in eV/Å^2 (default: 4.0).
    stride : int, optional
        How often to print to COLVAR, in number of timesteps (default: 10).
    warmup: int, optional
        Warm-up period in which the bias is shifted from the initial CV value
        to the target value (in number of timesteps).
    colvar_file : str, optional
        Name of the COLVAR output file.
    outfile : str or None, optional
        If provided, write the PLUMED input to this file.

    Returns
    -------
    str
        The PLUMED input file as a string.
    """
    plumed_input = f"""UNITS LENGTH=A ENERGY=eV TIME=ps

# Distances defining the Volmer coordinate
d_OH: DISTANCE ATOMS={oxygen_id + 1},{hydrogen_id + 1}
d_MH: DISTANCE ATOMS={surface_id + 1},{hydrogen_id + 1}

# Keeps OH and MH distances within a physical range
uwall: UPPER_WALLS ARG=d_OH,d_MH AT=5.0,5.0 KAPPA=100.0,100.0

metad: METAD ...
    ARG=d_OH,d_MH 
    PACE=1000 
    HEIGHT=0.005
    SIGMA=0.05,0.05
    FILE=HILLS 
    BIASFACTOR=10.0 
    TEMP=300.0
    GRID_MIN=0.5,0.5 
    GRID_MAX=5.5,5.5
    GRID_BIN=250,250
...

# Output
PRINT STRIDE={stride} ARG=metad.bias,uwall.bias,d_OH,d_MH FILE={colvar_file}
"""

    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(plumed_input)

    return plumed_input
