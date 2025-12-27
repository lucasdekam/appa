from typing import Optional
from ase import Atoms
import numpy as np

DEFAULT_MAX_MH_DISTANCE = 3
DEFAULT_STRIDE = 10
DEFAULT_KAPPA = 4.0
DEFAULT_WARMUP = 2000.0


def generate_plumed_volmer(
    atoms: Atoms,
    oxygen_id: int,
    hydrogen_id: int,
    surface_id: int,
    cv_target: float,
    kappa: float = DEFAULT_KAPPA,
    stride: int = DEFAULT_STRIDE,
    max_dist_mh: float = DEFAULT_MAX_MH_DISTANCE,
    warmup: float = DEFAULT_WARMUP,
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
    max_dist_mh: float, optional
        Maximum M-H distance to avoid proton escape (default: 2.8).
    warmup: float, optional
        Warm-up period in which the bias is shifted from the initial CV value
        to the target value (number of timesteps).
    colvar_file : str, optional
        Name of the COLVAR output file.
    outfile : str or None, optional
        If provided, write the PLUMED input to this file.

    Returns
    -------
    str
        The PLUMED input file as a string.
    """
    pos = atoms.positions
    d_OH = np.linalg.norm(pos[oxygen_id, :] - pos[hydrogen_id, :])
    d_MH = np.linalg.norm(pos[surface_id, :] - pos[hydrogen_id, :])
    xi0 = d_OH - d_MH

    plumed_input = f"""UNITS LENGTH=A ENERGY=eV

# Distances defining the Volmer coordinate
d_OH: DISTANCE ATOMS={oxygen_id + 1},{hydrogen_id + 1}
d_MH: DISTANCE ATOMS={surface_id + 1},{hydrogen_id + 1}

# Reaction coordinate xi = d(O-H) - d(Pt-H)
xi: COMBINE ARG=d_OH,d_MH COEFFICIENTS=1,-1 PERIODIC=NO

# Move restraint from initial position to target value
t1: TIME 
target: MATHEVAL ARG=t FUNC="{xi0} + ({cv_target} - {xi0})*min(1,t1/{warmup})" PERIODIC=NO

# Harmonic umbrella restraint
restraint: RESTRAINT ARG=xi AT=target KAPPA={kappa}

# Upper wall to avoid proton escape
uwall: UPPER_WALLS ARG=d_MH AT={max_dist_mh} KAPPA=10.0

# Output
PRINT STRIDE={stride} ARG=xi,restraint.bias,uwall.bias,d_OH,d_MH FILE={colvar_file}
"""

    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(plumed_input)

    return plumed_input
