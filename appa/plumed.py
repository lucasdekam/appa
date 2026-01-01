from typing import Optional
from ase import Atoms
import numpy as np

DEFAULT_STRIDE = 10
# DEFAULT_KAPPA = 4.0
# DEFAULT_WARMUP = 2000
# atoms: Atoms,
# cv_target: float,
# warmup: int = DEFAULT_WARMUP,
# kappa: float = DEFAULT_KAPPA,


def generate_plumed_volmer_2(
    atoms: Atoms,
    hydrogen_id: int,
    stride: int = DEFAULT_STRIDE,
    colvar_file: str = "COLVAR",
    outfile: Optional[str] = None,
):
    z_pt = atoms.positions[atoms.symbols == "Pt", 2].max()
    oxygen_indices = [atom.index + 1 for atom in atoms if atom.symbol == "O"]
    oxygen_numbers = ",".join(map(str, oxygen_indices))
    hydrogen_number = hydrogen_id + 1
    plumed_input = f"""UNITS LENGTH=A ENERGY=eV TIME=ps

oxygens: GROUP ATOMS={oxygen_numbers}
proton: GROUP ATOMS={hydrogen_number}

# CV1: Coordination Number (1.0 = bonded, 0.0 = desorbed)
cn: COORDINATION GROUPA=proton GROUPB=oxygens R_0=1.1 NN=6 MM=12

# CV2: Height above Pt surface
pos_H: POSITION ATOM={hydrogen_number}
z_H: COMBINE ARG=pos_H.z COEFFICIENTS=1.0 PARAMETERS={z_pt:.2f} POWERS=1 PERIODIC=NO

# Walls
uwall: UPPER_WALLS ARG=z_H AT=5.0 KAPPA=100.0
lwall: LOWER_WALLS ARG=z_H AT=-1.0 KAPPA=100.0

# Metadynamics
metad: METAD ...
    ARG=cn,z_H
    PACE=1000 HEIGHT=0.005 SIGMA=0.1,0.1
    GRID_MIN=-0.1,-1.5 GRID_MAX=3.0,6.0 GRID_BIN=300,300
    BIASFACTOR=10.0 TEMP=300.0 FILE=HILLS
...

# Output
PRINT STRIDE={stride} ARG=metad.bias,uwall.bias,z_H,cn FILE={colvar_file}
"""

    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(plumed_input)

    return plumed_input


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


# Upper wall (removed now):
# uwall: UPPER_WALLS ARG=d_MH AT={max_dist_mh} KAPPA=10.0
# print uwall.bias also


# Reaction coordinate xi = d(O-H) - d(Pt-H)
# xi: COMBINE ARG=d_OH,d_MH COEFFICIENTS=1,-1 PERIODIC=NO

# Harmonic umbrella restraint
# restraint: MOVINGRESTRAINT ...
#     ARG=xi
#     STEP0=0 AT0={xi0:.2f} KAPPA0=0.0
#     STEP1={warmup} AT1={cv_target:.2f} KAPPA1={kappa}
# ...

# pos = atoms.positions
# d_OH = np.linalg.norm(pos[oxygen_id, :] - pos[hydrogen_id, :])
# d_MH = np.linalg.norm(pos[surface_id, :] - pos[hydrogen_id, :])
# xi0 = d_OH - d_MH
