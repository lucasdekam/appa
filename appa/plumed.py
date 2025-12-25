def generate_plumed_volmer(
    oxygen_id,
    hydrogen_id,
    surface_id,
    cv_target,
    kappa=4.0,
    stride=10,
    max_dist_mh=2.8,
    colvar_file="COLVAR",
    zero_based=True,
    outfile=None,
):
    """
    Generate a PLUMED input file for a Volmer-step umbrella sampling window.

    Parameters
    ----------
    oxygen_id : int
        Atom ID of the oxygen atom.
    hydrogen_id : int
        Atom ID of the transferring hydrogen atom.
    surface_id : int
        Atom ID of the platinum atom.
    cv_target : float
        Target value of the reaction coordinate ξ = d(O-H) - d(Pt-H) in Å.
    kappa : float, optional
        Umbrella force constant in eV/Å^2 (default: 4.0).
    stride : int, optional
        STRIDE for PRINT (default: 10).
    max_dist_mh: float, optional
        Maximum M-H distance to avoid proton escape (default: 2.8).
    colvar_file : str, optional
        Name of the COLVAR output file.
    zero_based : bool, optional
        Whether the provided atom IDs are zero-based (ASE-style).
        If True, +1 is applied to all atom IDs for PLUMED.
    outfile : str or None, optional
        If provided, write the PLUMED input to this file.

    Returns
    -------
    str
        The PLUMED input file as a string.
    """

    # PLUMED uses 1-based atom indexing
    shift = 1 if zero_based else 0
    oxygen = oxygen_id + shift
    hydrogen = hydrogen_id + shift
    metal = surface_id + shift

    plumed_input = f"""UNITS LENGTH=A ENERGY=eV

# Distances defining the Volmer coordinate
d_OH: DISTANCE ATOMS={oxygen},{hydrogen}
d_MH: DISTANCE ATOMS={metal},{hydrogen}

# Reaction coordinate xi = d(O-H) - d(Pt-H)
xi: COMBINE ARG=d_OH,d_MH COEFFICIENTS=1,-1 PERIODIC=NO

# Harmonic umbrella restraint
restraint: RESTRAINT ARG=xi AT={cv_target} KAPPA={kappa}

# Upper wall to avoid proton escape
uwall: UPPER_WALLS ARG=d_MH AT={max_dist_mh} KAPPA=10.0

# Output
PRINT STRIDE={stride} ARG=xi,restraint.bias,uwall.bias,d_OH,d_MH FILE={colvar_file}
"""

    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(plumed_input)

    return plumed_input
