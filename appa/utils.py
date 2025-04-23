"""
Reading and writing constraints in ASE
"""

from typing import Optional
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase import Atoms


def write_with_fixatoms(fname: str, atoms: Atoms, verbose=True, **kwargs):
    """
    Write an Atoms object to a file, including fixed atom constraints.

    This function checks for any `FixAtoms` constraints in the provided
    `Atoms` object. If such a constraint is found, it marks the fixed
    atoms in the `Atoms` object and writes the updated object to the
    specified file.

    Parameters
    ----------
    fname : str
        The file name or path where the `Atoms` object should be written.
    atoms : Atoms
        The `Atoms` object containing atomic positions, constraints, and other properties.
    **kwargs
        Additional keyword arguments passed to the ASE `write` function.
    """
    fix_atoms = None
    for cstr in atoms.constraints:
        if isinstance(cstr, FixAtoms):
            fix_atoms = cstr
            if verbose:
                print(f"Constraint {fix_atoms} found, writing...")
            break

    fixed = np.zeros(len(atoms), dtype=bool)
    fixed[fix_atoms.index] = True
    atoms.arrays["fixed"] = fixed
    write(fname, atoms, **kwargs)


def read_with_fixatoms(
    fname: str, ignore_element: Optional[str] = None, **kwargs
) -> Atoms:
    """
    Load atomic structure from a file and apply fixed atom constraints.

    This function reads atomic structure data from the specified file,
    retrieves the "fixed" array from the atomic data, and applies a
    `FixAtoms` constraint to the atoms based on the mask provided in
    the "fixed" array. Additionally, it allows for the exclusion of
    specific elements from the structure, such as ignoring a neon cap
    on the electrolyte box.

    Parameters
    ----------
    fname : str
        The path to the file containing atomic structure data.
    ignore_element : Optional[str]
        The chemical symbol of an element to exclude from the structure.
        For example, this can be used to ignore a neon cap on the electrolyte box.
    **kwargs
        Additional keyword arguments passed to the ASE `read` function.

    Returns
    -------
    Atoms
        An `Atoms` object with the loaded structure and applied constraints.
    """
    atoms = read(fname, **kwargs)
    if ignore_element is not None:
        indices = [a.index for a in atoms if a.symbol != ignore_element]
        atoms = atoms[indices]
    fixed = atoms.arrays["fixed"]
    atoms.set_constraint(FixAtoms(mask=fixed))
    return atoms
