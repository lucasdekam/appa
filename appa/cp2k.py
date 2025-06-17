"""
Tool for writing CP2K inputs for energy+force calculations.
"""

import os
import numpy as np
from ase.io import write
from ase import Atoms
import yaml

from cp2k_input_tools.generator import CP2KInputGenerator

DEFAULT_XYZ_FILENAME = "coord.xyz"


def read_params(filename: str) -> dict:
    """
    The YAML parameter file should have the following structure:

    ```
    'dft':
        ... # CP2K input file DFT section according to the cp2k_input_tools format
    'global':
        ... # global parameters
    'kinds':
        Cs:
            basis_set: DZVP-MOLOPT-SR-GTH
            potential: GTH-PBE-q9
        ... # specifying the basis sets and potentials for the different elements that might
            # (but do not have to) appear in the configurations
    ```
    """
    with open(filename, "r", encoding="utf-8") as fhandle:
        input = yaml.safe_load(fhandle)
    return input


def _get_cell_dict(atoms: Atoms) -> dict:
    return {
        "abc": [a.item() for a in atoms.cell.cellpar()[:3]],
        "alpha_beta_gamma": [a.item() for a in atoms.cell.cellpar()[3:]],
    }


def _get_kind_list(atoms: Atoms, kind_params: dict) -> list:
    kinds_present = [str(sym) for sym in np.unique(atoms.get_chemical_symbols())]
    return [
        {
            "basis_set": [kind_params[key]["basis_set"]],
            "potential": kind_params[key]["potential"],
            "_": key,
        }
        for key in kinds_present
    ]


def _get_force_eval_dict(atoms: Atoms, params: dict, print_forces: bool = True) -> dict:
    force_eval = {}
    force_eval["+dft"] = params["dft"]
    force_eval["+subsys"] = {
        "+cell": _get_cell_dict(atoms),
        "+topology": {
            "coord_file_format": "xyz",
            "coord_file_name": DEFAULT_XYZ_FILENAME,
        },
        "+kind": _get_kind_list(atoms, params["kinds"]),
    }
    force_eval["method"] = "quickstep"

    if print_forces:
        force_eval["+print"] = {
            "forces": {"_": "ON", "add_last": "numeric", "filename": "./forces.out"}
        }
    return force_eval


def _get_input_dict(
    atoms: Atoms, params: dict, project_name: str = "cp2k", print_forces: bool = True
) -> dict:
    # Setup global
    global_ = params["global"]
    global_["project_name"] = project_name

    # Setup force_eval
    force_eval = _get_force_eval_dict(atoms, params, print_forces=print_forces)

    cp2k = {
        "+force_eval": [force_eval],
        "+global": global_,
    }
    return cp2k


def write_input(
    directory: str,
    atoms: Atoms,
    params: dict,
    project_name: str = "cp2k",
    print_forces: bool = True,
):
    """
    Writes a CP2K input file and corresponding XYZ file for a given atomic structure.

    Parameters:
        directory (str): Path to the directory where input files will be written.
        atoms (Atoms): Atomic structure to be used in the simulation.
        params (dict): Parameter dictionary containing DFT settings and basis sets.
        project_name (str, optional): Name of the CP2K project. Defaults to "cp2k".
        print_forces (bool, optional): Whether to include force printing in the input. Defaults to True.
    """
    cp2k = _get_input_dict(atoms, params, project_name, print_forces)

    # Write input file
    generator = CP2KInputGenerator()

    print(f"Writing to directory {directory}...", flush=True)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "input.inp"), "w", encoding="utf-8") as fhandle:
        for line in generator.line_iter(cp2k):
            fhandle.write(f"{line}\n")

    # Write atomistic configuration
    write(os.path.join(directory, DEFAULT_XYZ_FILENAME), atoms)
