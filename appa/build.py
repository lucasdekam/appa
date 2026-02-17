"""
Tools for building electrochemical interfaces.
"""

import os
from typing import Dict, List, Optional, Union
import random

import numpy as np
from ase import Atoms, build, io, constraints
from ase.cell import Cell
from scipy import constants


def calc_number(rho: float, v: float, mol_mass: float) -> int:
    """
    Calculate the number of molecules based on density in g/cm^3,
    volume in Å^3, and molecular mass in g/mol.
    """
    n = (
        rho
        * (v * (constants.angstrom / constants.centi) ** 3)
        * constants.Avogadro
        / mol_mass
    )
    return int(n)


def calc_water_number(rho: float, v: float) -> int:
    """
    Calculate the number of water molecules based on density in g/cm^3
    and volume in Å^3.
    """
    return calc_number(rho, v, 18.015)


class Boundary:
    """
    Represents a boundary with margins and provides standardized boundary and string representation.

    Attributes
    ----------
    boundary : np.ndarray
        The standardized 3x2 boundary array.
    boundary_string : str
        String representation of the boundary adjusted by the margin.
    volume : float
        Volume of the box bounded by the boundary, including the margin space.

    Parameters
    ----------
    boundary : Union[List[float], np.ndarray]
        A 3-dimensional array representing the boundary. It must be of shape (3,) or (3, 2).
    margin : Union[float, List[float], np.ndarray], optional
        Margin to apply to the boundary, by default 1.0.
    """

    def __init__(
        self,
        boundary: Union[List[float], np.ndarray],
        margin: Union[float, List[float], np.ndarray] = 1.0,
    ) -> None:
        self._boundary = np.reshape(np.array(boundary), (3, -1))
        self._margin = np.array(margin).reshape(-1)

        if self._boundary.shape[1] == 1:
            self._boundary = np.concatenate([np.zeros((3, 1)), self._boundary], axis=-1)
        elif self._boundary.shape[1] != 2:
            raise AttributeError(
                "Boundary must be [a, b, c] or [[a1, a2], [b1, b2], [c1, c2]]."
            )

        if len(self._margin) not in {1, 3}:
            raise AttributeError("Margin must have 1 or 3 elements.")

        self.boundary = self._boundary
        self.boundary_string = self._compute_boundary_string()
        self.volume = np.prod(np.diff(self.boundary, axis=-1))

    def _compute_boundary_string(self) -> str:
        """
        Computes the string representation of the boundary adjusted by the margin.

        Returns
        -------
        str
            String representation of the adjusted boundary.
        """
        adjusted_boundary = self.boundary.copy()
        adjusted_boundary[:, 0] += self._margin
        adjusted_boundary[:, 1] -= self._margin
        return np.array2string(np.transpose(adjusted_boundary).flatten())[1:-1]


def packmol_waterbox(
    boundary: Boundary,
    ions: Optional[Dict[str, Boundary]] = None,
    rho: float = 1.0,
    seed: int = -1,
    verbose: bool = False,
    **kwargs,
) -> Atoms:
    """
    Generate a water box configuration with optional ions using Packmol.

    Parameters
    ----------
    boundary : Boundary
        Boundary object representing the water box.
    ions : Optional[Dict[str, Boundary]], optional
        Dictionary of ion boundaries, by default None.
    rho : float, optional
        Water density in g/cm^3, by default 1.0.
    seed : int, optional
        Random seed for reproducibility, by default -1 (random behavior).
    verbose : bool, optional
        If True, keeps temporary files, by default False.
    kwargs : dict, optional
        Additional arguments for writing the output file.

    Returns
    -------
    Atoms
        The generated atomic structure containing water molecules and optional ions.
        Does not include a cell.
    """
    try:
        import mdapackmol
        from MDAnalysis import Universe
    except ImportError as exc:
        raise ImportError(
            "The 'mdapackmol' and 'MDAnalysis' packages are required for this function."
        ) from exc

    # Generate a unique random ID for file naming
    tmp_pdb = f"tmp.pdb"
    out_xyz = f"waterbox.xyz"

    def _get_packmol_structure(
        atoms: Atoms, number: int, boundary: Boundary, seed: int
    ):
        io.write(tmp_pdb, atoms)
        u = Universe(tmp_pdb)
        return mdapackmol.PackmolStructure(
            u,
            number=number,
            instructions=[
                f"inside box {boundary.boundary_string}",
                f"seed {seed}",
            ],
        )

    if ions is None:
        ions = {}

    packmol_structures = []

    for symbol, ion_boundary in ions.items():
        ion = Atoms(symbol, positions=[[0, 0, 0]])
        packmol_structures.append(_get_packmol_structure(ion, 1, ion_boundary, seed))

    water = build.molecule("H2O")
    packmol_structures.append(
        _get_packmol_structure(
            water,
            calc_water_number(rho, boundary.volume),
            boundary,
            seed,
        )
    )

    system = mdapackmol.packmol(packmol_structures)
    system.atoms.write(out_xyz, **kwargs)
    return_atoms = io.read(out_xyz)
    if not verbose:
        os.remove(tmp_pdb)
        os.remove(out_xyz)
        try:
            os.remove("packmol.stdout")
        except FileNotFoundError:
            pass
    return return_atoms


class Interface:
    """
    Represents an interface with an electrode and water box.

    Attributes
    ----------
    atoms : Atoms
        Combined structure with the electrode and water box.
    boundary : Boundary
        Boundary object representing the water box.
    n_wat : int
        Number of water molecules in the water box.
    ions : Dict[str, Boundary]
        Dictionary of ion boundaries.

    Parameters
    ----------
    electrode : Atoms
        Electrode structure.
    d_water : float, optional
        Thickness of the water layer in Å, by default 30.0.
    d_vacuum : float, optional
        Thickness of the vacuum layer in Å, by default 15.0.
    ions : Optional[Dict[str, float]], optional
        Dictionary of ions and their distances from the surface, by default None.
    ion_delta_z : float, optional
        Half-width of the ion boundary layer in Å, by default 2.5.
    """

    def __init__(
        self,
        electrode: Atoms,
        d_water: float = 30.0,
        d_vacuum: float = 15.0,
        ions: Optional[Dict[str, float]] = None,
        ion_delta_z: float = 2.5,
        rho: float = 1.0,
    ) -> None:
        # Adjust electrode coordinates and calculate slab thickness
        coord = electrode.get_positions()
        coord[:, 2] -= coord[:, 2].min()
        d_slab = coord[:, 2].max()

        # Ensure the cell is orthogonal
        cellpar = electrode.cell.cellpar()
        if not np.allclose(cellpar[3:], [90.0, 90.0, 90.0]):
            raise ValueError("Cell should be orthogonal with angles 90-90-90.")

        # Create a new cell with updated dimensions
        a, b = cellpar[:2]
        new_cell = Cell.new([a, b, d_slab + d_water + d_vacuum])

        # Initialize the interface Atoms object
        self.atoms = Atoms(
            symbols=electrode.get_chemical_symbols(),
            positions=coord,
            cell=new_cell,
            pbc=True,
            constraint=electrode.constraints,
        )

        # Define the boundary for the water box
        self.boundary = Boundary(
            [[0, a], [0, b], [d_slab, d_slab + d_water]],
            margin=[1, 1, 2],
        )
        self.rho = rho

        # Define ions
        self.ion_boundary_dict = self._make_ion_boundary_dict(
            a=a, b=b, d_slab=d_slab, ion_delta_z=ion_delta_z, ions=ions
        )

    def _make_ion_boundary_dict(
        self,
        a: float,
        b: float,
        d_slab: float,
        ion_delta_z: float,
        ions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Boundary]:
        if ions is None:
            ions = {}

        ion_boundary_dict = {}
        for symbol, distance in ions.items():
            if distance <= 2.5:
                raise ValueError("Distance from surface must be greater than 2.5 Å.")
            z_min = d_slab + distance - ion_delta_z
            z_max = d_slab + distance + ion_delta_z
            ion_boundary_dict[symbol] = Boundary(
                [
                    [0, a],
                    [0, b],
                    [
                        z_min,
                        z_max,
                    ],
                ],
                margin=[1, 1, 0],
            )
        return ion_boundary_dict

    def add_electrolyte(
        self,
        seed: int = -1,
        verbose: bool = False,
    ) -> Atoms:
        """
        Add a water box to the interface.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default -1 (random behavior).
        verbose : bool, optional
            If True, keeps temporary files, by default False.

        Returns
        -------
        Atoms
            Combined structure with the water box and electrode.
        """
        waterbox = packmol_waterbox(
            self.boundary,
            self.ion_boundary_dict,
            self.rho,
            seed,
        )
        waterbox.cell = self.atoms.cell
        waterbox.pbc = True

        self.atoms += waterbox
        return self.atoms


class Electrode:
    """
    Represents an electrode structure based an fcc(111) surface.

    Parameters
    ----------
    material : str
        The chemical symbol of the material to construct the electrode.
    size : List[int]
        The dimensions of the electrode in terms of unit cells [nx, ny, nz].
    a : float
        The lattice constant of the material. If None, takes the ASE default.
    fix_layers : int, optional
        The number of atomic layers to fix in the structure (default is 2).
    """

    def __init__(
        self,
        material: str,
        size: List[int],
        a: Optional[float] = None,
        fix_layers: int = 2,
    ):
        self.atoms = build.fcc111(
            symbol=material,
            a=a,
            size=size,
            orthogonal=True,
            periodic=True,
        )
        indices = np.array([atom.index for atom in self.atoms])
        self.atoms.set_constraint(
            constraints.FixAtoms(indices=indices[: size[0] * size[1] * fix_layers])
        )
        self.size = size

    def add_hydrogens(self, coverage: float, topsite_probability: float):
        """
        Adds hydrogen atoms to the electrode surface with specified coverage and site preference.

        Parameters
        ----------
        coverage : float
            The fraction of surface sites to be covered with hydrogen (0.0 to 1.0).
        topsite_probability : float
            The probability of placing hydrogen on top sites versus FCC sites (0.0 to 1.0).
        """
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if random.random() < coverage:
                    if random.random() < topsite_probability:
                        site = "ontop"
                        height = 1.5
                    else:
                        site = "fcc"
                        height = 1.0
                    build.add_adsorbate(
                        self.atoms,
                        "H",
                        height=height,
                        offset=(i, j),
                        position=site,
                    )


def build_bulk_electrolyte(
    cell: tuple[float, float, float],
    rho: float = 1.0,
    seed: int = -1,
    ions: Optional[List[str]] = None,
):
    """
    Generate a bulk electrolyte system with specified ions and density.

    Parameters
    ----------
    cell : tuple[float, float, float]
        Dimensions of the simulation cell as (x, y, z).
    rho : float, optional
        Density of the electrolyte in g/cm³. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is -1 (no specific seed).
    ions : List[str], optional
        List of ion types to include in the electrolyte. Default is None.
        Include ions multiple times if you want multiple ions of the same type
        in the electrolyte.
    """
    if ions is None:
        ions = []

    boundary = Boundary(cell, margin=1.0)

    ion_boundary_dict = {}
    for ion in ions:
        ion_boundary_dict[ion] = boundary

    atoms = packmol_waterbox(
        boundary=boundary,
        ions=ion_boundary_dict,
        rho=rho,
        seed=seed,
    )
    atoms.cell = Cell.new(cell)
    atoms.set_pbc(True)
    return atoms
