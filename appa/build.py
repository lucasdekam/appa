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
import mdapackmol
from MDAnalysis import Universe


def calc_number(rho: float, v: float, mol_mass: float) -> int:
    """
    Calculate the number of molecules based on density, volume, and molecular mass.

    Parameters
    ----------
    rho : float
        Density in g/cm^3.
    v : float
        Volume in Å^3.
    mol_mass : float
        Molecular mass in g/mol.

    Returns
    -------
    int
        Number of molecules.
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
    Calculate the number of water molecules based on density and volume.

    Parameters
    ----------
    rho : float
        Water density in g/cm^3.
    v : float
        Volume in Å^3.

    Returns
    -------
    int
        Number of water molecules.
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


class WaterBox:
    """
    Represents a water box containing water molecules and optional additional structures.

    Attributes
    ----------
    boundary : Boundary
        Boundary object representing the water box.
    n_wat : int
        Number of water molecules.
    ions : Dict[str, Boundary]
        Dictionary of ion boundaries.

    Parameters
    ----------
    boundary : Boundary
        Boundary object representing the water box.
    n_wat : Optional[int], optional
        Number of water molecules. If not provided, it is calculated based on the boundary volume and water density.
    rho : float, optional
        Water density in g/cm^3, by default 1.0.
    ions : Optional[Dict[str, Boundary]], optional
        Dictionary of ion boundaries, by default None.
    """

    def __init__(
        self,
        boundary: Boundary,
        n_wat: Optional[int] = None,
        rho: float = 1.0,
        ions: Optional[Dict[str, Boundary]] = None,
    ) -> None:
        self.boundary = boundary
        if n_wat is None:
            n_wat = calc_water_number(rho, self.boundary.volume)
        self.n_wat = n_wat
        self.ions = ions if ions is not None else {}

    def write(
        self,
        fname: str,
        seed: int = -1,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Write the water box configuration to a file.

        Parameters
        ----------
        fname : str
            Output filename.
        seed : int, optional
            Random seed for reproducibility, by default -1 (random behavior).
        verbose : bool, optional
            If True, keeps temporary files, by default False.
        kwargs : dict
            Additional arguments for writing the file.
        """
        packmol_structures = []

        for symbol, boundary in self.ions.items():
            ion = Atoms(symbol, positions=[[0, 0, 0]])
            io.write("tmp.pdb", ion)
            u = Universe("tmp.pdb")
            packmol_structures.append(
                mdapackmol.PackmolStructure(
                    u,
                    number=1,
                    instructions=[
                        f"inside box {boundary.boundary_string}",
                        f"seed {seed}",
                    ],
                )
            )

        water = build.molecule("H2O")
        io.write("tmp.pdb", water)
        u = Universe("tmp.pdb")
        packmol_structures.append(
            mdapackmol.PackmolStructure(
                u,
                number=self.n_wat,
                instructions=[
                    f"inside box {self.boundary.boundary_string}",
                    f"seed {seed}",
                ],
            )
        )

        system = mdapackmol.packmol(packmol_structures)
        system.atoms.write(fname, **kwargs)
        if not verbose:
            os.remove("tmp.pdb")
            os.remove("packmol.stdout")


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
        self.n_wat = calc_water_number(1.0, self.boundary.volume)

        # Define ions
        if ions is None:
            ions = {}
        self.ions = {}
        for symbol, distance in ions.items():
            if distance <= 2.5:
                raise ValueError("Distance from surface must be greater than 2.5 Å.")
            self.ions[symbol] = Boundary(
                [
                    [0, a],
                    [0, b],
                    [
                        d_slab + distance - ion_delta_z,
                        d_slab + distance + ion_delta_z,
                    ],
                ],
                margin=[1, 1, 0],
            )

    def run(
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
        sol = WaterBox(self.boundary, n_wat=self.n_wat, ions=self.ions)
        sol.write("waterbox.xyz", verbose=verbose, seed=seed)
        waterbox = io.read("waterbox.xyz")
        waterbox.cell = self.atoms.cell
        waterbox.pbc = True

        self.atoms += waterbox
        if not verbose:
            os.remove("waterbox.xyz")
        return self.atoms

    def write(
        self,
        fname: str,
        seed: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Write the interface structure to a file.

        Parameters
        ----------
        fname : str
            Output filename.
        seed : int, optional
            Random seed for reproducibility, by default -1 (random behavior).
        verbose : bool, optional
            If True, keeps temporary files, by default False.
        """
        self.run(seed=seed, verbose=verbose)
        io.write(fname, self.atoms)


class Electrode:
    """
    Represents an electrode structure based an FCC (111) surface.

    Parameters
    ----------
    material : str
        The chemical symbol of the material to construct the electrode.
    size : List[int]
        The dimensions of the electrode in terms of unit cells [nx, ny, nz].
    a : float
        The lattice constant of the material.
    fix_layers : int, optional
        The number of atomic layers to fix in the structure (default is 2).
    """

    def __init__(
        self,
        material: str,
        size: List[int],
        a: float,
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
