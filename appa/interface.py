"""
Adapted from github.com/robinzyb/ectoolkits 
"""

from typing import Literal, Dict, List
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase import units
from MDAnalysis.lib.distances import minimize_vectors, capped_distance


class Interface(Atoms):
    """
        Object inherited from Atoms object in ASE.
        Add method for vacuum slab model
    Args:
        Atoms (_type_): Atoms object int ASE
    """

    def find_element_idx_list(self, element: str) -> list:
        """
        find atom index provided that element symbol

        _extended_summary_

        Args:
            element (str): element symbol

        Returns:
            list: list of atom indices
        """
        cs = self.get_chemical_symbols()
        cs = np.array(cs)
        idx_list = np.where(cs == element)[0]
        return list(idx_list)

    def find_surf_idx(
        self,
        element: str = None,
        tolerance: float = 1.4,
        dsur: Literal["up", "dw"] = "up",
        check_cross_boundary=False,
        trans_z_dist=5,
    ) -> list:
        """
            find atom indexs at surface

        _extended_summary_

        Args:
            element (str): element symbol
            tolerance (float, optional): tolerance for define a layer. Defaults to 1.4.
            dsur (str, optional): direction of surface, 'up' or 'dw'. for a vacuum-slab model,
            you have up surface and down surface. Defaults to 'up'.

        Returns:
            list: list of atom indices
        """
        tmp = self.copy()
        if check_cross_boundary:
            while tmp.crosses_z_boundary(element=element):
                tmp.translate([0, 0, trans_z_dist])
                tmp.wrap()

        if element:
            idx_list = tmp.find_element_idx_list(element)
            z_list = tmp[idx_list].get_positions().T[2]
        else:
            z_list = tmp.get_positions().T[2]

        if dsur == "up":
            z = z_list.max()
        elif dsur == "dw":
            z = z_list.min()
        else:
            raise ValueError("dsur should be 'up' or 'dw'")

        zmin = z - tolerance
        zmax = z + tolerance
        idx_list = tmp.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)

        return idx_list

    def del_surf_layer(
        self, element: str = None, tolerance=0.1, dsur="up", check_cross_boundary=False
    ):
        """delete the layer atoms,

        _extended_summary_

        Args:
            element (str, optional): _description_. Defaults to None.
            tolerance (float, optional): _description_. Defaults to 0.1.
            dsur (str, optional): _description_. Defaults to 'up'.

        Returns:
            _type_: _description_
        """

        del_list = self.find_surf_idx(
            element=element,
            tolerance=tolerance,
            dsur=dsur,
            check_cross_boundary=check_cross_boundary,
        )

        tmp = self.copy()
        del tmp[del_list]
        return tmp

    def find_idx_from_range(self, zmin: int, zmax: int, element: str = None) -> list:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): minimum in z
            zmax (int): maximum in z
            element (str, optional): element symbol, None means all atoms. Defaults to None.

        Returns:
            list: list of atom indices
        """
        idx_list = []
        if element:
            for atom in self:
                if atom.symbol == element:
                    if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                        idx_list.append(atom.index)
        else:
            for atom in self:
                if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                    idx_list.append(atom.index)
        return idx_list

    def del_from_range(self, zmin: int, zmax: int, element: str = None) -> Atoms:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): _description_
            zmax (int): _description_
            element (str, optional): _description_. Defaults to None.

        Returns:
            Atoms: _description_
        """
        tmp = self.copy()
        del_idx_list = self.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)

        del tmp[del_idx_list]

        return tmp

    def crosses_z_boundary(self, element: str = None):
        """# check if slab cross z boundary"""
        if element:
            metal_idx_list = self.find_element_idx_list(element=element)
        else:
            metal_idx_list = list(range(len(self)))

        cellpar = Cell(self.cell).cellpar()

        coords = Atoms(self[metal_idx_list]).get_positions()
        coords_z = coords[:, 2]

        coord_z_max = coords[coords_z.argmax()]
        coord_z_min = coords[coords_z.argmin()]
        vec_raw = coord_z_max - coord_z_min

        vec_minimized = minimize_vectors(vectors=vec_raw, box=cellpar)

        if np.isclose(vec_minimized[2], vec_raw[2], atol=1e-5, rtol=0):
            return False
        return True

    def identify_water_molecules(self, oh_cutoff: float) -> Dict[int, List[int]]:
        """
        Identify water molecules
        """
        h_atoms = self[self.symbols == "H"]
        o_atoms = self[self.symbols == "O"]

        water_dict = {i: [] for i in range(len(o_atoms))}

        for h_idx, hpos in enumerate(h_atoms.positions):
            pairs, distances = capped_distance(
                hpos,
                o_atoms.positions,
                max_cutoff=oh_cutoff,
                box=Cell(self.cell).cellpar(),
                return_distances=True,
            )

            if len(pairs) > 0:
                closest_o_idx = pairs[np.argmin(distances)][1]
                water_dict[closest_o_idx].append(h_idx)

        water_dict = {
            key: value for key, value in water_dict.items() if len(value) == 2
        }
        return water_dict

    def equilibrate(self, calculator: Calculator, temperature, steps):
        """
        Pre-equilibrate with MD, 0.5fs timestep
        """
        self.calc = calculator
        dyn = Langevin(
            atoms=self,
            timestep=0.5 * units.fs,
            temperature_K=temperature,
            friction=0.01 / units.fs,
        )
        dyn.run(steps=steps)
