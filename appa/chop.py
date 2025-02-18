"""
Cutting up interface configurations 
"""

import numpy as np
from ase.geometry import get_layers
from .interface import Interface

MIN_SURFACE_CUT_DISTANCE = 3.0
MIN_BULK_CONFIG_SIZE = 8


def cut_atoms_z(
    atoms,
    max_cut_z=15.0,
    interface_prob=0.7,
    layer_tolerance=1.44,
    metal="Pt",
    oh_cutoff=1.3,
):
    """Cut an ASE Atoms object along the z-direction while preserving x and y.

    - max_cut_z: Maximum thickness of the extracted slab (in Å).
    - interface_prob: Probability of sampling at the interface vs. bulk water.
    """

    interface = Interface(atoms)

    # With 50% chance, flip the box upside down
    if np.random.rand() < 0.5:
        interface.positions[:, 2] = -interface.positions[:, 2]
        interface.wrap()

    # Find metal surface & move to bottom of the box
    surf_ids = interface.find_surf_idx(
        element="Pt",
        tolerance=layer_tolerance,
        dsur="up",
        check_cross_boundary=True,
    )
    surf_z = interface[surf_ids].positions[:, 2].mean()
    interface.positions -= np.array([0, 0, surf_z])

    # Wrap water, but keep metal slab together
    water = interface[(interface.symbols == "O") | (interface.symbols == "H")]
    water.wrap()
    interface = interface[(interface.symbols == metal)] + water

    if np.random.rand() < interface_prob:
        # Identify layers
        metal_atoms = interface[interface.symbols == metal]
        temp_shift = metal_atoms.positions[:, 2].min()
        metal_atoms.positions -= np.array([0, 0, temp_shift])
        layers, _ = get_layers(metal_atoms, miller=(0, 0, 1), tolerance=layer_tolerance)
        metal_atoms.positions += np.array([0, 0, temp_shift])
        unique_layers = np.unique(layers)

        # Randomize the size of the interface cut
        num_metal_layers = np.random.choice([2, 3, 4, 5], p=[0.1, 0.6, 0.25, 0.05])
        layer_cutoff = unique_layers[-num_metal_layers]
        z_min = metal_atoms[layers >= layer_cutoff].positions[:, 2].min()
        z_max = np.random.uniform(MIN_SURFACE_CUT_DISTANCE, max_cut_z)

        # Cut the region, select only Pt and O
        new = interface[
            ((interface.symbols == "O") | (interface.symbols == metal))
            & (interface.positions[:, 2] >= z_min)
            & (interface.positions[:, 2] <= z_max)
        ]

        # Add all hydrogens, assign them to oxygens to form water molecules
        new += interface[interface.symbols == "H"]
        water_dict = new.identify_water_molecules(oh_cutoff=oh_cutoff)
        h_list = [h for h_list in water_dict.values() for h in h_list]
        hydrogens = new[new.symbols == "H"][h_list]
        new = new[(new.symbols == metal) | (new.symbols == "O")] + hydrogens

        # Center the slab and add vacuum
        new.center(vacuum=5, axis=2)
        new.positions -= np.array([0, 0, new.positions[:, 2].min()])
        new.info["structure"] = "interface"

    else:
        # Define a random bulk region around the center of the water layer
        water_center = interface[interface.symbols == "O"].positions[:, 2].mean()
        width = np.random.uniform(MIN_BULK_CONFIG_SIZE, max_cut_z)
        z_min = water_center - width / 2
        z_max = water_center + width / 2

        # Cut the region
        new = interface[
            (interface.symbols == "O")
            & (interface.positions[:, 2] >= z_min)
            & (interface.positions[:, 2] <= z_max)
        ]

        # Add all hydrogens, assign them to oxygens to form water molecules
        new += interface[interface.symbols == "H"]
        water_dict = new.identify_water_molecules(oh_cutoff=oh_cutoff)
        h_list = [h for h_list in water_dict.values() for h in h_list]
        hydrogens = new[new.symbols == "H"][h_list]
        new = new[(new.symbols == metal) | (new.symbols == "O")] + hydrogens

        # Center the slab and add vacuum
        new.center(vacuum=0, axis=2)
        new.positions -= np.array([0, 0, new.positions[:, 2].min()])
        new.info["structure"] = "bulk"

    # Identify water molecules
    hydrogens = new[new.symbols == "H"]
    oxygens = new[new.symbols == "O"]
    water_dict = new.identify_water_molecules(oh_cutoff=1.2)
    o_indices = list(water_dict.keys())
    h_indices = [h for h_list in water_dict.values() for h in h_list]

    # Add up valid water molecules and platinum atoms
    new = oxygens[o_indices] + hydrogens[h_indices] + new[new.symbols == "Pt"]

    return new
