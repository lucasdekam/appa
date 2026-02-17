import numpy as np
from ase import units
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def harmonic_wall_force(atoms, zmax, k=1.0):
    positions = atoms.get_positions()
    forces = np.zeros_like(positions)

    for i, pos in enumerate(positions):
        if pos[2] > zmax:
            dz = pos[2] - zmax
            forces[i, 2] -= k * dz

    return forces


def wall_hook(atoms: Atoms, zmax=30.0, k=1.0):
    wall_forces = harmonic_wall_force(atoms, zmax, k)
    atoms.forces = atoms.get_forces() + wall_forces


def log_md_status(atoms: Atoms, dyn):
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    time_fs = dyn.get_time() / units.fs

    print(
        f"{time_fs:8.1f} fs | "
        f"E_pot = {epot: .3f} eV | "
        f"E_kin = {ekin: .3f} eV | "
        f"T = {temp:6.1f} K | "
        f"E_tot = {epot + ekin: .3f} eV",
        flush=True,
    )


def run_langevin_md(
    atoms,
    temperature_K,
    steps,
    zmax,
    k,
    timestep_fs=0.5,
    friction_fs=0.01,
    log_interval=20,
    traj=None,
):
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_fs / units.fs,
    )

    dyn.attach(
        wall_hook,
        interval=1,
        atoms=atoms,
        zmax=zmax,
        k=k,
    )

    dyn.attach(log_md_status, interval=log_interval, atoms=atoms, dyn=dyn)

    if traj is not None:
        dyn.attach(traj.write, interval=log_interval)

    dyn.run(steps)
    return dyn.atoms
