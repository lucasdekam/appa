import click

from appa.cli.build import build
from appa.cli.equilibrate import equilibrate
from appa.cli.lammps import lammps
from appa.cli.select import select
from appa.cli.plumed import plumed
from appa.cli.vasp import vasp
from appa.cli.convert import convert


@click.group()
def appa():
    """appa: tools for MLMD of electrochemical interfaces. Yip yip!"""
    pass


appa.add_command(build)
appa.add_command(equilibrate)
appa.add_command(lammps)
appa.add_command(select)
appa.add_command(plumed)
appa.add_command(vasp)
appa.add_command(convert)

if __name__ == "__main__":
    appa()
