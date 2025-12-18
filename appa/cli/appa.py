import click

from appa.cli.build import build
from appa.cli.equilibrate import equilibrate
from appa.cli.lammps import lammps


@click.group()
def appa():
    """appa: tools for MLMD of electrochemical interfaces. Yip yip!"""
    pass


appa.add_command(build)
appa.add_command(equilibrate)
appa.add_command(lammps)

if __name__ == "__main__":
    appa()
