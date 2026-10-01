---
name: appa
description: Set up and drive MLMD of electrochemical interfaces with the appa CLI — build electrode/water slabs, write LAMMPS inputs for MACE/GRACE/NequIP/metatomic models (including charge-conditioned GRACE with a total charge and work function), generate PLUMED and VASP inputs, and convert trajectories. Use whenever the task is to set up, launch or prepare a LAMMPS or MLIP simulation of a metal/water or electrode/electrolyte interface, write an input.lmp or system.data, build a slab-plus-water starting structure, sweep the surface charge, or label configurations with VASP — and use it INSTEAD of hand-writing a LAMMPS input file for these systems.
user-invocable: true
allowed-tools:
  - Read
  - Bash
  - Write
  - Edit
---

# appa — MLMD of electrochemical interfaces

Source: `~/Repositories/appa` (Lucas's own package, docs at
<https://lucasdekam.github.io/appa/>). Install editable into whichever
interpreter is doing the setup:

```bash
~/miniforge3/bin/python -m pip install -e ~/Repositories/appa
```

Everything is behind one entry point, `appa`, with `--help` on every command and
subcommand. **`--help` is the contract — read it instead of the source.**

## Rule zero

> Never hand-write an `input.lmp` or a `system.data` for a metal/water or
> electrode/electrolyte system. Run `appa lammps`.

This is not a style preference. See "Species order", below — the failure mode of
a hand-written input is a run that completes normally and computes the wrong
chemistry.

## When to reach for this

Use it when the system is a **slab electrode with water or electrolyte on top**,
driven by a machine-learned potential:

- setting up or launching a LAMMPS MD run of such a system
- building a starting structure (slab + packed water + ions + adsorbed H)
- sweeping surface charge, temperature, or model across an array job
- writing PLUMED input for a Volmer step, or VASP input to label configurations
- compressing or converting the resulting trajectories

**Do not** use it for bulk systems with no surface, for non-LAMMPS engines, or
as a general LAMMPS input generator — `appa` hard-codes an NVT slab workflow
(`units metal`, `atom_style atomic`, frozen bottom layers, Nosé–Hoover on the
rest). Anything outside that shape is a hand-written input, and that is fine.

For *analysing* what comes out, see the `watanalysis` skill; for getting it onto
a cluster, `hpc` and `lrz`.

## Why this saves tokens

Three distinct savings, largest first:

1. **One line instead of forty.** A correct LAMMPS input for one of these
   systems is ~35 lines in which the `pair_style` keywords, the frozen-atom
   group, the `compute`/`thermo_style` plumbing and the species mapping all have
   to be exactly right. Emitting that by hand costs on the order of 1200 output
   tokens *per run*, and a charge or temperature sweep multiplies it. The CLI
   call is ~40 tokens and emits the identical, already-debugged file every time.
2. **No source reading.** The commands are stable and self-describing. Run
   `appa lammps --help` once if unsure; do not open `appa/lammps.py` to find out
   what a flag does, and do not re-derive the pair-style syntax for an
   architecture from the LAMMPS docs.
3. **No echoing.** Write the inputs straight into the run directory and report
   what was written. Do not `cat input.lmp` back into the conversation to admire
   it — it is generated, deterministic, and re-readable on demand. Check a
   generated file only when something has actually failed.

The same three apply to `appa build`, `appa vasp input` and `appa convert xtc`.

## The pipeline

| stage | command | writes |
|---|---|---|
| build a slab + water (+ ions, + adsorbed H) | `appa build` | `interface.xyz` |
| pre-equilibrate the packed water | `appa equilibrate STRUCTURE MODEL` | `equilibrated.xyz` |
| write the LAMMPS run | `appa lammps INITIAL` | `input.lmp`, `system.data` |
| (optional) enhanced sampling | `appa plumed volmer` | `plumed.dat` |
| label configurations with DFT | `appa vasp input` / `appa vasp collect` | VASP dirs / `collected.xyz` |
| prepare training data | `appa convert xyz2grace`, `convert extract-isolated` | GRACE DataFrame |
| shrink trajectories | `appa convert xtc` | `.xtc` |

`appa build` needs the **`packmol` binary on PATH** (through `mdapackmol`), and
drops `tmp.pdb`, `waterbox.xyz` and `packmol.stdout` into the current directory
— run it in a scratch or run directory, not in a repo root.

Packmol's water is structurally terrible. `appa equilibrate` exists because of
that: short ASE MD with a harmonic wall that keeps water out of the vacuum gap.
Skipping it and starting LAMMPS straight from `appa build` output usually blows
up or wastes the first few ps.

## `appa lammps` — the command you will use most

```bash
appa lammps initial.xyz --architecture grace --model ~/train/seed/1/final_model --steps 20000
```

Options: `--architecture` and `--model` (both required), `--steps`,
`--temperature` (K), `--timestep` (**ps**, default 0.0005 = 0.5 fs),
`--dump-freq`, `--plumed-file`, `--charge`, `--padding`, `--boundary`,
`--thermostat {nose-hoover,csvr}`, `--damping` (ps), `--wall-distance`,
`--wall-species`, `--wall-k`, `--surface-species`. It writes `input.lmp`
and `system.data` into the **current working directory** — `cd` to the run
directory first; there is no `-o`.

For more than one run, drive the Python API instead of shelling out in a loop
(see "Sweeps" below).

### Architectures, and how each one is launched

| `--architecture` | `pair_style` written | how to run LAMMPS |
|---|---|---|
| `grace` | `grace pad_verbose` | plain `srun .../lmp -in input.lmp` |
| `mace-mliap` | `mliap unified <model> 0` | needs Kokkos flags, and the model converted first |
| `nequip`, `allegro` | `nequip` / `allegro` | plain, unless built with Kokkos |
| `mtt` | `metatomic/kk` | Kokkos |

MACE needs two extra things, and **the conversion must happen on a GPU node**,
so put it in the job script rather than running it at submit time:

```bash
mace_create_lammps_model path/to/mymace.model --format=mliap
srun .../lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lmp
```

### Species order — the thing that silently ruins runs

`appa` derives the LAMMPS type order from `np.unique(chemical_symbols)`, which is
**alphabetical**, and uses that same list for both the `specorder` of
`system.data` and the trailing symbols of `pair_coeff`. They therefore cannot
disagree. A Pt/water cell gives

```
pair_coeff * * final_model H O Pt      →  type 1 = H, type 2 = O, type 3 = Pt
```

Two consequences worth holding onto:

- **Never edit one of the two files by hand.** Swapping the `pair_coeff` symbols
  or regenerating `system.data` with a different `specorder` makes LAMMPS run
  happily while feeding the model the wrong elements. Nothing in the log says so;
  the energies just are not the model's energies. Regenerate both together.
- Downstream tools that index LAMMPS types need this mapping: MDAnalysis
  selections on a `system.data` are `"type 2"` for oxygen in the example above,
  and PLUMED atom ids are 1-based in the same order.

### Fixed atoms come from the structure file

`appa lammps` reads the `FixAtoms` constraint off the input `.xyz` and freezes
exactly those atoms (`setforce 0`, zero velocity, thermostat on the rest).
`appa build --fix-layers 2` puts it there. If a conversion step drops the
constraint — many round-trips through non-extxyz formats do — the bottom layers
silently become mobile and the slab drifts. Check that the echoed
`Fixed atom indices:` list is non-empty and the right length before submitting.

### Charged interfaces: charge-conditioned GRACE

A charge-conditioned (FiLM) GRACE model takes the system's total charge as an
input and exports the work function dE/dq alongside energy and forces. Running
that in LAMMPS rather than through ASE needs the
[charge-conditioned fork](https://github.com/lucasdekam/lammps/tree/grace), which
adds a `q` keyword to `pair_style grace` and publishes dE/dq as the pair style's
global extra quantity.

```bash
appa lammps initial.xyz --architecture grace --model .../final_model --charge -0.5
```

Charge is in electrons, **negative for excess electrons** (the GPAW-SJM and
training-data convention). That writes

```
pair_style grace pad_verbose q -0.5
pair_coeff  * * .../final_model H O Pt
compute     workfunc all pair grace
```

and appends `c_workfunc[1]` to `thermo_style` plus a `work_function` column to
`energy.log`, so the work function arrives in the files you already collect —
there is nothing extra to fetch.

Four things to know:

- `--charge` with a model that is **not** charge-conditioned is a hard error in
  LAMMPS, by design: a silently ignored charge would give the q = 0 answer
  wearing a charge label. `--charge` with a non-GRACE `--architecture` is an
  error in `appa`.
- `--charge 0` is not the same as omitting it: it still asserts a
  charge-conditioned model and still logs dE/dq. That is the right way to get the
  neutral point of a sweep.
- The reported dE/dq is that of the **padded** system, and unlike the padded
  energy that is not a constant offset. Use `--padding 0` for a single point or
  a rerun where the absolute work function must be exact. Keep the default for
  MD — `padding 0` retraces the TensorFlow graph whenever the neighbor count
  changes, which is every step.
- The stock GRACE build is enough for uncharged runs. Only the fork understands
  `q`, and it is deliberately built **without Kokkos**: the Kokkos pair styles
  read a `.npz` from `grace_utils export_kokkos`, which rejects FiLM's
  instruction graph.

### Boundary, thermostat and wall

- **The boundary follows `atoms.pbc`**: `p` where periodic, `f` where not, so an
  extxyz with `pbc="T T F"` gives `boundary p p f`; `--boundary` overrides. A
  script that sets `atoms.pbc = True` before building keeps `p p p`. With `f`
  an escaping atom is lost and LAMMPS stops instead of wrapping.
- `--thermostat csvr` writes `fix nve` + `fix temp/csvr` on the mobile group
  (EXTRA-FIX package). `--damping` is in ps for either thermostat.
- `--wall-distance D` puts a one-sided harmonic wall D Å above the top frozen
  (electrode) atom, on O by default, `F = -k (z - z0)` above it, as in
  `appa equilibrate` and the RAZOR MD. It is `fix wall/harmonic zhi` placed
  5 Å beyond the plane with `eps = k/2`. **It needs a non-periodic z**; appa
  raises on `p p p` rather than let LAMMPS fail at run time.

## Sweeps and array jobs

For several runs, build the simulations in Python and write one directory each —
this is what `write_array_job_inputs` is for, and the folder numbering matches a
SLURM array index.

```python
from ase.io import read
from appa.lammps import AtomisticSimulation, write_array_job_inputs

atoms = read("equilibrated.xyz")
sims = []
for q in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    sim = AtomisticSimulation(atoms)
    sim.set_potential("final_model", architecture="grace", total_charge=q)
    sim.set_molecular_dynamics(temperature=300, timestep=0.0005)
    sim.set_log()
    sim.set_dump()
    sim.set_run(n_steps=20000, restart_freq=5000)
    sims.append(sim)

write_array_job_inputs("runs", sims, folder_name="q")   # runs/q_000 ... runs/q_004
```

Call order matters: `set_potential` before `set_log`, because `set_log` only adds
the work-function column if a charge was set. `set_rerun(dump)` replaces
`set_molecular_dynamics` + `set_run` for re-evaluating an existing trajectory
with a different model or charge — the natural way to get dE/dq along a
trajectory that was run neutral.

`examples/lammps-array-job` and the job script in `docs/dynamics.md` are working
SLURM templates; copy them rather than writing a submission script from scratch.
The pattern in both: stage the run onto node-local scratch, `appa lammps` *inside*
the job so the input is generated next to the data, run, copy back.

## `appa select` — picking frames to label

```bash
appa select -d frames_dir/ --size 400 -o selected.xyz [-s O -s H -s Pt] [--bw 0.065]
```

QUESTS (`quests`) maximum set coverage: per-atom descriptors (k = 32, 5 Å),
a per-frame entropy, then greedy `quests.compression.fps.msc`, which picks the
frame whose most novel environment plus entropy is largest. `-s` filters out
whole frames by species; it does not select atoms.

Three things it does not do yet, all worked out in lorem-q-work's
`datasets/razor_additional/single_q-1/build_frames.py`. Move them here when
that builder moves to a central place:

- **Interfacial rows only.** Keep the descriptor rows of the atoms that
  matter, e.g. O within 4 Å of the top metal layer. Each row still sees its
  full environment, and the kernel matrices shrink ~13x: 1150 candidates
  against 960 seed frames select 400 in ~30 s.
- **Seeding with existing labels.** `msc` has no seed argument. Start its
  kernel accumulator from `kernel_sum(candidates, labelled)` instead of 0, so
  "novel" means novel against the dataset, not only against the other picks
  (`seeded_msc` there). Without it, a selection happily re-picks what is
  already labelled.
- **Charge-aware novelty (planned).** The kernel is Gaussian in descriptor
  distance, so appending a column `q * h / sigma_q` to every row gives exactly
  the product kernel `K_struct * exp(-dq^2 / 2 sigma_q^2)`. A structure then
  counts as redundant only if a similar one is labelled at a similar charge.
  `sigma_q ~ 0.25 e`, the stencil spacing, gives overlaps of 0.61 / 0.14 / ~0
  at dq = 0.25 / 0.5 / 1 e. The frame entropy is unchanged (q is constant in a
  frame). This replaces hand-set per-charge quotas in multi-charge selections.
  If candidates are copied onto other charges to choose the labelling charge,
  keep them within ~0.25 e of the charge their MD ran at: labelling far from
  it pins the work function (razor_additional cycle 2).

Pre-thin trajectories in time before computing descriptors (one frame per ps
is plenty). Coverage saturates quickly: in the q = -1 selection, each pick's
most novel environment already had ~10 similar labelled ones by pick 50 and
~55 by pick 370, so the novelty curve is a good guide to how many frames are
worth labelling.

## Gotchas

- **`--timestep` is in picoseconds**, not fs. 0.0005 is 0.5 fs. Passing `0.5`
  gives a 500 fs timestep and a trajectory that explodes immediately.
- **`boundary p p p`** always, so the vacuum gap has to be large enough that the
  slab does not see its own image. `appa build --d-vacuum 20` is the default for
  a reason; do not trim it to save atoms.
- **`energy.log` is written every single step** (`fix print 1`), independent of
  `--dump-freq`. On a long run it gets large; it is also the highest-resolution
  record you have, so thin it when copying back rather than turning it off.
- **`appa convert xtc` needs `system.data` beside the trajectory** to be
  interpretable later, and keeps only positions at 5 decimals. Convert for
  archiving and analysis, not before you are done with velocities.
- The CLI imports every subcommand at startup, so a broken optional dependency
  (`quests`/`numba` vs the installed NumPy is the recurring one) takes down
  *all* of `appa`, including commands that do not use it. If `appa --help`
  itself traceback, fix the environment — nothing is wrong with the command you
  were trying to run.
