# simlake

A versioned data lake for molecular dynamics trajectories, built on
[Zarr v3](https://zarr.dev) and [Icechunk](https://icechunk.io). Read it with
MDAnalysis or with xarray and dask, from local disk or object storage.

- **Bring existing files.** Reference XTC and DCD files in place without copying, or copy
  trajectories in as compressed Zarr.
- **Read only what you need.** Atoms are stored in groups (protein, lipids, water, ...),
  so a protein-only analysis never reads the solvent.
- **Version everything.** Every write is a commit; tag the data behind a paper and read it
  back later.
- **Analyse at scale.** Every replica of a system is one lazy xarray dataset, ready for
  dask.

> **Status:** prototype. The on-disk conventions (v0.3) may still change.

## Install

```bash
micromamba create -n simlake -c conda-forge python=3.12 "zarr>=3.4" icechunk xarray dask \
    mdanalysis mdanalysistests pandas duckdb cython pytest
micromamba run -n simlake pip install "cast-value-rs>=0.4.2" pcodec
micromamba run -n simlake pip install -e .           # builds the Cython XTC decoder
```

Optional: `s3fs` for object storage, `moto` for the S3 tests, `bokeh` for the dask
dashboard, `jupytext matplotlib mdanalysisdata` for the example notebooks.

## Quick start

```python
import MDAnalysis as mda
import simlake as sl

lake = sl.Lake.create("kinase.lake")
lake.create_system("adk", "adk.tpr")                        # atoms grouped by molecule type

lake.register("adk", "run0.xtc")                            # in place: nothing copied
lake.add_replica("adk", mda.Universe("adk.tpr", "run1.xtc"))  # copied in, compressed

lake.query("SELECT system, replica, storage, n_frames FROM trajectories")

u = lake.universe("adk", replica=0)                         # a normal MDAnalysis Universe
protein = lake.universe("adk", replica=1, groups="protein") # reads only the protein

ds = lake.open_dataset("adk")                               # all replicas, lazy (dask)
rmsd = ds.md.select("backbone").md.rmsd().compute()

lake.tag("paper-v1")                                        # read it back with ref="paper-v1"
```

The same API works on object storage:

```python
sl.configure("s3://", region="us-west-2")                   # credentials from the environment
lake = sl.Lake.create("s3://my-bucket/lakes/kinase")
lake.register("adk", "s3://raw-bucket/adk/run0.xtc")
u = mda.Universe("simlake://s3://my-bucket/lakes/kinase?system=adk&replica=0")
```

## Examples

- [`examples/showcase_v2.ipynb`](examples/showcase_v2.ipynb): a tour of the API, ending with
  dask on a real membrane protein (YiiP, 111k atoms x 9,001 frames) and the same lake on S3.
- [`examples/showcase.ipynb`](examples/showcase.ipynb): a longer walkthrough with
  cross-checks against MDAnalysis.

The notebooks are generated from the `.py` scripts next to them:
`jupytext --to notebook --execute examples/showcase_v2.py`.

## Performance

On YiiP (111,815 atoms x 9,001 frames, 3.5 GB XTC), one full pass:

| | frames/s | vs MDAnalysis |
|---|---|---|
| MDAnalysis reading the XTC | 419 | 1.0x |
| simlake, XTC in place | 858 | 2.0x |
| simlake, native copy, protein only | 4,106 | 9.8x |

Registering the XTC in place takes 0.1 s; copying it in takes 43 s. More results, the
on-disk layout and design decisions are in [`docs/design.md`](docs/design.md).

## Development

```bash
micromamba run -n simlake python -m pytest         # tests (S3 tests need moto)
micromamba run -n simlake python benchmarks/large_bench.py
```
