# simlake (prototype)

A versioned data lake for molecular dynamics trajectories, built on Zarr v3 and
[Icechunk](https://icechunk.io), readable from both MDAnalysis and xarray.

This is a working prototype of the "hybrid" design: systems grouped by topology,
replicas stored natively (quantized, chunked so each atom group reads on its own) or
virtually (references into existing XTC/DCD files), one Icechunk repository per lake,
and a SQL catalog. Lakes and source files can live on local disk or S3.

**Start with the showcase** (below): a narrated, one-minute tour that ends with the
same lake on S3.

## Install

```bash
micromamba create -n simlake -c conda-forge python=3.12 "zarr>=3.4" icechunk xarray \
    mdanalysis mdanalysistests dask pyarrow pandas duckdb pytest
micromamba install -n simlake -c conda-forge numba moto s3fs   # XTC decoder; S3 tests
micromamba install -n simlake -c conda-forge jupytext nbconvert ipykernel matplotlib   # showcase
micromamba run -n simlake pip install "cast-value-rs>=0.4.2" pcodec   # not on conda-forge yet
micromamba run -n simlake pip install -e .
```

## Showcase

**[`examples/showcase_v2.ipynb`](examples/showcase_v2.ipynb)** shows the API in short
cells, one idea per cell: create a lake, add trajectories (in place or copied),
query, MDAnalysis, xarray + dask, versions, and the same lake on S3.

**[`examples/showcase.ipynb`](examples/showcase.ipynb)** is the detailed walkthrough,
with timings, cross-checks against MDAnalysis, change detection and more plots.

Both notebooks have their outputs saved, so they render on GitHub without running
anything. They use the MDAnalysisTests data (adenylate kinase in water, 47,681 atoms),
a temporary directory and a local moto S3 server.

Each notebook is generated from a [jupytext](https://jupytext.readthedocs.io) percent
script of the same name (`.py`), which also runs as plain Python. Regenerate after editing:

```bash
micromamba run -n simlake jupytext --to notebook --execute examples/showcase_v2.py
```

The numbered list below describes the walkthrough.

1. **Build a lake from existing files.** Register an XTC in place (no coordinates
   copied), and copy a second run in natively, making the protein whole on the way
   in with an MDAnalysis transformation.
2. **Find data with SQL** over the catalog.
3. **MDAnalysis.** A replica is a normal Universe. Reading one atom group (`groups="protein"`)
   skips the other groups' chunks.
4. **xarray + dask.** Build five analyses lazily: Rg, LID–CORE and NMP–CORE domain
   distances, RMSF, RMSD. Nothing is read until one `dask.compute()` runs them together
   on a thread pool. The dask RMSD matches MDAnalysis exactly. The plots show the
   raw replica's periodic-boundary artefact and pin it to the LID domain.
5. **Versioning.** Tag, add a replica, read the lake as it was at the tag.
6. **Change detection.** Touch an in-place source file: `verify()` flags it and reads are refused.
7. **The same lake on S3.** Lake in one bucket, XTC referenced in place in another,
   opened with `mda.Universe("simlake://s3://...")`, dask analysis from the cloud.

`examples/quickstart.py` is a shorter, less narrated version.

## Use

```python
import simlake as sl, MDAnalysis as mda

lake = sl.Lake.create("my.lake")

# A system is one topology, its atoms split into named groups (default: by molecule type)
lake.create_system("adk_water", "adk.tpr", metadata={"forcefield": "oplsaa"})
lake.create_system("kinase_atp", "complex.tpr",                    # or your own partition
                   groups={"protein": "protein", "ligand": "resname ATP", "water": "resname SOL"})

# Native: copy a trajectory in (topology identity is checked against the system)
lake.add_replica("adk_water", mda.Universe("adk.tpr", "run0.xtc"), metadata={"temperature": 300})

# Virtual: reference existing files in place; nothing is copied
lake.register("adk_water", "/project/adk/run3.xtc")              # XTC, decoded on read
lake.register("adk_water", "s3://raw-bucket/adk/run4.xtc")       # ... also from object storage
lake.create_system("adk_vacuum", "adk.psf")
lake.register("adk_vacuum", "/project/adk/vac.dcd")              # DCD (local)

# Several writes as one atomic commit
with lake.transaction("replicas 1-2") as tx:
    s = tx.system("adk_water")
    s.add_replica(mda.Universe("adk.tpr", "run1.xtc"))
    s.add_replica(mda.Universe("adk.tpr", "run2.xtc"))
lake.tag("paper-v1")

# Find
lake.query("SELECT system, replica, n_frames FROM trajectories WHERE temperature = 300")

# MDAnalysis: a normal Universe (topology + coordinates from the lake)
u = lake.universe("adk_water", 0)                      # or mda.Universe(lake.uri("adk_water", 0))
p = lake.universe("adk_water", 0, groups="protein")    # reads only the protein's chunks
old = lake.universe("adk_water", 0, ref="paper-v1")    # any tag, branch or snapshot id

# xarray: every replica at once, lazy (dask), padded along frame
ds = lake.open_dataset("adk_water")
prot = ds.md.group("protein").position                # (replica, frame, atom, spatial)
ca = ds.md.select("name CA").position                 # any topology-based MDAnalysis selection
rmsd = ds.md.select("backbone").md.rmsd().compute()  # dask; MDAnalysis' QCP kernel per frame

lake.verify()    # are the files behind virtual replicas still there and unchanged?
lake.history()   # commits
```

### On S3 (or another object store)

```python
sl.configure("s3://", region="us-west-2")      # or endpoint_url=..., access_key_id=... ; default: env
lake = sl.Lake.create("s3://my-bucket/lakes/adk")
lake.register("adk_water", "s3://raw-bucket/adk/run0.xtc")    # references across buckets
u = mda.Universe("simlake://s3://my-bucket/lakes/adk?system=adk_water&replica=0")
```

Credentials never go in URIs. `configure()` settings travel with pickled readers, so
workers started with `spawn`/`forkserver` can reach the same endpoint. GCS and Azure
are wired up the same way in `simlake/locations.py` but untested.

### Parallel analysis: prefer dask

`ds.md.rmsd()` and any xarray expression run as dask graphs that read Zarr chunks
directly. The threaded scheduler needs no pickling and no forking. MDAnalysis'
`multiprocessing`/`dask` backends also work, but only with `forkserver` or `spawn`
(see Findings).

`examples/quickstart.py` runs all of this on the MDAnalysisTests data.

## Layout (conventions v0.3, see `simlake/conventions.py`)

```
systems/<name>/                 topology_hash, n_atoms, groups, metadata
  topology/                     names resnames resids segids types masses charges elements bonds
                                storage_order; source/ (original topology files)
  replicas/<i>/                 storage, n_frames, precision, metadata, [sources]
    time (frame) f8 ps | step (frame) i8 | box (frame,3,3) f4 A
    native:      positions/<group> (frame, atom, spatial), one array per atom group;
                 int32-quantized (0.01 A = XTC precision), pcodec; ~4 MB chunks,
                 shards sized per group (~128 MB; small groups: one long shard)
    virtual:xtc: position (frame, atom, spatial), nm -> one chunk per XTC frame, decoded
                 by the simlake.xtc_frame codec; time/step/box copied in
    virtual:dcd: position_sa (frame, spatial, atom) -> byte ranges in the original DCD
```

### Atom groups

A system's atoms are partitioned into named **groups**: every atom belongs to exactly
one group, so frames can always be rebuilt in full. By default atoms are grouped by
molecule type (`protein`, `nucleic`, `lipids`, `water`, `ions`, and `other` for the
rest). Pass `groups={name: selection}` for your own partition; overlapping groups are
rejected, and atoms no group claims go into `other`.

On disk, each group is its own standard Zarr array, `positions/<group>`. All groups use
the same frames per chunk, so the xarray view joins them block for block into one
`position` variable with a single `atom` dimension. Large groups are split into equal
atom blocks; shards are sized per group, so a 4-atom ions group is one small object,
not thousands. Reading one group, through `universe(..., groups=...)`,
`ds.md.group(...)` or a selection inside it, touches only that group's array. Groups
are ordered by first atom, so in the usual case the combined order is the topology
order. `topology/storage_order` maps it back when it isn't.

Why not one array with chunk edges on group boundaries? That needs Zarr's rectilinear
chunk grid, which is still experimental in zarr-python 3.4 and can't be sharded. A data
lake should only use standard Zarr, so other tools can read it for years. The API would
not change if the layout ever moves to it.

Replicas are separate Zarr groups, so replicas of different lengths need no padding
on disk, and writers don't need to resize a shared array. The `(replica, frame, ...)`
hypercube exists only in the xarray view, which pads with NaN.

## Results (MDAnalysisTests data, local SSD, 32 cores)

`benchmarks/bench.py`. The test trajectories are small, so the benchmark adds a
200-frame synthetic XTC: the 10 test frames repeated with 0.05 A of jitter.

**Compression** (adk in water, 47,681 atoms):

| Codec | Size vs XTC |
|---|---|
| float32 + zstd | 0.46x |
| int32-quantized + zstd-5 | 0.55x |
| int32-quantized + blosc bitshuffle | 0.73x |
| **int32-quantized + pcodec (default)** | **0.91x** |

By group: protein 6.2%, water 93.8% of stored coordinate bytes (`System.stored_bytes()`).

**Sequential reads through MDAnalysis**, 200-frame synthetic (47,681 atoms), frames/s:

| | frames/s | vs XTC |
|---|---|---|
| XTC original (MDAnalysis, C decoder) | 1,000 | 1.0x |
| lake native, all atoms | 1,025 | 1.0x |
| lake native, `groups="protein"` (3,341 atoms) | 8,638 | 8.6x |
| lake virtual XTC (numba decoder, frames decoded in parallel) | 1,260 | 1.3x |

Native reads of all atoms got faster with group-aligned chunks (from 0.87x XTC
before): water is split into several atom blocks, which decode in parallel.

On the 98-frame, 3,341-atom DCD: original 20,400, virtual 2,100, native copy 11,000.

> An earlier version of this README reported higher native and protein-only numbers.
> They were wrong: the reader caches its last decoded block, the test trajectories fit
> in one block, and the benchmark never cleared it between passes. The benchmark now
> clears the cache before each pass.

### Large trajectories (`benchmarks/large_bench.py`)

One full pass per read; warm page cache, so this measures decoding and the read stack,
not disk. YiiP is real data (MDAnalysisData); the noise sets come from
`simlake.testing.noise_trajectory` (realistic for speed, worst case for compression).

| frames/s | YiiP 111,815 atoms x 9,001 | noise 100k x 5,000 | noise 1M x 500 |
|---|---|---|---|
| MDAnalysis XTC | 419 | 384 | 38 |
| lake, XTC in place | **858** (2.0x) | **916** (2.4x) | **97** (2.6x) |
| lake, native | 425 (1.0x) | 446 (1.2x) | 51 (1.3x) |
| MDAnalysis XTC, protein atoms only | 581 | 495 | 49 |
| lake in place, protein group | 1,965 | 1,930 | 221 |
| lake native, protein group | **4,106** (9.8x) | **4,516** (11.8x) | **636** (16.7x) |
| xarray+dask backbone RMSD, native | 4,639 | 4,448 | 679 |

| | YiiP | noise 100k | noise 1M |
|---|---|---|---|
| register in place | 0.1 s | < 0.1 s | < 0.1 s |
| copy in (native) | 43 s | 23 s | 25 s |
| native size vs XTC | 1.21x | 0.95x | 0.96x |

YiiP's default groups: protein 8,814, lipids 30,096, water 72,717, ions 132, and
`other` 56 (its zinc sites, modelled as a zinc plus six dummy atoms).

In-place XTC reads are about 2x MDAnalysis' XTC reader everywhere; reading one group
from a native copy is 10-17x. Reading every atom of a native copy is no faster than
XTC, and on real data the native copy is 1.2x the XTC's size, which is worth a look.

Dask on the threaded scheduler scales to about 8 threads (2-3x), then flattens:
zarr-python coordinates every read on one event-loop thread. `dask.distributed`
processes were slower here, because each task reopens the lake.

## Findings from building it

- **Icechunk is not fork-safe.** MDAnalysis `multiprocessing`/`dask` backends hang
  under the Linux default `fork`. They work under `forkserver` or `spawn`, and the
  reader warns when it is pickled under `fork`.
- **Change detection comes free.** Each virtual chunk records the source's mtime, and
  Icechunk refuses to read it if the file changed. `Lake.verify()` reports these.
- **pcodec beats general-purpose compressors** on quantized coordinates by a wide
  margin. It is close to XTC without XTC's frame-sequential decoding.
- **XTC can be read in place.** A numba port of xdrfile's frame decoder is bit-identical
  to MDAnalysis on every XTC in MDAnalysisTests. Single-threaded it is about half
  as fast as the C reader. Zarr decodes a block of frames on several threads, which
  makes it faster than MDAnalysis overall.
- **Icechunk works against S3, including references across buckets.** It detects
  changed sources by ETag, the way it uses mtime for local files.
- **Rectilinear chunk grids were tried and set aside.** One array with chunk edges on
  group boundaries works in Zarr 3.4, but only behind an experimental flag and without
  sharding (it allows rectilinear shards only around regular inner chunks). One array
  per group gives the same reads using standard Zarr, with sharding.
- **Zarr 3.4's spec-level `scale_offset` + `cast_value` codecs** handle quantization,
  so other Zarr implementations can read it. `cast_value` needs `cast-value-rs`, and
  NaN can't pass through the int cast, so quantized arrays use fill value 0.

## Not done yet

- Other virtual formats (TRR, NetCDF, H5MD), DCD from object storage, XTC with
  GROMACS 2023 large-file headers (magic 2023).
- Velocities, forces and observables. Positions, box, time and step only so far.
- Per-group compression settings (e.g. coarser precision for water). The layout allows
  it now (one array per group); the API doesn't expose it yet.
- Appending to a running replica, branches per writer, `materialize()` of virtual replicas.
- Real AWS/GCS/Azure: only a local moto S3 server is tested. No network-latency benchmarks yet.
- Native reads use only a few cores when inner chunks are large relative to the file.
- Larger benchmarks (e.g. YiiP, 111k atoms x 9000 frames), random access, cloud reads.
- Topology identity is atom names + residue names + residue ids. Policy for
  near-identical topologies (protonation states, ion counts) is open.

## Tests

```bash
micromamba run -n simlake python -m pytest
```
