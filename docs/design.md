# simlake design notes

## Layout (conventions v0.3, `simlake/conventions.py`)

A lake is one [Icechunk](https://icechunk.io) repository of Zarr v3 groups and arrays.

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

A **system** is one topology. Two trajectories can be replicas of the same system only if
atom names, residue names and residue ids match atom by atom (`topology_hash`).

Replicas are separate Zarr groups, so replicas of different lengths need no padding on
disk and writers never resize a shared array. The `(replica, frame, ...)` hypercube
exists only in the xarray view, which stacks replicas by frame index and pads with NaN.
`open_dataset` warns when replicas differ in `dt` or start time, since the same `frame`
then means a different time in each.

## Atom groups

A system's atoms are partitioned into named groups: every atom belongs to exactly one,
so frames can always be rebuilt in full. By default atoms are grouped by molecule type
(`protein`, `nucleic`, `lipids`, `water`, `ions`, and `other` for the rest). Custom
groups are MDAnalysis selections; overlaps are rejected and unclaimed atoms go into
`other`.

Each group is its own standard Zarr array, `positions/<group>`. All groups share frames
per chunk, so the xarray view joins them block for block into one `position` variable
with a single `atom` dimension. Large groups are split into equal atom blocks; shards are
sized per group, so a 4-atom ions group is one small object, not thousands. Reading one
group (`universe(..., groups=...)`, `ds.md.group(...)`, or a selection inside it) touches
only that group's array. Groups are ordered by first atom, so the combined order is
usually the topology order; `topology/storage_order` maps it back when it isn't.

**Why not one array with chunk edges on group boundaries?** That needs Zarr's
rectilinear chunk grid, which is experimental in zarr-python 3.4 and can't be sharded.
A data lake should only use standard Zarr, so other tools can read it for years. The API
would not change if the layout moves to it later.

## Reading files in place

`Lake.register` references XTC (local or object storage) and DCD (local) files without
copying coordinates. Each XTC frame is one chunk, decoded by a Cython port of the
xdrfile frame decoder (`simlake/_xtc.pyx`), checked bit-for-bit against MDAnalysis.
Time, step and box are copied into the lake at registration (a header scan).

Every reference records the source's mtime (local) or ETag (object storage); Icechunk
refuses to read a changed source, and `Lake.verify()` reports changed or missing files.

When several files are joined into one replica, overlapping frames are dropped with the
later file winning, as in MDAnalysis' `ChainReader(continuous=True)`. That covers GROMACS
restarts without `-append` (duplicated boundary frame) and restarts from older
checkpoints. Each source records `dropped_frames`.

## Parallelism

- **Prefer dask on xarray.** Dask tasks read Zarr chunks directly; the threaded
  scheduler needs no pickling or forking.
- **Icechunk is not fork-safe.** MDAnalysis' `multiprocessing`/`dask` backends hang under
  the Linux default `fork`; they work with `forkserver` or `spawn`, and the reader warns
  when it is pickled under `fork`.
- **Codecs offload decoding** to zarr's thread pool with `asyncio.to_thread`, as zarr's own
  zstd/blosc codecs do. Without it every decode runs on zarr's single event-loop thread.
- **Scaling flattens at about 8 threads** (2-3x on decode-heavy work): zarr-python
  coordinates every read on one event-loop thread. `dask.distributed` adds per-task cost;
  in-process (`processes=False`) clients give the dashboard at little cost, while process
  workers reopen the lake per task. Native dask blocks span a shard along frame to keep
  task graphs small.

## Findings

- **pcodec beats general-purpose compressors** on quantized coordinates (0.91x XTC's
  compression on AdK, against 0.55x for zstd), without XTC's frame-sequential decoding.
- **Zarr 3.4's `scale_offset` + `cast_value` codecs** quantize at the spec level, so other
  Zarr implementations can read the data. `cast_value` needs `cast-value-rs`, and NaN
  cannot pass the int cast, so quantized arrays use fill value 0.
- **The XTC decoder** went numba (717 frames/s) -> numba with a 64-bit bit reader and
  uint64 unpacking -> Cython (5,300 frames/s single-threaded, 3.6x MDAnalysis' C reader
  on decode alone).
- **Icechunk works against S3 with references across buckets.**

## Benchmarks

Full pass per read, warm page cache (decoding and read stack, not disk), 32-core machine.
`benchmarks/large_bench.py`; raw numbers in `benchmarks/large_bench_results.json`.
YiiP is real data (MDAnalysisData); the noise sets come from
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

Compression on AdK in water (`benchmarks/bench.py`):

| Codec | Compression vs XTC |
|---|---|
| float32 + zstd | 0.46x |
| int32-quantized + zstd-5 | 0.55x |
| int32-quantized + blosc bitshuffle | 0.73x |
| **int32-quantized + pcodec (default)** | **0.91x** |

Reading every atom of a native copy is no faster than XTC, and on YiiP the native copy
is 1.2x the XTC's size; both are open.

## Roadmap

- `materialize()`: convert an in-place replica into native storage in one commit.
- Velocities, forces and observables (positions, box, time and step only so far).
- Per-group compression settings (e.g. coarser water); the layout allows it.
- Aligning replicas by step or time in the xarray view (`align="step" | "time"`).
- Other in-place formats (TRR, NetCDF, H5MD), DCD from object storage, GROMACS 2023
  large-file XTC (magic 2023).
- Appending to running replicas; a branch per writer.
- Real AWS/GCS/Azure (only a local moto S3 server is tested); network benchmarks.
- Scaling past zarr-python's single event loop.
- A policy for near-identical topologies (protonation states, ion counts).
