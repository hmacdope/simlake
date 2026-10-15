"""Throughput harness for the in-place XTC reader.

Run:  python benchmarks/xtc_bench.py [n_frames]

Measures, on a synthetic XTC (the MDAnalysisTests adk-in-water frames repeated with
0.05 A jitter, 47,681 atoms):

* decode, 1 thread    simlake.xtc.decode_frame on bytes already in memory
* MDAnalysis XTC      MDAnalysis' C reader (file I/O + decode), the baseline
* lake, all atoms     a virtual:xtc replica read through MDAnalysis (reader + zarr + Icechunk)
* lake, xarray        the same replica as a dask array, computed with the threaded scheduler
"""

import os
import sys
import tempfile
import time
import warnings

import dask
import numpy as np
import MDAnalysis as mda
from MDAnalysisTests.datafiles import TPR, XTC

import simlake as sl
from simlake import xtc

warnings.filterwarnings("ignore")


def make_xtc(path, n_frames, jitter=0.05, seed=0):
    u = mda.Universe(TPR, XTC)
    base = [ts.positions.copy() for ts in u.trajectory]
    dims = u.trajectory[0].dimensions.copy()
    rng = np.random.default_rng(seed)
    with mda.Writer(path, n_atoms=u.atoms.n_atoms) as w:
        for i in range(n_frames):
            u.trajectory.ts.positions = base[i % len(base)] + rng.normal(0, jitter, base[0].shape).astype(np.float32)
            u.trajectory.ts.dimensions = dims
            u.trajectory.ts.time = 10.0 * i
            w.write(u.atoms)
    return path


def best_of(fn, repeats=3):
    return max(fn() for _ in range(repeats))


def run(n_frames=500, workdir=None):
    workdir = workdir or tempfile.mkdtemp()
    path = os.path.join(workdir, "bench.xtc")
    if not os.path.exists(path):
        make_xtc(path, n_frames)
    with open(path, "rb") as fh:
        raw = fh.read()
    with open(path, "rb") as fh:
        info = xtc.scan(fh, len(raw))
    frames = [raw[o:o + n] for o, n in zip(info["offset"], info["length"])]
    xtc.decode_frame(frames[0])  # compile

    def decode():
        t0 = time.perf_counter()
        for f in frames:
            xtc.decode_frame(f)
        return len(frames) / (time.perf_counter() - t0)

    def mdanalysis():
        u = mda.Universe(TPR, path)
        t0 = time.perf_counter()
        for _ in u.trajectory:
            u.atoms.positions  # same work as the lake loop: analyses read positions
        return u.trajectory.n_frames / (time.perf_counter() - t0)

    lake_path = os.path.join(workdir, "lake")
    if not os.path.exists(lake_path):
        lake = sl.Lake.create(lake_path)
        lake.create_system("adk", TPR)
        lake.register("adk", path)
    lake = sl.Lake.open(lake_path)

    def lake_mda():
        u = lake.universe("adk", 0)
        u.trajectory.close()
        t0 = time.perf_counter()
        for _ in u.trajectory:
            u.atoms.positions
        return u.trajectory.n_frames / (time.perf_counter() - t0)

    def lake_xarray():
        pos = lake.open_dataset("adk").position
        t0 = time.perf_counter()
        with dask.config.set(scheduler="threads"):
            pos.sum(["atom", "spatial"]).compute()
        return pos.sizes["frame"] / (time.perf_counter() - t0)

    results = {
        "decode, 1 thread": best_of(decode),
        "MDAnalysis XTC (C)": best_of(mdanalysis),
        "lake, all atoms (MDAnalysis)": best_of(lake_mda),
        "lake, xarray + dask threads": best_of(lake_xarray),
    }
    print(f"{len(frames)} frames x {info['n_atoms']} atoms, {len(raw) / 2**20:.0f} MB XTC; frames/s:")
    for k, v in results.items():
        print(f"  {k:32s} {v:8,.0f}")
    return results


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 500,
        workdir=os.environ.get("XTC_BENCH_DIR"))
