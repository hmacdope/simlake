"""Benchmarks on large trajectories: ingest, storage size and read throughput.

Run:  python benchmarks/large_bench.py [dataset ...]

Datasets (see benchmarks/make_trajectories.py and MDAnalysisData):

  yiip        YiiP membrane transporter, 111,815 atoms x 9,001 frames (real, MDAnalysisData)
  noise_100k  100,000 atoms x 5,000 frames, numpy random walk (benchmarks/make_trajectories.py)
  noise_1m    1,000,000 atoms x 500 frames, numpy random walk
  adk_x1      AdK in water, 47,681 atoms x 10,000 frames (earlier AdK-based generator)
  adk_x8      AdK in water tiled 2x2x2, 381,448 atoms x 2,000 frames (earlier generator)

Noise trajectories are fine for read speed but a worst case for compression (no local
structure); YiiP gives the realistic compression numbers.

Files live under data/. Every read is one full pass through the trajectory with a warm
OS page cache (the XTC files fit in memory), so this measures decoding and the read
stack, not disk. Results are printed and saved to data/large_bench.json.
"""

import json
import os
import shutil
import sys
import time
import warnings

import dask
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

import simlake as sl

warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DATASETS = {
    "yiip": ("mdanalysisdata/yiip_equilibrium/YiiP_system.pdb",
             "mdanalysisdata/yiip_equilibrium/YiiP_system_90ns_center.xtc"),
    "noise_100k": ("synthetic/noise_100000a_5000f.gro", "synthetic/noise_100000a_5000f.xtc"),
    "noise_1m": ("synthetic/noise_1000000a_500f.gro", "synthetic/noise_1000000a_500f.xtc"),
    "adk_x1": ("synthetic/adk_water_x1_10000f.gro", "synthetic/adk_water_x1_10000f.xtc"),
    "adk_x8": ("synthetic/adk_water_x8_2000f.gro", "synthetic/adk_water_x8_2000f.xtc"),
}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def iterate(u):
    """One full pass reading every frame's positions; returns frames/s."""
    def run():
        for _ in u.trajectory:
            u.atoms.positions
        return u.trajectory.n_frames
    n, t = timed(run)
    return n / t


def bench(name):
    top, traj = (os.path.join(DATA, p) for p in DATASETS[name])
    lake_path = os.path.join(DATA, "lakes", name)
    shutil.rmtree(lake_path, ignore_errors=True)
    os.makedirs(os.path.dirname(lake_path), exist_ok=True)

    orig = mda.Universe(top, traj)
    n_atoms, n_frames = orig.atoms.n_atoms, orig.trajectory.n_frames
    r = {"dataset": name, "n_atoms": n_atoms, "n_frames": n_frames, "xtc_bytes": os.path.getsize(traj)}
    print(f"\n== {name}: {n_atoms:,d} atoms x {n_frames:,d} frames, XTC {r['xtc_bytes'] / 2**30:.2f} GiB")

    lake = sl.Lake.create(lake_path)
    lake.create_system(name, top)
    groups = lake.system(name).groups
    r["groups"] = {g["name"]: g["stop"] - g["start"] for g in groups}
    print("   groups:", r["groups"])

    _, r["register_s"] = timed(lambda: lake.register(name, traj))
    _, r["copy_in_s"] = timed(lambda: lake.add_replica(name, mda.Universe(top, traj)))
    stored = lake.system(name).stored_bytes(1)
    r["native_bytes"] = sum(stored.values())
    r["native_bytes_by_group"] = stored
    print(f"   register in place {r['register_s']:.1f} s | copy in {r['copy_in_s']:.0f} s, "
          f"{r['native_bytes'] / 2**30:.2f} GiB ({r['native_bytes'] / r['xtc_bytes']:.2f}x XTC)")

    protein = "protein" if "protein" in r["groups"] else groups[0]["name"]
    reads = {
        "MDAnalysis XTC": lambda: iterate(mda.Universe(top, traj)),
        "lake in place (XTC)": lambda: iterate(lake.universe(name, 0)),
        "lake native": lambda: iterate(lake.universe(name, 1)),
        f"MDAnalysis XTC, {protein} atoms": lambda: iterate_selection(top, traj, protein),
        f"lake in place, group {protein}": lambda: iterate(lake.universe(name, 0, groups=protein)),
        f"lake native, group {protein}": lambda: iterate(lake.universe(name, 1, groups=protein)),
    }
    r["frames_per_s"] = {}
    for label, fn in reads.items():
        r["frames_per_s"][label] = fps = fn()
        print(f"   {label:40s} {fps:9,.0f} frames/s")

    ds = lake.open_dataset(name)
    for rep, label in ((0, "in place"), (1, "native")):
        sel = ds.isel(replica=[rep]).md.select("protein and backbone" if protein == "protein" else "all")
        with dask.config.set(scheduler="threads"):
            _, t = timed(lambda: sel.md.rmsd().compute())
        r["frames_per_s"][f"xarray+dask backbone RMSD, {label}"] = fps = n_frames / t
        print(f"   {'xarray+dask backbone RMSD, ' + label:40s} {fps:9,.0f} frames/s")
    return r


def iterate_selection(top, traj, group):
    """MDAnalysis reading the whole XTC but only touching one group's atoms."""
    u = mda.Universe(top, traj)
    ag = u.select_atoms(group)

    def run():
        for _ in u.trajectory:
            ag.positions
        return u.trajectory.n_frames
    n, t = timed(run)
    return n / t


def main(names):
    results = []
    for name in names:
        top, traj = (os.path.join(DATA, p) for p in DATASETS[name])
        if not (os.path.exists(top) and os.path.exists(traj)):
            print(f"\n== {name}: missing {traj}; see the module docstring")
            continue
        results.append(bench(name))
    out = os.path.join(DATA, "large_bench.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["yiip", "noise_100k", "noise_1m"])
