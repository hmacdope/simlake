"""Compression and read-speed benchmarks on the MDAnalysisTests trajectories.

Run:  python benchmarks/bench.py [workdir]

The test trajectories are small (10 and 98 frames), so throughput numbers are
indicative only. Compression ratios per frame are representative.
"""

import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
import zarr
from zarr import codecs as zc
from zarr.storage import MemoryStore
import MDAnalysis as mda
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC

import simlake as sl
from simlake.conventions import DEFAULT_PRECISION

warnings.filterwarnings("ignore", category=DeprecationWarning)


def quantize(precision=DEFAULT_PRECISION):
    return [zc.ScaleOffset(offset=0, scale=1.0 / precision), zc.CastValue(data_type="int32")]


def stored_bytes(data, *, filters=(), compressors=(), serializer="auto"):
    arr = zarr.create_array(
        MemoryStore(), shape=data.shape, chunks=data.shape, dtype=data.dtype,
        filters=list(filters), compressors=list(compressors), serializer=serializer, fill_value=0,
    )
    arr[:] = data
    return arr.nbytes_stored()


def compression(positions, xtc_bytes):
    fsa = np.ascontiguousarray(positions.transpose(0, 2, 1))  # (frame, spatial, atom)
    bitshuffle = zc.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    variants = [
        ("float32, zstd-5 (no quantization)", positions, [], [zc.ZstdCodec(level=5)], "auto"),
        ("quantized int32, zstd-5", positions, quantize(), [zc.ZstdCodec(level=5)], "auto"),
        ("quantized int32, zstd-19", positions, quantize(), [zc.ZstdCodec(level=19)], "auto"),
        ("quantized int32, blosc-zstd bitshuffle", positions, quantize(), [bitshuffle], "auto"),
        ("quantized, (frame,spatial,atom) + delta, bitshuffle", fsa,
         quantize() + [zc.Delta(dtype="<i4")], [bitshuffle], "auto"),
        ("quantized, (frame,spatial,atom) + pcodec", fsa, quantize(), [], zc.PCodec(level=8)),
        ("quantized, (frame,atom,spatial) + pcodec  [simlake default]", positions, quantize(), [], zc.PCodec(level=8)),
    ]
    raw = positions.nbytes
    print(f"\nCompression: adk in water, {positions.shape[1]} atoms x {positions.shape[0]} frames")
    print(f"  {'variant':55s} {'bytes':>11s} {'vs raw':>7s} {'vs XTC':>7s}")
    print(f"  {'XTC file (reference)':55s} {xtc_bytes:11,d} {raw / xtc_bytes:6.2f}x {1.0:6.2f}x")
    for name, data, filters, compressors, serializer in variants:
        try:
            n = stored_bytes(data, filters=filters, compressors=compressors, serializer=serializer)
        except Exception as exc:  # e.g. optional codec backend missing
            print(f"  {name:55s} skipped: {type(exc).__name__}: {exc}")
            continue
        print(f"  {name:55s} {n:11,d} {raw / n:6.2f}x {xtc_bytes / n:6.2f}x")


def iterate(u, passes):
    t0 = time.perf_counter()
    n = 0
    for _ in range(passes):
        # The simlake reader keeps the decoded block it last read. These test
        # trajectories fit in one block, so clear it or later passes never decode.
        if hasattr(u.trajectory, "_blocks"):
            u.trajectory._blocks = {}
        for _ in u.trajectory:
            u.atoms.positions  # touch coordinates
            n += 1
    return n / (time.perf_counter() - t0)


def make_long_xtc(path, n_frames=200, jitter=0.05, seed=0):
    """A longer XTC for throughput: the 10 test frames repeated with small jitter (A)."""
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


def throughput(lake, dcd_dir):
    print("\nSequential read throughput (local disk, warm cache, decode caches cleared), frames/s")
    cases = [
        ("XTC original (MDAnalysis XTCReader)", lambda: mda.Universe(TPR, XTC), 20),
        ("lake native, all 47,681 atoms", lambda: lake.universe("adk_water", 0), 20),
        ("lake native, protein group (3,341 atoms)", lambda: lake.universe("adk_water", 0, groups="protein"), 20),
        ("lake virtual XTC (references into the same file)", lambda: lake.universe("adk_water", 1), 20),
        ("lake virtual XTC, protein group", lambda: lake.universe("adk_water", 1, groups="protein"), 20),
        ("XTC original, 200-frame synthetic", lambda: mda.Universe(TPR, os.path.join(dcd_dir, "long.xtc")), 2),
        ("lake native, 200 frames, all atoms", lambda: lake.universe("adk_long", 0), 2),
        ("lake native, 200 frames, protein group", lambda: lake.universe("adk_long", 0, groups="protein"), 2),
        ("lake virtual XTC, 200 frames", lambda: lake.universe("adk_long", 1), 2),
        ("DCD original (MDAnalysis DCDReader)", lambda: mda.Universe(PSF, os.path.join(dcd_dir, "adk.dcd")), 5),
        ("lake virtual DCD (references into the same file)", lambda: lake.universe("adk_vac", 0), 5),
        ("lake native copy of the DCD", lambda: lake.universe("adk_vac", 1), 5),
    ]
    for name, make, passes in cases:
        u = make()
        iterate(u, 1)  # warm up
        print(f"  {name:52s} {iterate(u, passes):9,.0f}")


def main(workdir):
    lake = sl.Lake.create(os.path.join(workdir, "lake"))
    dcd_dir = os.path.join(workdir, "sources")
    os.makedirs(dcd_dir)
    shutil.copy(DCD, os.path.join(dcd_dir, "adk.dcd"))

    u = mda.Universe(TPR, XTC)
    positions = np.stack([u.atoms.positions.copy() for _ in u.trajectory])
    compression(positions, os.path.getsize(XTC))

    lake.create_system("adk_water", TPR)
    lake.add_replica("adk_water", u)
    shutil.copy(XTC, os.path.join(dcd_dir, "adk_water.xtc"))
    lake.register("adk_water", os.path.join(dcd_dir, "adk_water.xtc"))
    long_xtc = make_long_xtc(os.path.join(dcd_dir, "long.xtc"))
    lake.create_system("adk_long", TPR)
    lake.add_replica("adk_long", mda.Universe(TPR, long_xtc))
    lake.register("adk_long", long_xtc)
    print(f"\n200-frame synthetic: XTC {os.path.getsize(long_xtc):,d} B, lake native "
          f"{sum(a.nbytes_stored() for _, a in lake.system('adk_long').replica_group(0).arrays()):,d} B")

    lake.create_system("adk_vac", PSF)
    lake.register_dcd("adk_vac", os.path.join(dcd_dir, "adk.dcd"))
    lake.add_replica("adk_vac", mda.Universe(PSF, DCD))

    sys_ = lake.system("adk_water")
    g = sys_.replica_group(0)
    by_group = sys_.stored_bytes(0)
    total = sum(by_group.values())
    print("\nStored coordinates by atom group: " +
          ", ".join(f"{k} {v:,d} B ({100 * v / total:.1f}%)" for k, v in by_group.items()))

    throughput(lake, dcd_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as d:
            main(d)
