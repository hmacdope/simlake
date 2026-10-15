"""Writing replicas: native (quantized, one sharded Zarr array per atom group) and
virtual (references into existing XTC/DCD files)."""

import os
import struct

import icechunk
import numpy as np
from MDAnalysis.lib.formats.libdcd import DCDFile
from MDAnalysis.coordinates.DCD import DCDReader
from zarr import codecs as zc

from .conventions import DEFAULT_PRECISION, LENGTH_UNIT, TARGET_CHUNK_BYTES, TARGET_SHARD_BYTES, TIME_UNIT
from . import groups as _groups
from . import locations, xtc
from .errors import UnsupportedVariant


def _coordinate_codecs(precision, level):
    """Quantize to integer multiples of ``precision``, then pcodec.

    pcodec is a numeric codec; on quantized coordinates it reaches ~0.9x the
    size of XTC, against ~0.55x for zstd (see benchmarks/bench.py).
    """
    filters = []
    if precision is not None:
        filters = [zc.ScaleOffset(offset=0, scale=1.0 / precision), zc.CastValue(data_type="int32")]
    return filters, zc.PCodec(level=level)


def position_chunking(group_meta, n_frames):
    """Chunk and shard shapes for each group's position array.

    All groups share the frames per chunk (set by the largest atom block), so they line
    up frame for frame. Shards are sized per group, so small groups get few long shards.
    """
    sizes = {g["name"]: g["stop"] - g["start"] for g in group_meta}
    blocks = {name: _groups.block_atoms(size) for name, size in sizes.items()}
    frames = max(1, min(n_frames, TARGET_CHUNK_BYTES // (max(blocks.values()) * 12)))
    max_shard = -(-n_frames // frames) * frames
    layout = {}
    for name, size in sizes.items():
        b = blocks[name]
        shard_atoms = -(-size // b) * b
        shard_frames = frames * max(1, TARGET_SHARD_BYTES // (frames * size * 12))
        layout[name] = {"chunks": (frames, b, 3), "shards": (min(shard_frames, max_shard), shard_atoms, 3)}
    return layout


def write_native(group, atoms, storage_order, group_meta, frames=slice(None),
                 precision=DEFAULT_PRECISION, level=8):
    """Iterate ``atoms.universe.trajectory[frames]`` and write a native replica into ``group``.

    Each atom group is written to its own array ``positions/<group>``.
    """
    traj = atoms.universe.trajectory
    selected = traj[frames]
    n_frames = len(selected)
    if n_frames == 0:
        raise ValueError("no frames selected")

    filters, serializer = _coordinate_codecs(precision, level)
    layout = position_chunking(group_meta, n_frames)
    positions = group.create_group("positions")
    parts = []
    for g in group_meta:
        index = storage_order[g["start"]:g["stop"]]
        contiguous = index[-1] - index[0] + 1 == len(index)
        arr = positions.create_array(
            g["name"],
            shape=(n_frames, len(index), 3),
            chunks=layout[g["name"]]["chunks"],
            shards=layout[g["name"]]["shards"],
            dtype="float32",
            filters=filters,
            serializer=serializer,
            compressors=None,
            fill_value=0,  # NaN cannot pass through the int32 cast of quantized arrays
            dimension_names=["frame", "atom", "spatial"],
        )
        arr.attrs["units"] = LENGTH_UNIT
        shard_frames = arr.shards[0]
        sel = slice(int(index[0]), int(index[-1]) + 1) if contiguous else index
        # Buffer one shard of frames per group, so every shard is written exactly once.
        parts.append({"arr": arr, "sel": sel, "buf": np.empty((shard_frames, len(index), 3), np.float32),
                      "start": 0})

    first_ts = traj[next(iter(range(traj.n_frames)[frames]))]
    periodic = first_ts.dimensions is not None
    box = np.full((n_frames, 3, 3), np.nan, dtype=np.float32) if periodic else None
    time = np.empty(n_frames, dtype=np.float64)
    step = np.empty(n_frames, dtype=np.int64)

    for i, ts in enumerate(selected):
        xyz = atoms.positions
        for p in parts:
            j = i - p["start"]
            p["buf"][j] = xyz[p["sel"]]
            if j + 1 == len(p["buf"]):
                p["arr"][p["start"]:i + 1] = p["buf"]
                p["start"] = i + 1
        if periodic and ts.dimensions is not None:
            box[i] = ts.triclinic_dimensions
        time[i] = ts.time
        step[i] = ts.data.get("step", ts.frame)
    for p in parts:
        if p["start"] < n_frames:
            p["arr"][p["start"]:n_frames] = p["buf"][: n_frames - p["start"]]

    _write_time_step(group, time, step)
    if periodic:
        b = group.create_array("box", data=box, dimension_names=["frame", "box_vector", "spatial"])
        b.attrs["units"] = LENGTH_UNIT
    group.attrs.update({
        "storage": "native",
        "n_frames": n_frames,
        "periodic": periodic,
        "precision": precision,
    })
    return n_frames


def _write_time_step(group, time, step):
    t = group.create_array("time", data=time, dimension_names=["frame"])
    t.attrs["units"] = TIME_UNIT
    group.create_array("step", data=step, dimension_names=["frame"])


def frames_to_keep(steps_per_file):
    """How many leading frames of each joined file to keep, so steps never repeat.

    The later file wins where files overlap, as in MDAnalysis' ``ChainReader(continuous=True)``:
    frames of earlier files at or beyond the next file's first step are dropped. That
    covers the duplicated boundary frame of GROMACS restarts without -append, and
    restarts from an older checkpoint (the rerun replaces what the crashed run wrote).
    """
    keep = [len(s) for s in steps_per_file]
    for k in range(1, len(steps_per_file)):
        if not len(steps_per_file[k]):
            continue
        first = steps_per_file[k][0]
        for j in range(k):
            beyond = np.nonzero(np.asarray(steps_per_file[j][: keep[j]]) >= first)[0]
            if beyond.size:
                keep[j] = int(beyond[0])
    return keep


# --------------------------------------------------------------------------- DCD


def dcd_layout(path):
    """Byte layout of a DCD file, or UnsupportedVariant if it cannot be referenced."""
    if locations.scheme(path) != "file":
        raise UnsupportedVariant("virtual DCD currently needs a local file; XTC works from object storage")
    f = DCDFile(path)
    try:
        header = f.header
        layout = {
            "path": os.path.abspath(path),
            "n_atoms": header["natoms"],
            "n_frames": f.n_frames,
            "header_size": f._header_size,
            "frame_size": f._framesize,
            "periodic": bool(header["is_periodic"]),
            "istart": header["istart"],
            "nsavc": header["nsavc"],
        }
        if f._firstframesize != f._framesize:
            raise UnsupportedVariant(f"{path}: DCD with fixed atoms is not supported")
        if f.ndims != 3:
            raise UnsupportedVariant(f"{path}: 4D CHARMM DCD is not supported")
    finally:
        f.close()

    with open(path, "rb") as fh:
        (marker,) = struct.unpack("<i", fh.read(4))
    if marker != 84:
        raise UnsupportedVariant(f"{path}: big-endian DCD is not supported yet")

    n = layout["n_atoms"]
    unitcell = 4 + 6 * 8 + 4 if layout["periodic"] else 0
    expected = unitcell + 3 * (4 + 4 * n + 4)
    if layout["frame_size"] != expected:
        raise UnsupportedVariant(
            f"{path}: unexpected frame size {layout['frame_size']} (expected {expected})"
        )
    layout["unitcell_bytes"] = unitcell
    return layout


def write_virtual_dcd(group, store, array_prefix, paths):
    """Reference coordinates in ``paths`` (concatenated) and copy time/step/box into the lake."""
    layouts = [dcd_layout(p) for p in paths]
    n_atoms = {lay["n_atoms"] for lay in layouts}
    if len(n_atoms) != 1:
        raise UnsupportedVariant(f"DCD files have different atom counts: {sorted(n_atoms)}")
    (n_atoms,) = n_atoms
    keep = frames_to_keep([lay["istart"] + np.arange(lay["n_frames"]) * lay["nsavc"] for lay in layouts])
    n_frames = sum(keep)

    pos = group.create_array(
        "position_sa",
        shape=(n_frames, 3, n_atoms),
        chunks=(1, 1, n_atoms),
        dtype="<f4",
        compressors=None,
        fill_value=None,
        dimension_names=["frame", "spatial", "atom"],
    )
    pos.attrs["units"] = LENGTH_UNIT

    specs = []
    time, step, boxes = [], [], []
    frame0 = 0
    sources = []
    for lay, n_keep in zip(layouts, keep):
        location = locations.source_url(lay["path"])
        fp = locations.fingerprint(lay["path"])
        checksum = locations.icechunk_checksum(fp)
        record = 4 + 4 * n_atoms + 4
        for i in range(n_keep):
            base = lay["header_size"] + i * lay["frame_size"] + lay["unitcell_bytes"]
            for k in range(3):
                specs.append(icechunk.VirtualChunkSpec(
                    index=[frame0 + i, k, 0],
                    location=location,
                    offset=base + k * record + 4,
                    length=4 * n_atoms,
                    **checksum,
                ))
        # Small per-frame metadata is copied in, so time/box queries never touch the source.
        reader = DCDReader(lay["path"])
        for i, ts in enumerate(reader):
            if i >= n_keep:
                break
            time.append(ts.time)
            step.append(lay["istart"] + i * lay["nsavc"])
            boxes.append(ts.triclinic_dimensions if ts.dimensions is not None else np.full((3, 3), np.nan))
        reader.close()
        sources.append({**fp, "format": "dcd", "n_frames": n_keep,
                        "dropped_frames": lay["n_frames"] - n_keep})
        frame0 += n_keep

    failed = store.set_virtual_refs(f"{array_prefix}/position_sa", specs)
    if failed:
        raise RuntimeError(f"failed to set {len(failed)} virtual references")

    _write_time_step(group, np.asarray(time, dtype=np.float64), np.asarray(step, dtype=np.int64))
    periodic = any(lay["periodic"] for lay in layouts)
    if periodic:
        b = group.create_array("box", data=np.asarray(boxes, dtype=np.float32),
                               dimension_names=["frame", "box_vector", "spatial"])
        b.attrs["units"] = LENGTH_UNIT
    group.attrs.update({
        "storage": "virtual:dcd",
        "n_frames": n_frames,
        "periodic": periodic,
        "precision": None,
        "sources": sources,
    })
    return n_atoms, n_frames


# --------------------------------------------------------------------------- XTC


def write_virtual_xtc(group, store, array_prefix, urls):
    """Reference the frames of XTC file(s) (local or object storage) without copying them.

    Each frame becomes one chunk, decoded on read by :class:`simlake.xtc.XTCFrameCodec`.
    Frame headers are scanned once, and time, step and box are copied into the lake.
    """
    scans, fps = [], []
    for url in urls:
        fp = locations.fingerprint(url)
        with locations.open_source(url) as fh:
            scans.append(xtc.scan(fh, fp["size"]))
        fps.append(fp)
    n_atoms = {s["n_atoms"] for s in scans}
    if len(n_atoms) != 1:
        raise UnsupportedVariant(f"XTC files have different atom counts: {sorted(n_atoms)}")
    (n_atoms,) = n_atoms
    keep = frames_to_keep([s["step"] for s in scans])
    scans = [{k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in s.items()} | {"dropped": len(s["offset"]) - n}
             for s, n in zip(scans, keep)]
    n_frames = sum(keep)

    pos = group.create_array(
        "position",
        shape=(n_frames, n_atoms, 3),
        chunks=(1, n_atoms, 3),
        dtype="float32",
        serializer=xtc.XTCFrameCodec(),
        compressors=None,
        fill_value=0.0,
        dimension_names=["frame", "atom", "spatial"],
    )
    pos.attrs["units"] = "nm"

    specs, frame0 = [], 0
    for scan_, fp in zip(scans, fps):
        checksum = locations.icechunk_checksum(fp)
        for i, (offset, length) in enumerate(zip(scan_["offset"], scan_["length"])):
            specs.append(icechunk.VirtualChunkSpec(
                index=[frame0 + i, 0, 0], location=fp["uri"],
                offset=int(offset), length=int(length), **checksum,
            ))
        frame0 += len(scan_["offset"])
    failed = store.set_virtual_refs(f"{array_prefix}/position", specs)
    if failed:
        raise RuntimeError(f"failed to set {len(failed)} virtual references")

    time = np.concatenate([s["time"] for s in scans])
    step = np.concatenate([s["step"] for s in scans])
    box = np.concatenate([s["box"] for s in scans]) * np.float32(10.0)  # nm -> Angstrom
    _write_time_step(group, time, step)
    periodic = bool(np.any(box != 0))
    if periodic:
        b = group.create_array("box", data=box.astype(np.float32),
                               dimension_names=["frame", "box_vector", "spatial"])
        b.attrs["units"] = LENGTH_UNIT
    group.attrs.update({
        "storage": "virtual:xtc",
        "n_frames": n_frames,
        "periodic": periodic,
        "precision": None,
        "sources": [{**fp, "format": "xtc", "n_frames": len(s["offset"]), "dropped_frames": s["dropped"]}
                    for fp, s in zip(fps, scans)],
    })
    return n_atoms, n_frames
