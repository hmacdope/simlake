"""MDAnalysis topology parser and trajectory reader for ``simlake://`` URIs.

Importing simlake registers both, so this works::

    u = mda.Universe("simlake:///path/to/lake?system=adk&replica=0")
    u = mda.Universe("my.psf", "simlake:///path/to/lake?system=adk&replica=0")
"""

import multiprocessing
import sys
import warnings
from urllib.parse import parse_qs, unquote

import numpy as np
from MDAnalysis.coordinates import base, core
from MDAnalysis.topology.base import TopologyReaderBase

from . import groups as _groups
from . import locations
from . import topology as _topology

SCHEME = "simlake://"


def parse_uri(uri):
    path, _, query = uri[len(SCHEME):].partition("?")
    q = {k: v[0] for k, v in parse_qs(query).items()}
    if "system" not in q:
        raise ValueError(f"simlake URI needs a 'system' parameter: {uri}")
    return {
        "path": unquote(path),
        "system": q["system"],
        "replica": int(q.get("replica", 0)),
        "ref": q.get("ref"),
        "groups": [unquote(g) for g in q["groups"].split(",")] if q.get("groups") else None,
    }


def _selection(system, group_names):
    """What to read for the requested groups (None = all atoms).

    Returns (group names in storage order, topology indices of the Universe's atoms in
    Universe order, permutation from stored order to Universe order or None if already
    in order). The Universe lists atoms in topology order.
    """
    storage_order = system.storage_order
    names = group_names if group_names is not None else system.group_names
    chosen, positions = _groups.select(system.groups, names)
    topo = storage_order[positions]
    perm = np.argsort(topo, kind="stable")
    if np.array_equal(perm, np.arange(len(perm))):
        perm = None
    return [g["name"] for g in chosen], np.sort(topo), perm


def _fork_start_method():
    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:  # not chosen yet: the platform default applies
        return sys.platform == "linux" and sys.version_info < (3, 14)
    return method == "fork"


def _is_uri(thing):
    return isinstance(thing, str) and thing.startswith(SCHEME)


def _open_system(spec):
    from .lake import Lake

    return Lake.open(spec["path"]).system(spec["system"], ref=spec["ref"])


class SimlakeParser(TopologyReaderBase):
    """Builds an MDAnalysis topology from a lake system."""

    format = "SIMLAKE"

    @staticmethod
    def _format_hint(thing):
        return _is_uri(thing)

    def parse(self, **kwargs):
        spec = parse_uri(self.filename)
        system = _open_system(spec)
        top = system.topology()
        if spec["groups"]:
            _, topo, _ = _selection(system, spec["groups"])
            top = _topology.subset(top, topo)
        return _topology.to_mda(top)


class SimlakeReader(base.ReaderBase):
    """Reads one replica of a lake system, native or virtual, frame by frame.

    Native replicas are decoded one chunk of frames at a time per group array, and the
    decoded block is kept, so sequential iteration decodes each chunk once. With
    ``groups`` only those groups' arrays are read (native); virtual replicas decode
    whole frames regardless.
    """

    format = "SIMLAKE"
    units = {"time": "ps", "length": "Angstrom"}

    @staticmethod
    def _format_hint(thing):
        return _is_uri(thing)

    def __init__(self, filename, convert_units=True, **kwargs):
        super().__init__(filename, convert_units=convert_units, **kwargs)
        self._spec = parse_uri(filename)
        self._open()
        if self._n_frames > 1:
            self._ts_kwargs.setdefault("dt", float(self._time[1] - self._time[0]))
        self.ts = self._Timestep(self.n_atoms, **self._ts_kwargs)
        self._read_frame(0)

    def _open(self):
        system = _open_system(self._spec)
        group = system.replica_group(self._spec["replica"])
        attrs = dict(group.attrs)
        self._storage = attrs["storage"]
        self._n_frames = attrs["n_frames"]
        self._time = group["time"][:]
        self._step = group["step"][:]
        self._box = group["box"][:] if "box" in group else None
        names, topo, perm = _selection(system, self._spec["groups"])
        self.n_atoms = len(topo)
        self._perm = perm

        if self._storage == "native":
            self._arrays = [group[f"positions/{name}"] for name in names]
        elif self._storage in ("virtual:dcd", "virtual:xtc"):
            if self._storage == "virtual:dcd":
                self._virtual = group["position_sa"]  # (frame, spatial, atom), Angstrom
                self._virtual_scale = None
            else:
                self._virtual = group["position"]  # (frame, atom, spatial), nm
                self._virtual_scale = np.float32(10.0)  # same float32 multiply as MDAnalysis
            # virtual arrays are in topology order
            self._atom_index = None if self._spec["groups"] is None else topo
        else:
            raise ValueError(f"unknown replica storage {self._storage!r}")
        self._blocks = {}

    @property
    def n_frames(self):
        return self._n_frames

    #: Frames fetched per request for virtual replicas, whose chunks are single
    #: frames; one request per block lets zarr fetch the chunks concurrently.
    virtual_block_frames = 64

    def _block(self, arr, frame, n=None):
        """Decoded block of ``n`` frames containing ``frame`` (cached per array).

        ``n`` defaults to the inner chunk's frames, so each chunk is decoded once during
        sequential iteration; the atom blocks of large groups decode in parallel.
        """
        n = n or arr.chunks[0]
        start = (frame // n) * n
        cached = self._blocks.get(arr.path)
        if cached is None or cached[0] != start:
            cached = (start, arr[start:start + n])
            self._blocks[arr.path] = cached
        return cached[1][frame - start]

    def _read_frame(self, frame):
        if frame < 0 or frame >= self._n_frames:
            raise IOError("frame index out of range")
        ts = self.ts
        out = ts._pos  # write straight into the timestep's buffer: no intermediate copies
        if self._storage == "native":
            if len(self._arrays) == 1 and self._perm is None:
                out[...] = self._block(self._arrays[0], frame)
            else:
                parts = [self._block(arr, frame) for arr in self._arrays]
                positions = np.concatenate(parts)
                out[...] = positions if self._perm is None else positions[self._perm]
        else:
            block = self._block(self._virtual, frame, self.virtual_block_frames)
            if self._virtual_scale is None:  # DCD: (spatial, atom)
                block = block.T
            if self._atom_index is not None:
                block = block[self._atom_index]
            if self._virtual_scale is None:
                out[...] = block
            else:  # XTC: nm -> Angstrom, the same float32 multiply as MDAnalysis
                np.multiply(block, self._virtual_scale, out=out)
        ts.has_positions = True
        ts.frame = frame
        ts.time = self._time[frame]
        ts.data["step"] = int(self._step[frame])
        if self._box is not None and np.all(np.isfinite(self._box[frame])):
            ts.dimensions = core.triclinic_box(*self._box[frame])
        else:
            ts.dimensions = None
        return ts

    def _read_next_timestep(self, ts=None):
        if self.ts.frame + 1 >= self._n_frames:
            raise IOError("end of trajectory")
        return self._read_frame(self.ts.frame + 1)

    def _reopen(self):
        self.ts.frame = -1

    def close(self):
        """Drop decoded blocks. The reader stays usable; blocks are decoded again on demand."""
        self._blocks = {}

    # Readers are pickled when analyses run with multiprocessing or dask backends.
    _UNPICKLABLE = ("_arrays", "_virtual", "_blocks")

    def __getstate__(self):
        if _fork_start_method():
            warnings.warn(
                "Pickling a simlake reader while multiprocessing uses 'fork'. Icechunk is not "
                "fork-safe and forked workers can deadlock; call "
                "multiprocessing.set_start_method('forkserver') (or 'spawn') first.",
                RuntimeWarning,
                stacklevel=2,
            )
        state = self.__dict__.copy()
        for k in self._UNPICKLABLE:
            state.pop(k, None)
        # endpoints/credentials configured in this process, for spawned workers
        state["_locations"] = locations.snapshot()
        return state

    def __setstate__(self, state):
        locations.restore(state.pop("_locations", {}))
        self.__dict__.update(state)
        self._open()
        frame = self.ts.frame
        if 0 <= frame < self._n_frames:
            self._read_frame(frame)
