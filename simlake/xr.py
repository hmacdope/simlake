"""Lazy xarray views of lake systems, and the ``.md`` accessor."""

import functools

import dask.array as da
import numpy as np
import xarray as xr

from . import topology as _topology


def _dask(arr, frame_chunk=None):
    chunks = arr.chunks
    if frame_chunk:
        chunks = (frame_chunk, *arr.shape[1:])
    return da.from_array(arr, chunks=chunks, lock=False, asarray=True, name=f"simlake-{arr.path}")


def _replica_dataset(system, replica, storage_order):
    group = system.replica_group(replica)
    attrs = dict(group.attrs)
    data = {
        "time": (("frame",), group["time"][:], {"units": "ps"}),
        "step": (("frame",), group["step"][:]),
    }
    if "box" in group:
        data["box"] = (("frame", "box_vector", "spatial"), _dask(group["box"]), {"units": "Angstrom"})

    if attrs["storage"] == "native":
        # One array per atom group; groups share frame chunking, so they join block for block.
        pos = da.concatenate([_dask(group[f"positions/{name}"]) for name in system.group_names], axis=1)
    else:
        # Virtual replicas have one frame per chunk in topology order. Batch frames into
        # larger dask chunks, convert to (frame, atom, spatial) in Angstrom, reorder.
        n_frames = attrs["n_frames"]
        if attrs["storage"] == "virtual:dcd":
            pos = _dask(group["position_sa"], frame_chunk=min(n_frames, 32)).transpose(0, 2, 1)
        else:
            pos = _dask(group["position"], frame_chunk=min(n_frames, 32)) * np.float32(10.0)
        if not np.array_equal(storage_order, np.arange(len(storage_order))):
            pos = pos[:, storage_order]
    data["position"] = (("frame", "atom", "spatial"), pos, {"units": "Angstrom"})

    ds = xr.Dataset(data, coords={"frame": np.arange(attrs["n_frames"])})
    ds["storage"] = attrs["storage"]
    ds["n_frames"] = attrs["n_frames"]
    return ds


def open_system_dataset(system):
    """All replicas of a system as one Dataset with a ``replica`` dimension.

    ``position`` is ``(replica, frame, atom, spatial)`` with atoms in storage order
    (sorted by group). The ``atom`` coordinate holds topology indices, and ``atom_group``
    names each atom's group. Replicas of different length are padded with NaN along
    ``frame``; use ``n_frames`` (per replica) or ``time.notnull()`` to mask.
    """
    replicas = system.replicas()
    if not replicas:
        raise ValueError(f"system {system.name!r} has no replicas")
    storage_order = system.storage_order
    parts = [_replica_dataset(system, r, storage_order) for r in replicas]
    ds = xr.concat(parts, dim="replica", join="outer", fill_value={"step": -1})

    top = system.topology()
    group_of = np.empty(len(storage_order), dtype=object)
    for g in system.groups:
        group_of[g["start"]:g["stop"]] = g["name"]
    ds = ds.assign_coords(
        replica=replicas,
        spatial=["x", "y", "z"],
        atom=storage_order,
        atom_group=("atom", group_of.astype(str)),
        atom_name=("atom", top["names"][storage_order]),
        atom_resname=("atom", top["resnames"][storage_order]),
        atom_resid=("atom", top["resids"][storage_order]),
        **({"atom_segid": ("atom", top["segids"][storage_order])} if "segids" in top else {}),
    )
    ds.attrs.update({
        "simlake_lake": system.lake.location,
        "simlake_system": system.name,
        "simlake_ref": system.ref or "main",
        "simlake_topology_hash": system.attrs["topology_hash"],
        "simlake_groups": ", ".join(g["name"] for g in system.groups),
    })
    return ds


@functools.lru_cache(maxsize=16)
def _topology_universe(lake, system, ref):
    import MDAnalysis as mda
    from .lake import Lake

    return mda.Universe(_topology.to_mda(Lake.open(lake).system(system, ref=ref).topology()))


def _as_index(positions):
    """Positions along ``atom`` as a slice when contiguous (keeps dask chunks intact)."""
    if len(positions) and positions[-1] - positions[0] + 1 == len(positions):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions


@xr.register_dataset_accessor("md")
class MDAccessor:
    """MD helpers on simlake datasets; selections use the MDAnalysis selection language."""

    def __init__(self, ds):
        self._ds = ds

    @property
    def topology_universe(self):
        """A coordinate-free MDAnalysis Universe for this system's topology (cached)."""
        a = self._ds.attrs
        return _topology_universe(a["simlake_lake"], a["simlake_system"], a["simlake_ref"])

    @property
    def groups(self):
        """Names of the atom groups present in this dataset, in storage order."""
        return [str(g) for g in dict.fromkeys(self._ds.atom_group.values)]

    def group(self, *names):
        """Only the atoms of the named groups. Reads only those groups' chunks."""
        unknown = set(names) - set(self.groups)
        if unknown:
            raise KeyError(f"unknown group(s) {sorted(unknown)}; this dataset has {self.groups}")
        positions = np.nonzero(np.isin(self._ds.atom_group.values, names))[0]
        return self._ds.isel(atom=_as_index(positions))

    def select(self, selection):
        """Only the atoms matching an MDAnalysis selection (topology-based selections only).

        Reads only the chunks that contain selected atoms.
        """
        index = self.topology_universe.select_atoms(selection).indices
        positions = np.nonzero(np.isin(self._ds.atom.values, index))[0]
        return self._ds.isel(atom=_as_index(positions))

    @property
    def positions(self):
        """``position`` as ``(replica, frame, atom, spatial)``."""
        return self._ds.position

    def rmsd(self, ref_frame=0, superposition=True):
        """RMSD of every frame against ``ref_frame`` of the same replica, computed with dask.

        Uses MDAnalysis' QCP kernel (``MDAnalysis.analysis.rms.rmsd``) frame by frame, so
        results match ``MDAnalysis.analysis.rms.RMSD`` without its parallel backends.
        Select atoms first, e.g. ``ds.md.select("backbone").md.rmsd()``.
        """
        pos = self._ds.position.chunk({"atom": -1, "spatial": -1})
        ref = pos.isel(frame=ref_frame, drop=True)
        return xr.apply_ufunc(
            _rmsd_kernel, pos, ref,
            input_core_dims=[["atom", "spatial"], ["atom", "spatial"]],
            kwargs={"superposition": superposition},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
        ).rename("rmsd")


def _rmsd_kernel(a, b, superposition=True):
    from MDAnalysis.analysis.rms import rmsd

    if np.isnan(a).any() or np.isnan(b).any():  # padding beyond a replica's last frame
        return np.nan
    return rmsd(a.astype(np.float64), b.astype(np.float64), center=True, superposition=superposition)
