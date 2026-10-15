"""Extracting, hashing, storing and rebuilding MDAnalysis topologies."""

import hashlib
import os

import numpy as np
from MDAnalysis.core import topologyattrs as ta
from MDAnalysis.core.topology import Topology
from MDAnalysis.topology.base import change_squash

from .conventions import ATOM_FIELDS, IDENTITY_FIELDS
from .errors import TopologyMismatch

_STRING_FIELDS = {"names", "resnames", "segids", "types", "elements"}

_ATTR_CLASSES = {
    "names": ta.Atomnames,
    "types": ta.Atomtypes,
    "masses": ta.Masses,
    "charges": ta.Charges,
    "elements": ta.Elements,
}


def extract(atoms):
    """Return a dict of per-atom numpy arrays (plus ``bonds``) for an AtomGroup.

    Bonds are re-indexed to positions within ``atoms`` and only bonds with both
    partners inside ``atoms`` are kept.
    """
    top = {}
    for field in ATOM_FIELDS:
        try:
            values = getattr(atoms, field)
        except (AttributeError, Exception):  # NoDataError subclasses both
            continue
        values = np.asarray(values)
        top[field] = values.astype(str) if field in _STRING_FIELDS else values

    try:
        bonds = atoms.universe.bonds.to_indices()
    except Exception:
        bonds = None
    if bonds is not None and len(bonds):
        local = np.full(atoms.universe.atoms.n_atoms, -1, dtype=np.int64)
        local[atoms.indices] = np.arange(atoms.n_atoms)
        mapped = local[bonds]
        top["bonds"] = mapped[(mapped >= 0).all(axis=1)]
    return top


def topology_hash(top):
    """Hash of the identity fields; equal hashes mean atom-by-atom identical systems."""
    h = hashlib.sha256()
    h.update(str(len(top["names"])).encode())
    for field in IDENTITY_FIELDS:
        h.update(field.encode())
        h.update(np.ascontiguousarray(top[field]).astype(str).astype("U").tobytes())
    return "sha256:" + h.hexdigest()


def check_match(expected_hash, expected_n_atoms, top, stored=None):
    """Raise TopologyMismatch with a useful message if ``top`` differs."""
    n = len(top["names"])
    if n != expected_n_atoms:
        raise TopologyMismatch(
            f"atom count differs: system has {expected_n_atoms}, trajectory has {n}"
        )
    if topology_hash(top) == expected_hash:
        return
    detail = ""
    if stored is not None:
        for field in IDENTITY_FIELDS:
            diff = np.nonzero(np.asarray(stored[field]) != np.asarray(top[field]))[0]
            if diff.size:
                i = diff[0]
                detail = (
                    f"; first difference in '{field}' at atom {i}: "
                    f"{stored[field][i]!r} != {top[field][i]!r}"
                )
                break
    raise TopologyMismatch(f"topology identity fields differ{detail}")


def write(group, top, storage_order, source_files=()):
    """Write a topology dict into a zarr group.

    ``storage_order[i]`` is the topology index of the atom stored at position ``i``
    in native position arrays (atoms sorted by group).
    """
    for field, values in top.items():
        dims = ["bond", "pair"] if field == "bonds" else ["atom"]
        group.create_array(field, data=values, dimension_names=dims)
    group.create_array("storage_order", data=np.asarray(storage_order, dtype=np.int64), dimension_names=["atom"])

    sources = group.create_group("source")
    for i, path in enumerate(source_files):
        with open(path, "rb") as fh:
            blob = np.frombuffer(fh.read(), dtype=np.uint8)
        arr = sources.create_array(str(i), data=blob)
        arr.attrs["filename"] = os.path.basename(path)


def read(group):
    """Read a topology group back into a dict of numpy arrays."""
    skip = {"storage_order"}
    return {name: arr[:] for name, arr in group.arrays() if name not in skip}


def source_files(group):
    """Return {filename: bytes} for the original topology files stored with a system."""
    if "source" not in group:
        return {}
    return {arr.attrs["filename"]: arr[:].tobytes() for _, arr in group["source"].arrays()}


def subset(top, index):
    """Topology dict restricted to ``index`` (sorted original atom indices)."""
    index = np.asarray(index)
    out = {k: v[index] for k, v in top.items() if k != "bonds"}
    if "bonds" in top:
        local = np.full(len(top["names"]), -1, dtype=np.int64)
        local[index] = np.arange(len(index))
        mapped = local[top["bonds"]]
        out["bonds"] = mapped[(mapped >= 0).all(axis=1)]
    return out


def to_mda(top):
    """Build an MDAnalysis Topology from a topology dict."""
    n_atoms = len(top["names"])
    resids = np.asarray(top["resids"])
    resnames = np.asarray(top["resnames"], dtype=object)
    segids = np.asarray(top.get("segids", np.full(n_atoms, "SYSTEM")), dtype=object)

    residx, (res_resids, res_resnames, res_segids) = change_squash(
        (resids, resnames, segids), (resids, resnames, segids)
    )
    segidx, (seg_segids,) = change_squash((res_segids,), (res_segids,))

    attrs = [
        ta.Atomids(np.arange(1, n_atoms + 1)),
        ta.Resids(res_resids),
        ta.Resnums(res_resids.copy()),
        ta.Resnames(res_resnames),
        ta.Segids(seg_segids),
    ]
    for field, cls in _ATTR_CLASSES.items():
        if field in top:
            values = top[field]
            attrs.append(cls(values.astype(object) if field in _STRING_FIELDS else values))
    if "bonds" in top and len(top["bonds"]):
        attrs.append(ta.Bonds([tuple(b) for b in top["bonds"].tolist()]))

    return Topology(
        n_atoms,
        len(res_resids),
        len(seg_segids),
        attrs=attrs,
        atom_resindex=residx,
        residue_segindex=segidx,
    )
