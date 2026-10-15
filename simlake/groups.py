"""Atom groups: a named partition of a system's atoms, stored so each group reads alone.

Groups cover every atom exactly once. Each group's coordinates are their own Zarr
array (``positions/<group>``), so reading one group never touches another's data.
The *storage order* is the groups concatenated (within a group, topology order is
kept); the xarray view presents atoms in that order.

Groups are ordered by their first atom, so when groups are already contiguous in the
topology (the usual case: protein, then ligand, then water) storage order equals
topology order.
"""

import numpy as np

from .conventions import ION_RESNAMES, LIPID_RESNAMES, MAX_BLOCK_ATOMS, WATER_RESNAMES

#: Default groups, tried in order; the first match claims an atom.
DEFAULT_GROUPS = {
    "protein": "protein",
    "nucleic": "nucleic",
    "lipids": f"resname {LIPID_RESNAMES}",
    "water": f"resname {WATER_RESNAMES}",
    "ions": f"resname {ION_RESNAMES}",
}
OTHER = "other"


def resolve(atoms, groups=None):
    """Assign every atom to exactly one group.

    ``groups`` maps names to MDAnalysis selections. Without it, atoms are grouped by
    molecule type (:data:`DEFAULT_GROUPS`), and empty groups are dropped. Atoms that no
    group claims go into ``"other"``. User-supplied groups may not overlap.

    Returns a list of ``{"name", "selection", "index"}`` (topology indices into
    ``atoms``), ordered by first atom.
    """
    n = atoms.n_atoms
    owner = np.full(n, -1, dtype=np.int64)
    names, selections = [], []
    local = np.full(atoms.universe.atoms.n_atoms, -1, dtype=np.int64)
    local[atoms.indices] = np.arange(n)

    user = groups is not None
    for name, selection in (groups or DEFAULT_GROUPS).items():
        if name == OTHER:
            raise ValueError(f"'{OTHER}' is reserved for atoms no group claims")
        idx = local[atoms.select_atoms(selection).indices]
        idx = idx[idx >= 0]
        if user:
            clash = idx[owner[idx] >= 0]
            if clash.size:
                raise ValueError(
                    f"groups '{names[owner[clash[0]]]}' and '{name}' overlap "
                    f"(e.g. atom {clash[0]}); groups must not share atoms"
                )
        else:
            idx = idx[owner[idx] < 0]  # first default group wins
        if idx.size == 0:
            if user:
                raise ValueError(f"group '{name}' ({selection!r}) selects no atoms")
            continue
        owner[idx] = len(names)
        names.append(name)
        selections.append(selection)

    rest = np.nonzero(owner < 0)[0]
    if rest.size:
        owner[rest] = len(names)
        names.append(OTHER)
        selections.append(None)

    out = [{"name": nm, "selection": sel, "index": np.nonzero(owner == i)[0]}
           for i, (nm, sel) in enumerate(zip(names, selections))]
    out.sort(key=lambda g: g["index"][0])
    return out


def layout(resolved):
    """Storage order and per-group [start, stop) in storage positions."""
    order = np.concatenate([g["index"] for g in resolved])
    meta, start = [], 0
    for g in resolved:
        stop = start + len(g["index"])
        meta.append({"name": g["name"], "selection": g["selection"], "start": start, "stop": stop})
        start = stop
    return order, meta


def block_atoms(size, max_block=MAX_BLOCK_ATOMS):
    """Atoms per chunk for a group of ``size`` atoms: the group split into equal blocks."""
    n_blocks = -(-size // max_block)
    return -(-size // n_blocks)


def select(meta, names):
    """The named groups (in storage order) and their storage positions."""
    by_name = {g["name"]: g for g in meta}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise KeyError(f"unknown group(s) {unknown}; this system has {[g['name'] for g in meta]}")
    chosen = sorted((by_name[n] for n in dict.fromkeys(names)), key=lambda g: g["start"])
    return chosen, np.concatenate([np.arange(g["start"], g["stop"]) for g in chosen])
