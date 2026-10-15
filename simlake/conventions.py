"""Constants that make up the (draft) simlake conventions, v0.3.

Layout of a lake (an Icechunk repository)::

    /                                   attrs: simlake_conventions
    systems/<name>/                     attrs: topology_hash, n_atoms, groups, metadata
        topology/                       (atom,) arrays in topology order: names resnames resids
                                        segids [types masses charges elements]; bonds (bond, 2);
                                        storage_order (atom,): topology index of each atom, groups
                                        concatenated in storage order;
                                        source/<i> (original topology files, raw bytes)
        replicas/<i>/                   attrs: storage ("native" | "virtual:xtc" | "virtual:dcd"),
                                               n_frames, precision, metadata, [sources]
            time  (frame,)              float64, ps
            step  (frame,)              int64
            box   (frame, 3, 3)         float32, Angstrom, triclinic vectors (periodic only)
            native:      positions/<group> (frame, atom, spatial), one array per group, atoms
                                      in topology order within the group; int32-quantized,
                                      pcodec; regular chunks, sharded
            virtual:xtc: position     (frame, atom, spatial)  topology order, nm, one chunk per frame
            virtual:dcd: position_sa  (frame, spatial, atom)  topology order, Angstrom

``groups`` is a list of ``{"name", "selection", "start", "stop"}``: a partition of all
atoms, with [start, stop) in storage order. See :mod:`simlake.groups`.

Chunking of native positions: every group uses the same frames per chunk (set by the
largest atom block), so groups line up frame for frame. Groups larger than
MAX_BLOCK_ATOMS are split into equal atom blocks. Shards are sized per group: about
TARGET_SHARD_BYTES, so small groups get few, long shards rather than many tiny objects.

Units follow MDAnalysis (Angstrom, ps) unless an array's ``units`` attribute says otherwise.
"""

CONVENTIONS_VERSION = "0.3"

LENGTH_UNIT = "Angstrom"
TIME_UNIT = "ps"

#: Default quantization step in Angstrom. 0.01 A == 0.001 nm, the XTC default.
DEFAULT_PRECISION = 0.01

#: Inner chunk and shard size targets. Inner chunks are the unit of decoding,
#: shards are the unit of storage (one object per shard on object stores).
TARGET_CHUNK_BYTES = 4 * 2**20
TARGET_SHARD_BYTES = 128 * 2**20

#: Groups larger than this are split into several (equal) blocks along the atom axis.
MAX_BLOCK_ATOMS = 16384

#: Residue names used by the default groups (see simlake.groups.DEFAULT_GROUPS).
WATER_RESNAMES = "SOL WAT HOH TIP3 TIP3P TIP4P TIP5P SPC SPCE T3P T4P W"
ION_RESNAMES = "NA CL K SOD CLA POT CAL MG ZN NA+ CL- K+ ION"
LIPID_RESNAMES = (
    "POPC POPE POPG POPS POPA DPPC DPPE DPPG DOPC DOPE DOPS DMPC DMPG DLPC DSPC "
    "CHL1 CHOL PSM SSM PIP2 POP2 CARD"
)

#: Per-atom topology fields stored when present on the Universe.
ATOM_FIELDS = ("names", "resnames", "resids", "segids", "types", "masses", "charges", "elements")

#: Fields that define topology identity. Two trajectories can be replicas of
#: the same system only if these match exactly, atom by atom.
IDENTITY_FIELDS = ("names", "resnames", "resids")
