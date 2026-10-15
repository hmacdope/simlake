"""simlake: a prototype versioned Zarr v3 / Icechunk lake for MD trajectories."""

from .conventions import CONVENTIONS_VERSION
from .errors import SimlakeError, SystemExists, TopologyMismatch, UnsupportedVariant
from .lake import Lake, System, Transaction
from .locations import configure
from . import mda as _mda  # registers the SIMLAKE topology parser and reader with MDAnalysis
from . import xr as _xr  # registers the ``.md`` xarray accessor
from . import xtc as _xtc  # registers the simlake.xtc_frame zarr codec

__all__ = [
    "CONVENTIONS_VERSION",
    "Lake",
    "configure",
    "System",
    "Transaction",
    "SimlakeError",
    "SystemExists",
    "TopologyMismatch",
    "UnsupportedVariant",
]
