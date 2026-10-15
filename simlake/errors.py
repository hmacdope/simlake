class SimlakeError(Exception):
    """Base class for simlake errors."""


class TopologyMismatch(SimlakeError):
    """Atoms being added do not match the system topology."""


class UnsupportedVariant(SimlakeError):
    """A file uses a variant of its format that cannot be referenced virtually."""


class SystemExists(SimlakeError):
    """A system with this name already exists in the lake."""
