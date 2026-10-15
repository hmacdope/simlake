"""Reading XTC files in place: a frame scanner and a Zarr codec that decodes one frame.

An XTC file is a sequence of independently compressed frames. Each frame becomes one
Zarr chunk that is a virtual reference to the frame's byte range; ``XTCFrameCodec``
turns those bytes into a ``(1, n_atoms, 3)`` float32 array in nm.

The decoder (``simlake/_xtc.pyx``) is a Cython port of the ``xdr3dfcoord`` read path
used by the GROMACS xdrfile library, checked bit-for-bit against MDAnalysis in the
tests. It releases the GIL, and the codec runs it on zarr's thread pool, so frames
decode in parallel.
"""

import asyncio
import struct
from dataclasses import dataclass

import numpy as np

from zarr.abc.codec import ArrayBytesCodec
from zarr.registry import register_codec

from . import _xtc
from .errors import UnsupportedVariant

XTC_MAGIC = 1995
_HEADER = struct.Struct(">iiif9fi")  # magic, natoms, step, time, box[9], natoms again
_COMPRESSED = struct.Struct(">f3i3iii")  # precision, minint[3], maxint[3], smallidx, nbytes
HEADER_BYTES = _HEADER.size  # 56
COMPRESSED_HEADER_BYTES = HEADER_BYTES + _COMPRESSED.size  # 92

# --------------------------------------------------------------------------- frame layout


def frame_size(header_bytes):
    """Total size in bytes of the frame whose first 92 bytes are given."""
    magic, natoms = struct.unpack_from(">ii", header_bytes)
    if magic != XTC_MAGIC:
        raise UnsupportedVariant(f"not an XTC frame (magic {magic}); only magic 1995 is supported")
    if natoms <= 9:
        return HEADER_BYTES + 12 * natoms
    nbytes = struct.unpack_from(">i", header_bytes, COMPRESSED_HEADER_BYTES - 4)[0]
    return COMPRESSED_HEADER_BYTES + ((nbytes + 3) // 4) * 4


def scan(fileobj, size):
    """Walk the frame headers of an XTC file without decoding coordinates.

    Returns a dict of numpy arrays: offset, length, step, time (ps) and box (nm, (n, 3, 3)),
    plus n_atoms.
    """
    offsets, lengths, steps, times, boxes = [], [], [], [], []
    n_atoms = None
    pos = 0
    while pos < size:
        fileobj.seek(pos)
        head = fileobj.read(COMPRESSED_HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise UnsupportedVariant(f"truncated XTC frame at byte {pos}")
        fields = _HEADER.unpack_from(head)
        magic, natoms, step, time = fields[:4]
        if magic != XTC_MAGIC:
            raise UnsupportedVariant(f"unexpected XTC magic {magic} at byte {pos}")
        if n_atoms is None:
            n_atoms = natoms
        elif natoms != n_atoms:
            raise UnsupportedVariant("XTC with a varying number of atoms is not supported")
        length = frame_size(head)
        offsets.append(pos)
        lengths.append(length)
        steps.append(step)
        times.append(time)
        boxes.append(fields[4:13])
        pos += length
    return {
        "n_atoms": n_atoms,
        "offset": np.asarray(offsets, dtype=np.int64),
        "length": np.asarray(lengths, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "time": np.asarray(times, dtype=np.float32).astype(np.float64),
        "box": np.asarray(boxes, dtype=np.float32).reshape(-1, 3, 3),
    }


# --------------------------------------------------------------------------- decoder


def decode_frame(data):
    """Decode one XTC frame (its full byte range) into an ``(n_atoms, 3)`` float32 array in nm."""
    data = memoryview(data).cast("B")
    head = _HEADER.unpack_from(data)
    magic, natoms = head[0], head[1]
    if magic != XTC_MAGIC:
        raise UnsupportedVariant(f"not an XTC frame (magic {magic})")
    if natoms <= 9:
        raw = np.frombuffer(data, dtype=">f4", count=3 * natoms, offset=HEADER_BYTES)
        return raw.astype(np.float32).reshape(natoms, 3)

    precision, *rest = _COMPRESSED.unpack_from(data, HEADER_BYTES)
    minint = np.asarray(rest[0:3], dtype=np.int64)
    maxint = np.asarray(rest[3:6], dtype=np.int64)
    smallidx, nbytes = rest[6], rest[7]
    # a few bytes of zero padding so the bit reader may look one word ahead
    buf = np.zeros(nbytes + 8, dtype=np.uint8)
    buf[:nbytes] = np.frombuffer(data, dtype=np.uint8, count=nbytes, offset=COMPRESSED_HEADER_BYTES)
    out = np.empty((natoms, 3), dtype=np.float32)
    n = _xtc.decompress(buf, natoms, minint, maxint, smallidx, precision, out)
    if n != natoms:
        raise ValueError(f"corrupt XTC frame: decoded {n} of {natoms} atoms")
    return out


# --------------------------------------------------------------------------- zarr codec


@dataclass(frozen=True)
class XTCFrameCodec(ArrayBytesCodec):
    """Read-only Zarr v3 codec: one chunk = the bytes of one XTC frame.

    The chunk shape must be ``(1, n_atoms, 3)`` and the dtype float32. Values are nm.
    """

    is_fixed_size = False

    @classmethod
    def from_dict(cls, data):
        return cls()

    def to_dict(self):
        return {"name": "simlake.xtc_frame"}

    def compute_encoded_size(self, input_byte_length, chunk_spec):
        raise NotImplementedError("XTC frames have variable size")

    def _decode_sync(self, chunk_bytes, chunk_spec):
        xyz = decode_frame(chunk_bytes.as_array_like())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(xyz.reshape(chunk_spec.shape))

    async def _decode_single(self, chunk_bytes, chunk_spec):
        # zarr runs codecs on one event-loop thread; hand decoding to zarr's thread pool
        # (as zarr's own zstd/blosc codecs do) so many frames decode at once. The decoder
        # releases the GIL. Pool size: zarr.config "threading.max_workers".
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def _encode_sync(self, chunk_array, chunk_spec):
        raise NotImplementedError("simlake.xtc_frame is read-only; write native replicas instead")

    async def _encode_single(self, chunk_array, chunk_spec):
        return self._encode_sync(chunk_array, chunk_spec)


register_codec("simlake.xtc_frame", XTCFrameCodec)
