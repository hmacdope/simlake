# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""XTC frame decompression (the xdr3dfcoord read path), in Cython.

A clean-room port of the algorithm used by GROMACS' xdrfile: coordinates are integers
(position / precision), packed per atom as one mixed-radix number, with runs of small
deltas for nearby atoms (and the first two atoms of a run swapped, for water).
Checked bit-for-bit against MDAnalysis in tests/test_xtc.py.

The bit reader loads a big-endian 64-bit window at the current bit position, so the
input buffer must carry 8 bytes of zero padding.
"""

from libc.stdint cimport int64_t, uint8_t, uint64_t

cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>

    static const int64_t SIMLAKE_MAGICINTS[73] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
        80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
        1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
        16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
        131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
        832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
        4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216};

    static inline uint64_t simlake_load_be64(const uint8_t *p) {
        uint64_t v;
        memcpy(&v, p, 8);
    #if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return v;
    #else
        return __builtin_bswap64(v);
    #endif
    }

    static inline uint64_t simlake_bswap64(uint64_t v) { return __builtin_bswap64(v); }
    """
    const int64_t SIMLAKE_MAGICINTS[73]
    uint64_t simlake_load_be64(const uint8_t *p) nogil
    uint64_t simlake_bswap64(uint64_t v) nogil

cdef int FIRSTIDX = 9  # first usable index into the magic integers


cdef inline uint64_t getbits(const uint8_t *buf, uint64_t *pos, int nbits) noexcept nogil:
    """Next ``nbits`` (<= 57) bits, most significant first."""
    if nbits == 0:
        return 0
    cdef uint64_t p = pos[0]
    cdef uint64_t w = simlake_load_be64(buf + (p >> 3)) << (p & 7)
    pos[0] = p + nbits
    return w >> (64 - nbits)


cdef inline int sizeofint(int64_t size) noexcept nogil:
    cdef int64_t num = 1
    cdef int bits = 0
    while size >= num and bits < 32:
        bits += 1
        num <<= 1
    return bits


cdef int sizeofints(const int64_t *sizes) noexcept nogil:
    cdef int64_t byts[32]
    cdef int nbytes = 1, i, bytecnt, bits = 0
    cdef int64_t tmp, num
    byts[0] = 1
    for i in range(3):
        tmp = 0
        bytecnt = 0
        while bytecnt < nbytes:
            tmp = byts[bytecnt] * sizes[i] + tmp
            byts[bytecnt] = tmp & 0xFF
            tmp >>= 8
            bytecnt += 1
        while tmp != 0:
            byts[bytecnt] = tmp & 0xFF
            bytecnt += 1
            tmp >>= 8
        nbytes = bytecnt
    num = 1
    nbytes -= 1
    while byts[nbytes] >= num:
        bits += 1
        num *= 2
    return bits + nbytes * 8


cdef inline void receiveints(const uint8_t *buf, uint64_t *pos, int nbits,
                             const int64_t *sizes, int64_t *out) noexcept nogil:
    """Three integers packed as one mixed-radix number, stored as little-endian bytes."""
    cdef int full, last, i, j, nbytes
    cdef uint64_t v
    cdef int64_t num, p
    cdef int64_t byts[32]
    if nbits <= 64:
        # All full bytes in one read, byte-swapped into little-endian order, then the
        # final (partial) byte; unpack with two divmods.
        full = (nbits - 1) >> 3
        last = nbits - 8 * full
        v = 0
        if full > 0:
            v = simlake_bswap64(getbits(buf, pos, 8 * full) << (64 - 8 * full))
        v |= getbits(buf, pos, last) << (8 * full)
        out[2] = <int64_t>(v % <uint64_t>sizes[2])
        v //= <uint64_t>sizes[2]
        out[1] = <int64_t>(v % <uint64_t>sizes[1])
        out[0] = <int64_t>(v // <uint64_t>sizes[1])
        return
    # More than 64 bits: long division one byte at a time.
    byts[1] = 0
    byts[2] = 0
    byts[3] = 0
    nbytes = 0
    while nbits > 8:
        byts[nbytes] = <int64_t>getbits(buf, pos, 8)
        nbytes += 1
        nbits -= 8
    if nbits > 0:
        byts[nbytes] = <int64_t>getbits(buf, pos, nbits)
        nbytes += 1
    for i in range(2, 0, -1):
        num = 0
        for j in range(nbytes - 1, -1, -1):
            num = (num << 8) | byts[j]
            p = num // sizes[i]
            byts[j] = p
            num = num - p * sizes[i]
        out[i] = num
    out[0] = byts[0] | (byts[1] << 8) | (byts[2] << 16) | (byts[3] << 24)


def decompress(const uint8_t[::1] buf, int natoms, const int64_t[::1] minint,
               const int64_t[::1] maxint, int smallidx, float precision, float[:, ::1] out):
    """Decode ``natoms`` coordinates from the compressed bytes into ``out`` (nm).

    Returns the number of atoms decoded. Releases the GIL.
    """
    cdef int64_t sizeint[3]
    cdef int bitsizeint[3]
    cdef int64_t sizesmall[3]
    cdef int64_t thiscoord[3]
    cdef int64_t prevcoord[3]
    cdef int64_t small[3]
    cdef int64_t tmp, smaller, smallnum
    cdef int bitsize, k, d, flag, is_smaller, run = 0, i = 0, atom = 0
    cdef uint64_t pos = 0
    cdef float inv_precision = <float>(1.0 / <double>precision)
    cdef const uint8_t *b = &buf[0]

    with nogil:
        for k in range(3):
            sizeint[k] = maxint[k] - minint[k] + 1
        if sizeint[0] > 0xFFFFFF or sizeint[1] > 0xFFFFFF or sizeint[2] > 0xFFFFFF:
            for k in range(3):
                bitsizeint[k] = sizeofint(sizeint[k])
            bitsize = 0
        else:
            bitsize = sizeofints(sizeint)

        smaller = SIMLAKE_MAGICINTS[max(FIRSTIDX, smallidx - 1)] // 2
        smallnum = SIMLAKE_MAGICINTS[smallidx] // 2
        for k in range(3):
            sizesmall[k] = SIMLAKE_MAGICINTS[smallidx]

        while i < natoms:
            if bitsize == 0:
                for k in range(3):
                    thiscoord[k] = <int64_t>getbits(b, &pos, bitsizeint[k])
            else:
                receiveints(b, &pos, bitsize, sizeint, thiscoord)
            i += 1
            for k in range(3):
                thiscoord[k] += minint[k]
                prevcoord[k] = thiscoord[k]

            flag = <int>getbits(b, &pos, 1)
            is_smaller = 0
            if flag == 1:
                run = <int>getbits(b, &pos, 5)  # run persists when flag == 0
                is_smaller = run % 3
                run -= is_smaller
                is_smaller -= 1

            if run > 0:
                for k in range(0, run, 3):
                    receiveints(b, &pos, smallidx, sizesmall, small)
                    i += 1
                    for d in range(3):
                        small[d] += prevcoord[d] - smallnum
                    if k == 0:
                        # the encoder swaps the first two atoms of a run (water: O after H)
                        for d in range(3):
                            tmp = small[d]
                            small[d] = prevcoord[d]
                            prevcoord[d] = tmp
                        for d in range(3):
                            out[atom, d] = <float>prevcoord[d] * inv_precision
                        atom += 1
                    else:
                        for d in range(3):
                            prevcoord[d] = small[d]
                    for d in range(3):
                        out[atom, d] = <float>small[d] * inv_precision
                    atom += 1
            else:
                for d in range(3):
                    out[atom, d] = <float>thiscoord[d] * inv_precision
                atom += 1

            smallidx += is_smaller
            if is_smaller < 0:
                smallnum = smaller
                if smallidx > FIRSTIDX:
                    smaller = SIMLAKE_MAGICINTS[smallidx - 1] // 2
                else:
                    smaller = 0
            elif is_smaller > 0:
                smaller = smallnum
                smallnum = SIMLAKE_MAGICINTS[smallidx] // 2
            for d in range(3):
                sizesmall[d] = SIMLAKE_MAGICINTS[smallidx]
    return atom
