import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hash_base import HashFunction
import struct

# ──────────────────────────────────────────────────────────────────────────────
# AquaHash-256  —  an ARX-based hash inspired by BLAKE3 / ChaCha
#
# Design:
#   • State: sixteen 32-bit words (512-bit wide), mirroring BLAKE3's internal
#     compression state.
#   • Mixing primitive: the BLAKE G function
#         a += b + m0;  d = ROT(d^a, 16)
#         c += d;       b = ROT(b^c, 12)
#         a += b + m1;  d = ROT(d^a,  8)
#         c += d;       b = ROT(b^c,  7)
#     — addition provides nonlinearity across bit positions
#     — XOR + rotation propagate changes across all bits (avalanche)
#   • 10 rounds of 8 column+diagonal G calls each (80 G calls total)
#   • Fixed IV: fractional parts of sqrt of first 8 primes (same as SHA-2/BLAKE)
#   • Message schedule: input padded to 64 bytes, loaded as 16 × uint32
#   • Output: XOR of upper and lower halves of final state → 256 bits
# ──────────────────────────────────────────────────────────────────────────────

MASK32 = 0xFFFFFFFF

# Initialization vector — first 32 bits of fractional parts of sqrt(prime_i)
_IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# Round constants — first 32 bits of fractional parts of cbrt(prime_i), primes 9-24
_RC = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
]

def _rot32(x: int, n: int) -> int:
    """Rotate x right by n bits (32-bit)."""
    return ((x >> n) | (x << (32 - n))) & MASK32

def _G(v, a, b, c, d, m0, m1):
    """BLAKE G mixing function — the core ARX primitive."""
    v[a] = (v[a] + v[b] + m0) & MASK32
    v[d] = _rot32(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d])      & MASK32
    v[b] = _rot32(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + m1) & MASK32
    v[d] = _rot32(v[d] ^ v[a],  8)
    v[c] = (v[c] + v[d])      & MASK32
    v[b] = _rot32(v[b] ^ v[c],  7)

# Message word permutation applied between rounds (same as BLAKE3)
_SIGMA = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

def _compress(block: bytes, chaining: list, counter: int) -> list:
    """
    One compression of a 64-byte block.
    chaining: list of 8 x uint32 chaining values (h0..h7)
    counter:  byte-count counter (split into lo/hi uint32)
    Returns updated chaining value as list of 8 x uint32.
    """
    # Pad block to exactly 64 bytes
    block = block.ljust(64, b'\x00')

    # Unpack 16 message words
    m = list(struct.unpack('<16I', block))

    # Initialize 16-word state
    # v[0..7]  = chaining value
    # v[8..11] = IV[0..3]
    # v[12]    = counter lo, v[13] = counter hi
    # v[14]    = block length (always 64 in our usage)
    # v[15]    = RC[15] (domain constant)
    v = list(chaining) + list(_IV[:4]) + [
        counter & MASK32,
        (counter >> 32) & MASK32,
        64,
        _RC[15],
    ]

    # 10 rounds of mixing (BLAKE3 uses 7; we use 10 for extra diffusion)
    for r in range(10):
        # Mix round constants into message schedule for nonlinearity
        ms = [(m[i] ^ _RC[(r + i) % 16]) & MASK32 for i in range(16)]

        # Column step (mixes within columns of the 4×4 state matrix)
        _G(v, 0, 4,  8, 12, ms[ 0], ms[ 1])
        _G(v, 1, 5,  9, 13, ms[ 2], ms[ 3])
        _G(v, 2, 6, 10, 14, ms[ 4], ms[ 5])
        _G(v, 3, 7, 11, 15, ms[ 6], ms[ 7])

        # Diagonal step (mixes across columns — kills linearity between columns)
        _G(v, 0, 5, 10, 15, ms[ 8], ms[ 9])
        _G(v, 1, 6, 11, 12, ms[10], ms[11])
        _G(v, 2, 7,  8, 13, ms[12], ms[13])
        _G(v, 3, 4,  9, 14, ms[14], ms[15])

        # Permute message words for next round
        m = [m[_SIGMA[i]] for i in range(16)]

    # Finalization: XOR upper and lower halves into chaining value
    return [(chaining[i] ^ v[i] ^ v[i + 8]) & MASK32 for i in range(8)]


class MyHash(HashFunction):
    """
    AquaHash-256 — ARX hash inspired by BLAKE3 / ChaCha.

    Key properties:
      • Nonlinear mixing via G (add + XOR + rotate)
      • 10 rounds of column + diagonal mixing per block
      • Round-keyed message schedule (RC injection)
      • Merkle-Damgård chaining over 64-byte blocks
      • 256-bit output (8 × uint32)
    """

    @property
    def name(self) -> str:
        return "AquaHash-256"

    @property
    def digest_size(self) -> int:
        return 32

    def _compress(self, data: bytes) -> bytes:
        # ── Initialization ────────────────────────────────────────────────────
        h = list(_IV)           # 8-word chaining value
        length = len(data)
        counter = 0             # byte-offset counter

        # ── Padding ───────────────────────────────────────────────────────────
        # Append 0x80, then zeros, then 64-bit little-endian length
        # so the padded message is a multiple of 64 bytes.
        data += b'\x80'
        pad_len = (55 - len(data) % 64) % 64
        data += b'\x00' * pad_len
        data += struct.pack('<Q', length * 8)   # bit-length in last 8 bytes

        # ── Process 64-byte blocks ────────────────────────────────────────────
        for i in range(0, len(data), 64):
            block = data[i:i + 64]
            h = _compress(block, h, counter)
            counter += 64

        # ── Serialize to 32 bytes ─────────────────────────────────────────────
        return struct.pack('<8I', *h)