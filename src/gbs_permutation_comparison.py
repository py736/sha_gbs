

try:
    import scipy
    import numpy as np
    import scipy.integrate
    if not hasattr(scipy.integrate, "simps"):
        scipy.integrate.simps = scipy.integrate.simpson

    import strawberryfields as sf
    from strawberryfields import ops
    print("strawberryfields available: probabilistic GBS sampling (Cell 15) will run.")
except Exception as e:
    print("strawberryfields NOT available (", repr(e), ") - probabilistic GBS sampling ")



# %pip install nistrng

import os
import sys
import time
import math
import json
import hashlib
import random
import struct
import platform
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

SEED = 20260825
random.seed(SEED)
np.random.seed(SEED)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Platform:", platform.platform())

CFG = {
    "avalanche_samples": 1000,
    "sac_samples": 1000,
    "bic_samples": 1000,
    "differential_samples": 2000,
    "linear_samples": 5000,
    "performance_trials": 200,
    "performance_warmup": 20,
    "round_diffusion_samples": 100,
    "structural_samples": 2000,
    "frequency_bytes": 1_000_000,
    "gbs_modes": 8,
    "gbs_interferometer_depth": 4,
    "gbs_feistel_rounds": 8,  # retained configuration key; now used as GBS-1600 rounds
    "gbs_squeezing_r": 0.35,
    "gbs_samples": 5000,
    # NEW: cryptographic state size of the GBS-derived candidate.
    "gbs_state_bits": 1600,
    "gbs_state_bytes": 200,
    "gbs_bit_permutation_multiplier": 17,
    "gbs_bit_permutation_offset": 1,
    "gbs_validation_samples": 10_000,
    "truncated_attack_bits": 16,
    "truncated_attack_max_queries": 200000,
}
print(json.dumps(CFG, indent=2))

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1

def rol32(x, n):
    x &= MASK32
    n %= 32
    return x if n == 0 else ((x << n) | (x >> (32-n))) & MASK32

def ror32(x, n):
    x &= MASK32
    n %= 32
    return x if n == 0 else ((x >> n) | (x << (32-n))) & MASK32

def rol64(x, n):
    x &= MASK64
    n %= 64
    return x if n == 0 else ((x << n) | (x >> (64-n))) & MASK64

def ror64(x, n):
    x &= MASK64
    n %= 64
    return x if n == 0 else ((x >> n) | (x << (64-n))) & MASK64

def u32_list(b):
    if len(b) % 4:
        raise ValueError("Length must be divisible by 4")
    return list(struct.unpack("<" + "I"*(len(b)//4), b))

def u64_list(b):
    if len(b) % 8:
        raise ValueError("Length must be divisible by 8")
    return list(struct.unpack("<" + "Q"*(len(b)//8), b))

def u32_bytes(x):
    return struct.pack("<" + "I"*len(x), *[(v & MASK32) for v in x])

def u64_bytes(x):
    return struct.pack("<" + "Q"*len(x), *[(v & MASK64) for v in x])

def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Equal lengths required")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))

def flip_bit(data, bit):
    x = bytearray(data)
    x[bit // 8] ^= 1 << (bit % 8)
    return bytes(x)

def random_bytes(n):
    return np.random.bytes(n)

KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
]

KECCAK_R = [
    [0,36,3,41,18],
    [1,44,10,45,2],
    [62,6,43,15,61],
    [28,55,25,21,56],
    [27,20,39,8,14],
]

def keccak_round(a, rc):
    c = [a[x] ^ a[x+5] ^ a[x+10] ^ a[x+15] ^ a[x+20] for x in range(5)]
    d = [c[(x-1)%5] ^ rol64(c[(x+1)%5], 1) for x in range(5)]
    for x in range(5):
        for y in range(5):
            a[x+5*y] = (a[x+5*y] ^ d[x]) & MASK64

    b = [0] * 25
    for x in range(5):
        for y in range(5):
            b[y + 5*((2*x + 3*y) % 5)] = rol64(a[x+5*y], KECCAK_R[x][y])

    for x in range(5):
        for y in range(5):
            a[x+5*y] = (
                b[x+5*y] ^
                ((~b[(x+1)%5 + 5*y]) & b[(x+2)%5 + 5*y])
            ) & MASK64

    a[0] ^= rc
    return a

def keccak(state, rounds=24, first_rounds=False):
    if len(state) != 200:
        raise ValueError("Keccak-f[1600] requires 200 bytes")
    if not 0 <= rounds <= 24:
        raise ValueError("rounds must be in [0,24]")

    a = u64_list(state)
    constants = KECCAK_RC[:rounds] if first_rounds else KECCAK_RC[24-rounds:]
    for rc in constants:
        keccak_round(a, rc)
    return u64_bytes(a)

def sha3_256_reference(message):
    rate = 136
    state = bytearray(200)

    p = bytearray(message)
    p.append(0x06)
    while len(p) % rate != rate - 1:
        p.append(0)
    p.append(0x80)

    for off in range(0, len(p), rate):
        block = p[off:off+rate]
        for i, v in enumerate(block):
            state[i] ^= v
        state = bytearray(keccak(bytes(state), 24))

    return bytes(state[:32])

def shake256_reference(message, output_len):
    rate = 136
    state = bytearray(200)

    p = bytearray(message)
    p.append(0x1F)
    while len(p) % rate != rate - 1:
        p.append(0)
    p.append(0x80)

    for off in range(0, len(p), rate):
        block = p[off:off+rate]
        for i, v in enumerate(block):
            state[i] ^= v
        state = bytearray(keccak(bytes(state), 24))

    out = bytearray()
    while len(out) < output_len:
        out.extend(state[:rate])
        if len(out) < output_len:
            state = bytearray(keccak(bytes(state), 24))
    return bytes(out[:output_len])

for m in [b"", b"abc", b"The quick brown fox jumps over the lazy dog"]:
    print(len(m), sha3_256_reference(m).hex() == hashlib.sha3_256(m).hexdigest() and True)

# F2 — SHAKE-256 XOF validation

shake_checks = []
for m in [b"", b"abc", b"The quick brown fox jumps over the lazy dog"]:
    for out_len in [32, 64]:
        ours = shake256_reference(m, out_len)
        ref = hashlib.shake_256(m).digest(out_len)
        shake_checks.append({
            "message_len": len(m),
            "output_len": out_len,
            "matches_hashlib": ours == ref,
        })

shake_df = pd.DataFrame(shake_checks)
display(shake_df)
print("SHAKE-256 all match:", bool(shake_df["matches_hashlib"].all()))

XOODOO_RC = [
    0x00000058, 0x00000038, 0x000003C0, 0x000000D0,
    0x00000120, 0x00000014, 0x00000060, 0x0000002C,
    0x00000380, 0x000000F0, 0x000001A0, 0x00000012,
]

def xoodoo_round(a, rc):
    # a[x + 4*y], x=0..3, y=0..2
    p = [a[x] ^ a[x+4] ^ a[x+8] for x in range(4)]
    e = [p[(x-1)%4] ^ rol32(p[(x-1)%4],5) ^ rol32(p[(x-1)%4],14) for x in range(4)]

    b = a[:]
    for x in range(4):
        for y in range(3):
            b[x+4*y] = a[x+4*y] ^ e[x]

    # rho-west
    a = [0]*12
    for x in range(4):
        a[x] = b[x]
        a[(x+1)%4 + 4] = b[x+4]
        a[x+8] = rol32(b[x+8], 11)

    a[0] ^= rc

    # chi
    b = a[:]
    for x in range(4):
        for y in range(3):
            a[x+4*y] = b[x+4*y] ^ ((~b[x+4*((y+1)%3)]) & b[x+4*((y+2)%3)])
            a[x+4*y] &= MASK32

    # rho-east
    b = a[:]
    a = [0]*12
    for x in range(4):
        a[x] = b[x]
        a[x+4] = rol32(b[x+4], 1)
        a[(x+2)%4 + 8] = rol32(b[x+8], 8)

    return [v & MASK32 for v in a]

def xoodoo(state, rounds=12, first_rounds=False):
    if len(state) != 48:
        raise ValueError("Xoodoo requires 48 bytes")
    if not 0 <= rounds <= 12:
        raise ValueError("rounds must be in [0,12]")

    a = u32_list(state)
    constants = XOODOO_RC[:rounds] if first_rounds else XOODOO_RC[12-rounds:]
    for rc in constants:
        a = xoodoo_round(a, rc)
    return u32_bytes(a)

ASCON_RC_12 = [
    0x00000000000000F0,
    0x00000000000000E1,
    0x00000000000000D2,
    0x00000000000000C3,
    0x00000000000000B4,
    0x00000000000000A5,
    0x0000000000000096,
    0x0000000000000087,
    0x0000000000000078,
    0x0000000000000069,
    0x000000000000005A,
    0x000000000000004B,
]

def ascon_round(s, rc):
    x0,x1,x2,x3,x4 = s

    x2 ^= rc

    x0 ^= x4
    x4 ^= x3
    x2 ^= x1

    t0 = (~x0) & x1
    t1 = (~x1) & x2
    t2 = (~x2) & x3
    t3 = (~x3) & x4
    t4 = (~x4) & x0

    x0 ^= t1
    x1 ^= t2
    x2 ^= t3
    x3 ^= t4
    x4 ^= t0

    x1 ^= x0
    x0 ^= x4
    x3 ^= x2
    x2 = (~x2) & MASK64

    x0 ^= ror64(x0,19) ^ ror64(x0,28)
    x1 ^= ror64(x1,61) ^ ror64(x1,39)
    x2 ^= ror64(x2,1)  ^ ror64(x2,6)
    x3 ^= ror64(x3,10) ^ ror64(x3,17)
    x4 ^= ror64(x4,7)  ^ ror64(x4,41)

    return [v & MASK64 for v in (x0,x1,x2,x3,x4)]

def ascon(state, rounds=12, first_rounds=False):
    if len(state) != 40:
        raise ValueError("Ascon requires 40 bytes")
    if not 0 <= rounds <= 12:
        raise ValueError("rounds must be in [0,12]")

    s = u64_list(state)
    constants = ASCON_RC_12[:rounds] if first_rounds else ASCON_RC_12[12-rounds:]
    for rc in constants:
        s = ascon_round(s, rc)
    return u64_bytes(s)

GIMLI_KAT_WORDS_IN = [
    0x00000000, 0x9e3779ba, 0x3c6ef37a, 0xdaa66d46,
    0x78dde724, 0x1715611a, 0xb54cdb2e, 0x53845566,
    0xf1bbcfc8, 0x8ff34a5a, 0x2e2ac522, 0xcc624026,
]

GIMLI_KAT_WORDS_OUT = [
    0xba11c85a, 0x91bad119, 0x380ce880, 0xd24c2c68,
    0x3eceffea, 0x277a921c, 0x4f73a0bd, 0xda5a9cd8,
    0x84b673f0, 0x34e52ff7, 0x9e2bef49, 0xf41bb8d6,
]


def gimli(state, rounds=24, first_rounds=False):
    if len(state) != 48:
        raise ValueError("Gimli requires 48 bytes")
    if not 0 <= rounds <= 24:
        raise ValueError("rounds must be in [0,24]")

    s = u32_list(state)

    round_numbers = range(24, 24-rounds, -1) if first_rounds else range(rounds, 0, -1)

    for r in round_numbers:
        for j in range(4):
            x = rol32(s[j], 24)
            y = rol32(s[4+j], 9)
            z = s[8+j]

            s[8+j] = (x ^ (z << 1) ^ ((y & z) << 2)) & MASK32
            s[4+j] = (y ^ x ^ ((x | z) << 1)) & MASK32
            s[j] = (z ^ y ^ ((x & y) << 3)) & MASK32

        if (r & 3) == 0:
            s[0],s[1] = s[1],s[0]
            s[2],s[3] = s[3],s[2]
            s[0] ^= (0x9E377900 | r)

        if (r & 3) == 2:
            s[0],s[2] = s[2],s[0]
            s[1],s[3] = s[3],s[1]

    return u32_bytes(s)


gimli_input = u32_bytes(GIMLI_KAT_WORDS_IN)
gimli_output = gimli(gimli_input)
gimli_output_words = u32_list(gimli_output)

print(
    "Gimli KAT:",
    gimli_output_words == GIMLI_KAT_WORDS_OUT
)

def chacha_qr(x, a,b,c,d):
    x[a] = (x[a] + x[b]) & MASK32
    x[d] = rol32(x[d] ^ x[a],16)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] = rol32(x[b] ^ x[c],12)
    x[a] = (x[a] + x[b]) & MASK32
    x[d] = rol32(x[d] ^ x[a],8)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] = rol32(x[b] ^ x[c],7)

CHACHA_QUARTERS = [
    (0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15),
    (0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14),
]

def chacha(state, rounds=20, first_rounds=False):
    if len(state) != 64:
        raise ValueError("ChaCha requires 64 bytes")
    if rounds < 0 or rounds % 2:
        raise ValueError("ChaCha rounds must be a non-negative even number")

    x = u32_list(state)
    for _ in range(rounds//2):
        for q in CHACHA_QUARTERS:
            chacha_qr(x,*q)
    return u32_bytes(x)

BLAKE2B_IV = [
    0x6A09E667F3BCC908,0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,0x5BE0CD19137E2179,
]

BLAKE2B_SIGMA = [
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
    [11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4],
    [7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8],
    [9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13],
    [2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9],
    [12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11],
    [13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10],
    [6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5],
    [10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0],

    # BLAKE2b rounds 10 and 11 repeat SIGMA[0] and SIGMA[1].
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
]

assert len(BLAKE2B_SIGMA) == 12
assert BLAKE2B_SIGMA[10] == BLAKE2B_SIGMA[0]
assert BLAKE2B_SIGMA[11] == BLAKE2B_SIGMA[1]

def blake2b_g(v,a,b,c,d,x,y):
    v[a]=(v[a]+v[b]+x)&MASK64
    v[d]=ror64(v[d]^v[a],32)
    v[c]=(v[c]+v[d])&MASK64
    v[b]=ror64(v[b]^v[c],24)
    v[a]=(v[a]+v[b]+y)&MASK64
    v[d]=ror64(v[d]^v[a],16)
    v[c]=(v[c]+v[d])&MASK64
    v[b]=ror64(v[b]^v[c],63)

def blake2b_compression(h, block, t0=0, t1=0, final=False, final2=False):
    if len(h)!=8:
        raise ValueError("h must contain 8 uint64 words")
    if len(block)!=128:
        raise ValueError("block must be 128 bytes")

    m=u64_list(block)
    v=h[:] + BLAKE2B_IV[:]
    v[12] ^= t0 & MASK64
    v[13] ^= t1 & MASK64
    if final: v[14] ^= MASK64
    if final2: v[15] ^= MASK64

    for s in BLAKE2B_SIGMA:
        blake2b_g(v,0,4,8,12,m[s[0]],m[s[1]])
        blake2b_g(v,1,5,9,13,m[s[2]],m[s[3]])
        blake2b_g(v,2,6,10,14,m[s[4]],m[s[5]])
        blake2b_g(v,3,7,11,15,m[s[6]],m[s[7]])
        blake2b_g(v,0,5,10,15,m[s[8]],m[s[9]])
        blake2b_g(v,1,6,11,12,m[s[10]],m[s[11]])
        blake2b_g(v,2,7,8,13,m[s[12]],m[s[13]])
        blake2b_g(v,3,4,9,14,m[s[14]],m[s[15]])

    return [h[i]^v[i]^v[i+8] for i in range(8)]

def blake2b_eval(block):
    h=BLAKE2B_IV[:]
    return u64_bytes(blake2b_compression(h,block))

print("BLAKE2b SIGMA rounds:", len(BLAKE2B_SIGMA))
print("BLAKE2b round schedule: PASS")

BLAKE3_IV = [
    0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
    0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19
]

BLAKE3_PERM = [2,6,3,10,7,0,4,15,12,1,13,14,11,5,8,9]

def blake3_g(v,a,b,c,d,x,y):
    v[a]=(v[a]+v[b]+x)&MASK32
    v[d]=ror32(v[d]^v[a],16)
    v[c]=(v[c]+v[d])&MASK32
    v[b]=ror32(v[b]^v[c],12)
    v[a]=(v[a]+v[b]+y)&MASK32
    v[d]=ror32(v[d]^v[a],8)
    v[c]=(v[c]+v[d])&MASK32
    v[b]=ror32(v[b]^v[c],7)

def blake3_round(v,m):
    blake3_g(v,0,4,8,12,m[0],m[1])
    blake3_g(v,1,5,9,13,m[2],m[3])
    blake3_g(v,2,6,10,14,m[4],m[5])
    blake3_g(v,3,7,11,15,m[6],m[7])
    blake3_g(v,0,5,10,15,m[8],m[9])
    blake3_g(v,1,6,11,12,m[10],m[11])
    blake3_g(v,2,7,8,13,m[12],m[13])
    blake3_g(v,3,4,9,14,m[14],m[15])

def blake3_compression(cv, block, counter=0, block_len=64, flags=0):
    if len(cv)!=8:
        raise ValueError("cv must contain 8 words")
    if len(block)!=64:
        raise ValueError("block must be 64 bytes")

    m=u32_list(block)
    v=cv[:] + BLAKE3_IV[:4] + [
        counter & MASK32, (counter>>32)&MASK32,
        block_len & MASK32, flags & MASK32
    ]

    for _ in range(7):
        blake3_round(v,m)
        m=[m[i] for i in BLAKE3_PERM]

    out=[(v[i]^v[i+8])&MASK32 for i in range(8)]
    out += [(v[i+8]^cv[i])&MASK32 for i in range(8)]
    return out

def blake3_eval(block):
    return u32_bytes(blake3_compression(BLAKE3_IV,block))

# ------------------------------------------------------------
# Haraka-style AES experiment (dependency-free)
# ------------------------------------------------------------

_AES_SBOX = [
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16]

_AES_RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

def _aes_xtime(a):
    a <<= 1
    if a & 0x100:
        a ^= 0x11b
    return a & 0xff

def _aes_key_expansion_128(key):
    Nk, Nr = 4, 10
    w = [list(key[4*i:4*i+4]) for i in range(Nk)]
    for i in range(Nk, 4*(Nr+1)):
        temp = list(w[i-1])
        if i % Nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [_AES_SBOX[b] for b in temp]
            temp[0] ^= _AES_RCON[i//Nk - 1]
        w.append([a ^ b for a, b in zip(w[i-Nk], temp)])
    round_keys = []
    for r in range(Nr+1):
        rk = []
        for c in range(4):
            rk += w[4*r+c]
        round_keys.append(rk)
    return round_keys

def _aes_add_round_key(state, rk):
    return [s ^ k for s, k in zip(state, rk)]

def _aes_sub_bytes(state):
    return [_AES_SBOX[b] for b in state]

def _aes_shift_rows(state):
    s = state[:]
    out = [0]*16
    for col in range(4):
        for row in range(4):
            out[row + 4*col] = s[row + 4*((col+row) % 4)]
    return out

def _aes_mix_single_column(a):
    r = [0]*4
    r[0] = _aes_xtime(a[0]) ^ (_aes_xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3]
    r[1] = a[0] ^ _aes_xtime(a[1]) ^ (_aes_xtime(a[2]) ^ a[2]) ^ a[3]
    r[2] = a[0] ^ a[1] ^ _aes_xtime(a[2]) ^ (_aes_xtime(a[3]) ^ a[3])
    r[3] = (_aes_xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ _aes_xtime(a[3])
    return r

def _aes_mix_columns(state):
    out = [0]*16
    for c in range(4):
        col = state[4*c:4*c+4]
        out[4*c:4*c+4] = _aes_mix_single_column(col)
    return out

def aes128_encrypt_block(block, key):
    round_keys = _aes_key_expansion_128(key)
    state = list(block)
    state = _aes_add_round_key(state, round_keys[0])
    for r in range(1, 10):
        state = _aes_sub_bytes(state)
        state = _aes_shift_rows(state)
        state = _aes_mix_columns(state)
        state = _aes_add_round_key(state, round_keys[r])
    state = _aes_sub_bytes(state)
    state = _aes_shift_rows(state)
    state = _aes_add_round_key(state, round_keys[10])
    return bytes(state)
_AES_SELFTEST = aes128_encrypt_block(
    bytes.fromhex("00112233445566778899aabbccddeeff"),
    bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
) == bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
print("Pure-Python AES-128 self-test (FIPS-197 vector):", _AES_SELFTEST)
if not _AES_SELFTEST:
    raise RuntimeError("AES self-test failed - aborting Haraka-style experiment")

HAVE_AES = True

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def haraka_style(state, rounds=5):
    if len(state) not in (32, 64):
        raise ValueError("Haraka-style experiment requires 32 or 64 bytes")

    lanes = [state[i:i+16] for i in range(0, len(state), 16)]

    for r in range(rounds):
        new = []
        for i, x in enumerate(lanes):
            key = hashlib.sha256(
                b"HARAKA-EXPERIMENT" + bytes([r, i])
            ).digest()[:16]
            new.append(aes128_encrypt_block(x, key))
        lanes = [xor_bytes(new[i], new[(i+1) % len(new)]) for i in range(len(new))]

    return b"".join(lanes)

def beam_splitter(theta, phi):
    c=np.cos(theta)
    s=np.sin(theta)
    return np.array([
        [c, np.exp(1j*phi)*s],
        [-np.exp(-1j*phi)*s, c]
    ],dtype=complex)


def build_interferometer_unitary(n_modes=8, depth=4, seed=20260825):
    rng=np.random.default_rng(seed)
    U=np.eye(n_modes,dtype=complex)

    for layer in range(depth):
        offset=layer % 2
        for i in range(offset,n_modes-1,2):
            theta=rng.uniform(0,np.pi/2)
            phi=rng.uniform(0,2*np.pi)
            B=beam_splitter(theta,phi)

            M=np.eye(n_modes,dtype=complex)
            M[np.ix_([i,i+1],[i,i+1])]=B
            U=M@U

    err=np.max(np.abs(U.conj().T@U-np.eye(n_modes)))
    if err>1e-10:
        raise RuntimeError(f"Interferometer is not unitary: error={err}")

    return U


def hafnian(A):
    n=A.shape[0]
    if n==0:
        return 1.0+0j
    if n%2:
        return 0.0+0j

    def rec(M):
        m=M.shape[0]
        if m==0:
            return 1.0+0j

        total=0.0+0j
        for j in range(1,m):
            keep=[k for k in range(m) if k not in (0,j)]
            total += M[0,j]*rec(M[np.ix_(keep,keep)])
        return total

    return rec(A)


def gbs_round_sbox(A4, seed):

    rng=np.random.default_rng(seed)
    scores=[]

    for x in range(16):
        inds=[i for i in range(4) if (x>>i)&1]

        if len(inds)>=2 and len(inds)%2==0:
            sub=A4[np.ix_(inds,inds)]
            score=float(abs(hafnian(sub))**2)
        else:
            sub=A4[np.ix_(inds,inds)]
            score=float(np.sum(np.abs(sub)**2))

        score += 1e-15*float(rng.random())
        scores.append(score)

    order=sorted(range(16),key=lambda x:(scores[x],x))

    sbox=[0]*16
    for rank,x in enumerate(order):
        sbox[x]=rank

    return sbox,scores


class GBSPhotonicPermutation:


    def __init__(self, n_modes=8, depth=4, feistel_rounds=8,
                 squeezing_r=0.35, seed=20260825,
                 state_bits=1600, bit_multiplier=17, bit_offset=1):
        if n_modes != 8:
            raise ValueError("The current exact GBS S-box generator uses exactly 8 photonic modes")
        if feistel_rounds < 1:
            raise ValueError("round count must be positive")
        if state_bits != 1600:
            raise ValueError("This updated candidate is defined for a 1600-bit state")
        if state_bits % 8 != 0 or state_bits % 4 != 0:
            raise ValueError("state_bits must be divisible by 8 and 4")
        if math.gcd(bit_multiplier, state_bits) != 1:
            raise ValueError("bit_multiplier must be coprime with state_bits")

        self.n=n_modes
        self.depth=depth
        self.rounds=feistel_rounds
        self.squeezing_r=squeezing_r
        self.seed=seed
        self.state_bits=state_bits
        self.state_bytes=state_bits//8
        self.n_nibbles=state_bits//4
        self.bit_multiplier=bit_multiplier
        self.bit_offset=bit_offset % state_bits
        self.inverse_multiplier=pow(bit_multiplier, -1, state_bits)

        self.unitary=build_interferometer_unitary(n_modes,depth,seed)
        self.A=np.tanh(squeezing_r)*(self.unitary@self.unitary.T)

        self.sboxes=[]
        self.inverse_sboxes=[]
        self.scores=[]
        for r in range(feistel_rounds):
            idx=[(2*i+r)%n_modes for i in range(4)]
            A4=self.A[np.ix_(idx,idx)]
            sbox,scores=gbs_round_sbox(A4,seed+r)
            self.sboxes.append(sbox)
            inv=[0]*16
            for x,y in enumerate(sbox):
                inv[y]=x
            self.inverse_sboxes.append(inv)
            self.scores.append(scores)

        self.bit_perm=[(self.bit_multiplier*i+self.bit_offset)%self.state_bits
                       for i in range(self.state_bits)]
        self.inv_bit_perm=[0]*self.state_bits
        for i,p in enumerate(self.bit_perm):
            self.inv_bit_perm[p]=i

        self.round_constants=[]
        for r in range(feistel_rounds):
            digest=hashlib.sha256(
                f"GBS1600|round|{seed}|{r}".encode("utf-8")
            ).digest()
            rc=bytearray()
            counter=0
            while len(rc)<self.state_bytes:
                rc.extend(hashlib.sha256(
                    digest + counter.to_bytes(4,"little")
                ).digest())
                counter+=1
            self.round_constants.append(bytes(rc[:self.state_bytes]))

    def _sbox_layer(self, state, sbox):
        out=bytearray(self.state_bytes)
        for i,v in enumerate(state):
            out[i]=(sbox[v & 0x0F] | (sbox[(v >> 4) & 0x0F] << 4))
        return bytes(out)

    def _inv_sbox_layer(self, state, inv):
        out=bytearray(self.state_bytes)
        for i,v in enumerate(state):
            out[i]=(inv[v & 0x0F] | (inv[(v >> 4) & 0x0F] << 4))
        return bytes(out)

    @staticmethod
    def _prefix_xor_bytes(state):
        out=bytearray(state)
        for i in range(1,len(out)):
            out[i] ^= out[i-1]
        return bytes(out)

    @staticmethod
    def _inverse_prefix_xor_bytes(state):
        out=bytearray(state)
        for i in range(len(out)-1,0,-1):
            out[i] ^= out[i-1]
        return bytes(out)

    def _bit_permute(self, state):
        out=bytearray(self.state_bytes)
        for i in range(self.state_bits):
            if (state[i//8] >> (i%8)) & 1:
                p=self.bit_perm[i]
                out[p//8] |= 1 << (p%8)
        return bytes(out)

    def _inverse_bit_permute(self, state):
        out=bytearray(self.state_bytes)
        for p in range(self.state_bits):
            if (state[p//8] >> (p%8)) & 1:
                i=self.inv_bit_perm[p]
                out[i//8] |= 1 << (i%8)
        return bytes(out)

    def _round(self, state, r):
        state=self._sbox_layer(state,self.sboxes[r])
        state=bytes(a^b for a,b in zip(state,self.round_constants[r]))
        state=self._prefix_xor_bytes(state)
        state=self._bit_permute(state)
        return state

    def _inverse_round(self, state, r):
        state=self._inverse_bit_permute(state)
        state=self._inverse_prefix_xor_bytes(state)
        state=bytes(a^b for a,b in zip(state,self.round_constants[r]))
        state=self._inv_sbox_layer(state,self.inverse_sboxes[r])
        return state

    def permutation_bytes(self, state, rounds=None):
        if len(state)!=self.state_bytes:
            raise ValueError(f"GBS-1600 requires exactly {self.state_bytes} bytes")
        if rounds is None:
            rounds=self.rounds
        if not 0 <= rounds <= self.rounds:
            raise ValueError(f"rounds must be in [0,{self.rounds}]")
        out=bytes(state)
        for r in range(rounds):
            out=self._round(out,r)
        return out

    def inverse_bytes(self, state, rounds=None):
        if len(state)!=self.state_bytes:
            raise ValueError(f"GBS-1600 requires exactly {self.state_bytes} bytes")
        if rounds is None:
            rounds=self.rounds
        if not 0 <= rounds <= self.rounds:
            raise ValueError(f"rounds must be in [0,{self.rounds}]")
        out=bytes(state)
        for r in range(rounds-1,-1,-1):
            out=self._inverse_round(out,r)
        return out

    def permutation_int(self,x,rounds=None):
        if not 0 <= x < (1<<self.state_bits):
            raise ValueError("Input must be a 1600-bit integer")
        return int.from_bytes(self.permutation_bytes(x.to_bytes(self.state_bytes,"little"),rounds),"little")

    def inverse_int(self,y,rounds=None):
        if not 0 <= y < (1<<self.state_bits):
            raise ValueError("Input must be a 1600-bit integer")
        return int.from_bytes(self.inverse_bytes(y.to_bytes(self.state_bytes,"little"),rounds),"little")

    def permutation(self,bitstring):
        if len(bitstring)!=self.state_bits or any(c not in "01" for c in bitstring):
            raise ValueError(f"bitstring must contain exactly {self.state_bits} binary characters")
        x=int(bitstring,2)
        y=self.permutation_int(x)
        return format(y,f"0{self.state_bits}b")


GBS_P=GBSPhotonicPermutation(
    n_modes=CFG["gbs_modes"],
    depth=CFG["gbs_interferometer_depth"],
    feistel_rounds=CFG["gbs_feistel_rounds"],
    squeezing_r=CFG["gbs_squeezing_r"],
    seed=SEED,
    state_bits=CFG["gbs_state_bits"],
    bit_multiplier=CFG["gbs_bit_permutation_multiplier"],
    bit_offset=CFG["gbs_bit_permutation_offset"],
)

print("Photonic GBS modes:",GBS_P.n)
print("Cryptographic state bits:",GBS_P.state_bits)
print("Cryptographic state bytes:",GBS_P.state_bytes)
print("Rounds:",GBS_P.rounds)
print("Unitary shape:",GBS_P.unitary.shape)
print("Unitary error:",np.max(np.abs(GBS_P.unitary.conj().T@GBS_P.unitary-np.eye(GBS_P.n))))
print("Bit permutation gcd:",math.gcd(GBS_P.bit_multiplier,GBS_P.state_bits))
print("Example output prefix:",GBS_P.permutation_bytes(bytes(200))[:16].hex())

gbs_validation_rng=np.random.default_rng(SEED)
gbs_validation_rows=[]
for i in range(CFG["gbs_validation_samples"]):
    x=bytes(gbs_validation_rng.bytes(GBS_P.state_bytes))
    y=GBS_P.permutation_bytes(x)
    xr=GBS_P.inverse_bytes(y)
    yr=GBS_P.permutation_bytes(xr)
    gbs_validation_rows.append({
        "sample":i,
        "input_equals_inverse_output":xr==x,
        "inverse_output_equals_repermuted_input":yr==y,
    })

gbs_validation_df=pd.DataFrame(gbs_validation_rows)
print("State bits:",GBS_P.state_bits)
print("Validation samples:",len(gbs_validation_df))
print("Inverse correct for all samples:",bool(gbs_validation_df["input_equals_inverse_output"].all()))
print("Round-trip output correct for all samples:",bool(gbs_validation_df["inverse_output_equals_repermuted_input"].all()))

# Small reproducible mapping sample for inspection/export; NOT exhaustive.
gbs_sample_inputs=[]
gbs_sample_outputs=[]
for x in [0,1,2,3,255,256,(1<<1599),(1<<1600)-1]:
    y=GBS_P.permutation_int(x)
    gbs_sample_inputs.append(x)
    gbs_sample_outputs.append(y)

gbs_perm_table=pd.DataFrame({
    "input_hex":[f"0{x:0400x}"[-400:] for x in gbs_sample_inputs],
    "output_hex":[f"0{y:0400x}"[-400:] for y in gbs_sample_outputs],
})
display(gbs_perm_table)

# Cell 15 — Actual probabilistic 8-mode GBS sampling

GBS_SAMPLING_AVAILABLE = False
actual_gbs_samples = None

try:
    import networkx as nx
    from strawberryfields.apps import sample as gbs_sample

    rng = np.random.default_rng(SEED)
    M = rng.random((CFG["gbs_modes"], CFG["gbs_modes"]))
    A_graph = (M + M.T) / 2
    np.fill_diagonal(A_graph, 0)
    A_graph = (A_graph > np.quantile(A_graph, 0.70)).astype(float)
    np.fill_diagonal(A_graph, 0)

    actual_gbs_samples = np.asarray(
        gbs_sample.sample(
            A_graph, 4.0,
            n_samples=CFG["gbs_samples"],
            threshold=True, loss=0.0,
        ),
        dtype=np.int8
    )
    GBS_SAMPLING_AVAILABLE = True
    print("GBS sample shape:", actual_gbs_samples.shape)
    print("Photonic sampling modes:", CFG["gbs_modes"], "(auxiliary; cryptographic state is 1600 bits)")

except Exception as e:
    print("GBS probabilistic sampling SKIPPED (strawberryfields unavailable or failed):", repr(e))
    print("This does not affect the deterministic 1600-bit GBS-derived permutation.")

if GBS_SAMPLING_AVAILABLE:
    tuples=[tuple(r.tolist()) for r in actual_gbs_samples]
    cnt=Counter(tuples)

    gbs_sampling_df=pd.DataFrame([{
        "modes":CFG["gbs_modes"],
        "samples":CFG["gbs_samples"],
        "cryptographic_state_bits":CFG["gbs_state_bits"],
        "unique_threshold_samples":len(cnt),
        "collision_rate":1-len(cnt)/CFG["gbs_samples"],
        "mean_clicks":actual_gbs_samples.sum(axis=1).mean(),
        "std_clicks":actual_gbs_samples.sum(axis=1).std(ddof=1),
    }])

    display(gbs_sampling_df)
else:
    print("GBS sample statistics SKIPPED (probabilistic sampling unavailable — see Cell 15 above).")

PERMUTATIONS={
    "Keccak-f[1600]":{
        "fn":keccak,"state_bytes":200,"state_bits":1600,"rounds":24,"type":"permutation",
    },
    "Xoodoo":{
        "fn":xoodoo,"state_bytes":48,"state_bits":384,"rounds":12,"type":"permutation",
    },
    "Ascon-p[320]":{
        "fn":ascon,"state_bytes":40,"state_bits":320,"rounds":12,"type":"permutation",
    },
    "Gimli":{
        "fn":gimli,"state_bytes":48,"state_bits":384,"rounds":24,"type":"permutation",
    },
    "ChaCha":{
        "fn":chacha,"state_bytes":64,"state_bits":512,"rounds":20,"type":"ARX core",
    },
    "BLAKE2b-compression":{
        "fn":blake2b_eval,"state_bytes":128,"state_bits":1024,
        "output_bytes":64,"output_bits":512,"rounds":12,"type":"compression function",
    },
    "BLAKE3-compression":{
        "fn":blake3_eval,"state_bytes":64,"state_bits":512,"rounds":7,"type":"compression function",
    },
    "Haraka-style-256":{
        "fn":haraka_style,"state_bytes":32,"state_bits":256,"rounds":5,"type":"AES-based experiment",
    },
    "Haraka-style-512":{
        "fn":haraka_style,"state_bytes":64,"state_bits":512,"rounds":5,"type":"AES-based experiment",
    },
    "GBS-photonic":{
        "fn":lambda b,rounds=8: GBS_P.permutation_bytes(b,rounds),
        "state_bytes":CFG["gbs_state_bytes"],
        "state_bits":CFG["gbs_state_bits"],
        "rounds":CFG["gbs_feistel_rounds"],
        "type":"GBS-derived 1600-bit research permutation",
        "gbs":True,
    },
}

registry_df=pd.DataFrame([
    {
        "algorithm":name,
        "state_bytes":spec["state_bytes"],
        "state_bits":spec["state_bits"],
        "rounds":spec["rounds"],
        "type":spec["type"],
    }
    for name,spec in PERMUTATIONS.items()
])

display(registry_df)

# Benchmark-function registry (moved earlier — BUG FIX)


def bench_keccak(data, rounds):
    return keccak(data, rounds)

def bench_xoodoo(data, rounds):
    return xoodoo(data, rounds)

def bench_ascon(data, rounds):
    return ascon(data, rounds)

def bench_gimli(data, rounds):
    return gimli(data, rounds)

def bench_chacha(data, rounds):
    return chacha(data, rounds)

def bench_blake2b(data, rounds):

    return blake2b_eval(data)

def bench_blake3(data, rounds):

    return blake3_eval(data)

def bench_haraka256(data, rounds):
    return haraka_style(data, rounds)

def bench_haraka512(data, rounds):
    return haraka_style(data, rounds)

def bench_gbs(data, rounds):
    return GBS_P.permutation_bytes(data, rounds)

BENCHMARK_FUNCTIONS = {
    "Keccak-f[1600]": bench_keccak,
    "Xoodoo": bench_xoodoo,
    "Ascon-p[320]": bench_ascon,
    "Gimli": bench_gimli,
    "ChaCha": bench_chacha,
    "BLAKE2b-compression": bench_blake2b,
    "BLAKE3-compression": bench_blake3,
    "Haraka-style-256": bench_haraka256,
    "Haraka-style-512": bench_haraka512,
    "GBS-photonic": bench_gbs,
}

print("BENCHMARK_FUNCTIONS registered for:", list(BENCHMARK_FUNCTIONS.keys()))

def generic_determinism_check(name, spec, trials=50):

    benchmark_functions = globals().get(
        "BENCHMARK_FUNCTIONS",
        None
    )

    if benchmark_functions is not None:
        fn = benchmark_functions[name]

    else:

        fn = spec["fn"]

    for _ in range(trials):

        x = random_bytes(
            spec["state_bytes"]
        )


        y1 = fn(
            x,
            spec["rounds"]
        )

        y2 = fn(
            x,
            spec["rounds"]
        )

        if y1 != y2:
            return False

        if len(y1) != spec.get("output_bytes", spec["state_bytes"]):
            return False

    return True


correctness_rows = []


for name, spec in PERMUTATIONS.items():

    if name == "GBS-photonic":

        ok = (
            "gbs_validation_df" in globals()
            and bool(gbs_validation_df["input_equals_inverse_output"].all())
            and bool(gbs_validation_df["inverse_output_equals_repermuted_input"].all())
        )

    else:

        try:

            ok = generic_determinism_check(
                name,
                spec
            )

        except Exception as e:

            print(
                f"{name} sanity-check error:",
                repr(e)
            )

            ok = False

    correctness_rows.append({
        "algorithm": name,
        "determinism_size_check": ok
    })


correctness_df = pd.DataFrame(
    correctness_rows
)

display(
    correctness_df
)


# ------------------------------------------------------------
# SHA3-256 known-answer test
# ------------------------------------------------------------

print(
    "SHA3 empty KAT:",
    sha3_256_reference(b"")
    == hashlib.sha3_256(b"").digest()
)
#---------------------------------------------------------------
# Gimli known-answer test
#---------------------------------------------------------------

if (
    "GIMLI_KAT_WORDS_IN" in globals()
    and
    "GIMLI_KAT_WORDS_OUT" in globals()
):

    gimli_kat_input = u32_bytes(
        GIMLI_KAT_WORDS_IN
    )

    gimli_kat_output = gimli(
        gimli_kat_input
    )

    gimli_kat_pass = (
        u32_list(gimli_kat_output)
        == GIMLI_KAT_WORDS_OUT
    )

    print(
        "Gimli KAT:",
        gimli_kat_pass
    )

else:

    print(
        "Gimli KAT: SKIPPED "
        "(GIMLI_KAT_WORDS_IN / GIMLI_KAT_WORDS_OUT "
        "not yet defined)"
    )

def benchmark(fn, x, trials, warmup):
    # Warm-up
    for _ in range(warmup):
        fn(x)

    # Timed execution
    t0 = time.perf_counter()

    for _ in range(trials):
        fn(x)

    elapsed = time.perf_counter() - t0

    sec_per_call = elapsed / trials

    throughput_MB_s = len(x) / sec_per_call / 1e6

    return sec_per_call, throughput_MB_s


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------

performance_rows = []

for name, spec in PERMUTATIONS.items():

    bench_fn = BENCHMARK_FUNCTIONS[name]

    x = random_bytes(spec["state_bytes"])

    # Benchmark
    sec, mbps = benchmark(
        lambda z, fn=bench_fn, s=spec: fn(z, s["rounds"]),
        x,
        CFG["performance_trials"],
        CFG["performance_warmup"]
    )

    performance_rows.append({
        "algorithm": name,
        "type": spec["type"],
        "state_bytes": spec["state_bytes"],
        "rounds": spec["rounds"],
        "latency_us": sec * 1e6,
        "throughput_MB_s": mbps,
    })


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

performance_df = pd.DataFrame(performance_rows)

performance_df = performance_df.sort_values(
    "latency_us"
).reset_index(drop=True)

display(performance_df)

# A4 — Memory footprint (NEW)

import tracemalloc

memory_rows = []

for name, spec in PERMUTATIONS.items():
    fn = BENCHMARK_FUNCTIONS[name]
    x = random_bytes(spec["state_bytes"])

    # warm up (avoid first-call import/allocation noise)
    fn(x, spec["rounds"])

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    for _ in range(20):
        fn(x, spec["rounds"])
    snap_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_rows.append({
        "algorithm": name,
        "state_bytes": spec["state_bytes"],
        "peak_traced_bytes_per_20_calls": peak,
        "approx_peak_bytes_per_call": peak / 20.0,
    })

memory_df = pd.DataFrame(memory_rows).sort_values(
    "approx_peak_bytes_per_call"
).reset_index(drop=True)
display(memory_df)

# A5 — Code size / implementation complexity (NEW)

import inspect

CODE_SIZE_TARGETS = {
    "Keccak-f[1600]": keccak_round,
    "Xoodoo": xoodoo_round,
    "Ascon-p[320]": ascon_round,
    "Gimli": gimli,
    "ChaCha": chacha_qr,
    "BLAKE2b-compression": blake2b_compression,
    "BLAKE3-compression": blake3_compression,
    "Haraka-style-256": haraka_style,
    "Haraka-style-512": haraka_style,
    "GBS-photonic": GBSPhotonicPermutation.permutation_int,
}

codesize_rows = []
for name, func in CODE_SIZE_TARGETS.items():
    try:
        src_text = inspect.getsource(func)
        lines = [l for l in src_text.splitlines() if l.strip() and not l.strip().startswith("#")]
        codesize_rows.append({
            "algorithm": name,
            "core_function": func.__name__,
            "source_lines_noncomment": len(lines),
            "source_bytes": len(src_text.encode("utf-8")),
        })
    except (OSError, TypeError) as e:
        codesize_rows.append({
            "algorithm": name,
            "core_function": getattr(func, "__name__", str(func)),
            "source_lines_noncomment": None,
            "source_bytes": None,
        })

codesize_df = pd.DataFrame(codesize_rows).sort_values(
    "source_lines_noncomment"
).reset_index(drop=True)
display(codesize_df)

def avalanche_test(name, spec, samples):
    fn = BENCHMARK_FUNCTIONS[name]

    nbits = spec["state_bits"]
    out_bits = spec.get("output_bits", nbits)
    vals = []

    for _ in range(samples):

        # Random input state
        x = random_bytes(spec["state_bytes"])

        # Randomly select one input bit
        i = random.randrange(nbits)

        # Original output
        y = fn(x, spec["rounds"])

        # Output after flipping exactly one input bit
        x_flipped = flip_bit(x, i)
        y2 = fn(x_flipped, spec["rounds"])

        # Number of changed output bits
        changed = hamming_distance(y, y2)

        vals.append(changed)

    a = np.asarray(vals, dtype=np.float64)

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": nbits,
        "samples": samples,

        "mean_changed_bits": float(a.mean()),
        "min_changed_bits": int(a.min()),
        "max_changed_bits": int(a.max()),
        "std_changed_bits": float(a.std(ddof=1)),

        # Ideal avalanche target = 50%
        "avalanche_percent": float(
            100.0 * a.mean() / out_bits
        ),

        # Distance from ideal 50%
        "absolute_deviation_from_50_percent": float(
            abs(100.0 * a.mean() / out_bits - 50.0)
        )
    }


# ---------------------------------------------------------
# Run avalanche experiment for ALL algorithms
# ---------------------------------------------------------

avalanche_rows = []

for name, spec in PERMUTATIONS.items():

    result = avalanche_test(
        name,
        spec,
        CFG["avalanche_samples"]
    )

    avalanche_rows.append(result)


# Convert results to DataFrame
avalanche_df = pd.DataFrame(avalanche_rows)


avalanche_df = avalanche_df.sort_values(
    "absolute_deviation_from_50_percent"
).reset_index(drop=True)


display(avalanche_df)

plt.figure(figsize=(12,5))
plt.bar(avalanche_df["algorithm"],avalanche_df["avalanche_percent"])
plt.axhline(50,linestyle="--",label="50% reference")
plt.ylabel("Changed output bits (%)")
plt.title("Avalanche effect — all original candidates")
plt.xticks(rotation=45,ha="right")
plt.legend()
plt.tight_layout()
plt.show()

def sac_summary(
    name,
    spec,
    samples=1000,
    max_input_positions=64,
    max_output_bits=256
):


    fn = BENCHMARK_FUNCTIONS[name]

    nbits = spec["state_bits"]


    num_input_positions = min(
        nbits,
        max_input_positions
    )


    input_positions = np.linspace(
        0,
        nbits - 1,
        num_input_positions,
        dtype=int
    )

    output_bits = min(
        nbits,
        max_output_bits
    )


    counts = np.zeros(
        (len(input_positions), output_bits),
        dtype=np.int64
    )

    # -----------------------------------------------------
    # SAC experiment
    # -----------------------------------------------------

    for _ in range(samples):

        # Random state
        x = random_bytes(
            spec["state_bytes"]
        )

        # Original output
        y = fn(
            x,
            spec["rounds"]
        )

        for ii, i in enumerate(input_positions):

            # Flip exactly one input bit
            x_flipped = flip_bit(
                x,
                int(i)
            )

            # New output
            y2 = fn(
                x_flipped,
                spec["rounds"]
            )

            # XOR outputs
            d = bytes(
                a ^ b
                for a, b in zip(y, y2)
            )

            # Record output-bit changes
            for j in range(output_bits):

                changed = (
                    d[j // 8] >>
                    (j % 8)
                ) & 1

                counts[ii, j] += changed

    # -----------------------------------------------------
    # Convert counts to probabilities
    # -----------------------------------------------------

    p = counts / samples

    # Ideal probability = 0.5
    bias = np.abs(p - 0.5)

    # -----------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": nbits,
        "input_positions_tested": len(input_positions),
        "output_bits_tested": output_bits,
        "samples": samples,

        # Average absolute distance from ideal 0.5
        "mean_abs_SAC_bias": float(
            bias.mean()
        ),

        # Worst observed deviation
        "max_abs_SAC_bias": float(
            bias.max()
        ),

        # Convert average bias into percentage points
        "mean_abs_SAC_bias_percent": float(
            bias.mean() * 100
        ),

        "max_abs_SAC_bias_percent": float(
            bias.max() * 100
        )
    }


# =========================================================
# Run SAC for ALL algorithms
# =========================================================

sac_rows = []

for name, spec in PERMUTATIONS.items():

    result = sac_summary(
        name,
        spec,
        samples=CFG["sac_samples"]
    )

    sac_rows.append(result)


# =========================================================
# Create result table
# =========================================================

sac_df = pd.DataFrame(
    sac_rows
)

# Smaller SAC bias = closer to ideal 0.5
sac_df = sac_df.sort_values(
    "mean_abs_SAC_bias"
).reset_index(drop=True)

display(sac_df)

def bic_test(
    name,
    spec,
    samples=1000,
    max_output_bits=128
):

    fn = BENCHMARK_FUNCTIONS[name]

    nbits = spec["state_bits"]

    # Number of output bits to analyze
    out = min(
        nbits,
        max_output_bits
    )

    observations = []

    # ---------------------------------------------------------
    # Generate observations
    # ---------------------------------------------------------

    for _ in range(samples):

        # Random input
        x = random_bytes(
            spec["state_bytes"]
        )

        # Random input bit to flip
        i = random.randrange(
            nbits
        )

        # Original output
        y = fn(
            x,
            spec["rounds"]
        )

        # Output after one input-bit flip
        x_flipped = flip_bit(
            x,
            i
        )

        y2 = fn(
            x_flipped,
            spec["rounds"]
        )

        # Output difference
        d = bytes(
            a ^ b
            for a, b in zip(y, y2)
        )

        observations.append([
            (d[j // 8] >> (j % 8)) & 1
            for j in range(out)
        ])

    # ---------------------------------------------------------
    # Convert observations to NumPy array
    # ---------------------------------------------------------

    X = np.asarray(
        observations,
        dtype=np.float64
    )

    # ---------------------------------------------------------
    # Pairwise correlation between output-bit changes
    # ---------------------------------------------------------

    C = np.corrcoef(
        X,
        rowvar=False
    )

    tri = np.triu_indices_from(
        C,
        k=1
    )

    vals = np.abs(
        C[tri]
    )

    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        mean_corr = np.nan
        max_corr = np.nan
    else:
        mean_corr = float(
            np.mean(vals)
        )

        max_corr = float(
            np.max(vals)
        )

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": nbits,
        "samples": samples,
        "output_bits_tested": out,
        "mean_abs_correlation": mean_corr,
        "max_abs_correlation": max_corr,
    }


# =========================================================
# Run BIC for ALL algorithms
# =========================================================

bic_rows = []

for name, spec in PERMUTATIONS.items():

    result = bic_test(
        name,
        spec,
        samples=CFG["bic_samples"]
    )

    bic_rows.append(
        result
    )


# =========================================================
# Create result DataFrame
# =========================================================

bic_df = pd.DataFrame(
    bic_rows
)

# Smaller correlation is closer to independent
bic_df = bic_df.sort_values(
    "mean_abs_correlation",
    na_position="last"
).reset_index(drop=True)

display(bic_df)

# First-r-round diffusion

def first_r_fn(name, state, r):


    # --------------------------------------------------------
    # Keccak-f[1600]
    # --------------------------------------------------------
    if name == "Keccak-f[1600]":
        return keccak(
            state,
            r,
            first_rounds=True
        )

    # --------------------------------------------------------
    # Xoodoo
    # --------------------------------------------------------
    if name == "Xoodoo":
        return xoodoo(
            state,
            r,
            first_rounds=True
        )

    # --------------------------------------------------------
    # Ascon-p[320]
    # --------------------------------------------------------
    if name == "Ascon-p[320]":
        return ascon(
            state,
            r,
            first_rounds=True
        )

    # --------------------------------------------------------
    # Gimli
    # --------------------------------------------------------
    if name == "Gimli":
        return gimli(
            state,
            r,
            first_rounds=True
        )

    # --------------------------------------------------------
    # ChaCha
   #----------------------------------------------------------
    if name == "ChaCha":
        if r % 2 != 0:
            raise ValueError(
                "ChaCha requires an even round count."
            )

        return chacha(
            state,
            r,
            first_rounds=True
        )

    # --------------------------------------------------------
    # Haraka-style-256
    # --------------------------------------------------------
    if name == "Haraka-style-256":
        return haraka_style(
            state,
            r
        )

    # --------------------------------------------------------
    # Haraka-style-512
    # --------------------------------------------------------
    if name == "Haraka-style-512":
        return haraka_style(
            state,
            r
        )

    # --------------------------------------------------------
    # GBS-photonic
    # --------------------------------------------------------

    if name == "GBS-photonic":

        if not 1 <= r <= GBS_P.rounds:
            raise ValueError(
                f"GBS-1600 round must be between "
                f"1 and {GBS_P.rounds}"
            )

        return GBS_P.permutation_bytes(
            state,
            rounds=r
        )
    raise ValueError(
        f"No first-round implementation for {name}"
    )


# ============================================================
# Algorithms included in first-round diffusion analysis
# ============================================================

ROUND_ANALYSIS_NAMES = [
    "Keccak-f[1600]",
    "Xoodoo",
    "Ascon-p[320]",
    "Gimli",
    "ChaCha",
    "Haraka-style-256",
    "Haraka-style-512",
    "GBS-photonic",
]


# ============================================================
# Generate diffusion results
# ============================================================

diffusion_rows = []


for name in ROUND_ANALYSIS_NAMES:

    spec = PERMUTATIONS[name]

    # --------------------------------------------------------
    # Determine valid round counts
    # --------------------------------------------------------

    if name == "ChaCha":

        # ChaCha only uses even round counts
        valid_rounds = range(
            2,
            spec["rounds"] + 1,
            2
        )

    else:

        valid_rounds = range(
            1,
            spec["rounds"] + 1
        )


    # --------------------------------------------------------
    # Evaluate every valid round count
    # --------------------------------------------------------

    for r in valid_rounds:

        vals = []

        for _ in range(
            CFG["round_diffusion_samples"]
        ):

            # Random input
            x = random_bytes(
                spec["state_bytes"]
            )

            # Random input bit
            i = random.randrange(
                spec["state_bits"]
            )

            # Output before bit flip
            y = first_r_fn(
                name,
                x,
                r
            )

            y2 = first_r_fn(
                name,
                flip_bit(x, i),
                r
            )

            # Hamming distance
            vals.append(
                hamming_distance(
                    y,
                    y2
                )
            )


        # ----------------------------------------------------
        # Calculate normalized avalanche percentage
        # ----------------------------------------------------

        avalanche_percent = (
            100.0 *
            np.mean(vals) /
            spec["state_bits"]
        )


        diffusion_rows.append({

            "algorithm": name,

            "round": r,

            "mean_changed_bits": float(
                np.mean(vals)
            ),

            "std_changed_bits": float(
                np.std(vals, ddof=1)
            ),

            "avalanche_percent": float(
                avalanche_percent
            ),

            "ideal_avalanche_percent": 50.0,

            "deviation_from_50_percent": float(
                abs(
                    avalanche_percent - 50.0
                )
            ),

            "samples": CFG[
                "round_diffusion_samples"
            ],

        })


# ============================================================
# Create DataFrame
# ============================================================

diffusion_df = pd.DataFrame(
    diffusion_rows
)


# ============================================================
# Display first results
# ============================================================

display(
    diffusion_df.head(20)
)


# ============================================================
# First-r-round diffusion plot
# ============================================================

plt.figure(figsize=(12, 6))

for name, group in diffusion_df.groupby("algorithm"):

    plt.plot(
        group["round"],
        group["avalanche_percent"],
        marker="o",
        label=name
    )


plt.axhline(
    50,
    linestyle="--",
    label="Ideal 50%"
)

plt.xlabel(
    "Number of first rounds executed"
)

plt.ylabel(
    "Changed output bits (%)"
)

plt.title(
    "Diffusion Growth Across First Rounds"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()

plt.figure(figsize=(12,6))
for name,g in diffusion_df.groupby("algorithm"):
    plt.plot(g["round"],g["avalanche_percent"],marker="o",label=name)

plt.axhline(50,linestyle="--",label="50% reference")
plt.xlabel("FIRST rounds executed")
plt.ylabel("Changed output bits (%)")
plt.title("Diffusion growth by first-round count")
plt.legend()
plt.tight_layout()
plt.show()

#  Low-round behavior summary (NEW)


low_round_rows = []
for name, _grp in diffusion_df.groupby("algorithm"):
    g_sorted = _grp.sort_values("round")
    r1 = g_sorted.iloc[0]
    r2 = g_sorted.iloc[1] if len(g_sorted) > 1 else g_sorted.iloc[0]
    low_round_rows.append({
        "algorithm": name,
        "round_1_label": int(r1["round"]),
        "round_1_pct_bits_changed": float(r1["avalanche_percent"]) if "avalanche_percent" in r1 else None,
        "round_2_label": int(r2["round"]),
        "round_2_pct_bits_changed": float(r2["avalanche_percent"]) if "avalanche_percent" in r2 else None,
    })

low_round_df = pd.DataFrame(low_round_rows)
display(low_round_df)

# ============================================================
#   Differential Screening
# ============================================================



def differential_screen(
    name,
    spec,
    samples=2000
):


    fn = BENCHMARK_FUNCTIONS[name]

    counts = Counter()

    # --------------------------------------------------------
    # Fixed input difference
    # --------------------------------------------------------

    input_bit = 0

    # --------------------------------------------------------
    # Differential experiment
    # --------------------------------------------------------

    for _ in range(samples):

        # Random input state
        x = random_bytes(
            spec["state_bytes"]
        )

        # Flip exactly input bit 0
        x2 = flip_bit(
            x,
            input_bit
        )

        # Original output
        y = fn(
            x,
            spec["rounds"]
        )

        # Differential output
        y2 = fn(
            x2,
            spec["rounds"]
        )

        # ΔY = F(X) XOR F(X')
        d = bytes(
            a ^ b
            for a, b in zip(y, y2)
        )

        counts[d] += 1

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    most_common_difference, most_common_count = (
        counts.most_common(1)[0]
    )

    max_probability = (
        most_common_count / samples
    )

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": spec["state_bits"],
        "input_difference": "bit 0 flipped",
        "samples": samples,

        "unique_output_differences": len(
            counts
        ),

        "unique_difference_ratio": (
            len(counts) / samples
        ),

        "max_empirical_probability": (
            max_probability
        ),

        "most_common_count": (
            most_common_count
        ),

        "most_common_difference_hex": (
            most_common_difference.hex()
        ),
    }


# ============================================================
# Run differential screening for ALL algorithms
# ============================================================

differential_rows = []

for name, spec in PERMUTATIONS.items():

    result = differential_screen(
        name,
        spec,
        samples=CFG["differential_samples"]
    )

    differential_rows.append(
        result
    )


# ============================================================
# Create DataFrame
# ============================================================

differential_df = pd.DataFrame(
    differential_rows
)

differential_df = differential_df.sort_values(
    "max_empirical_probability"
).reset_index(drop=True)

display(differential_df)

# Linear Screening



def parity(x):

    return x.bit_count() & 1


def linear_screen(
    name,
    spec,
    samples=5000
):


    fn = BENCHMARK_FUNCTIONS[name]

    nbits = spec["state_bits"]

    # --------------------------------------------------------
    # Random non-zero input/output masks
    # --------------------------------------------------------

    a = random.getrandbits(nbits)

    while a == 0:
        a = random.getrandbits(nbits)

    b = random.getrandbits(nbits)

    while b == 0:
        b = random.getrandbits(nbits)

    # --------------------------------------------------------
    # Count agreement
    # --------------------------------------------------------

    hits = 0

    for _ in range(samples):

        # Random input state
        x = random_bytes(
            spec["state_bytes"]
        )

        # Evaluate algorithm
        y = fn(
            x,
            spec["rounds"]
        )

        # Convert byte strings to integers
        xi = int.from_bytes(
            x,
            "little"
        )

        yi = int.from_bytes(
            y,
            "little"
        )

        # Input linear expression
        input_parity = parity(
            a & xi
        )

        # Output linear expression
        output_parity = parity(
            b & yi
        )

        # Check whether the two expressions agree
        if input_parity ^ output_parity == 0:
            hits += 1

    # --------------------------------------------------------
    # Empirical probability
    # --------------------------------------------------------

    p = hits / samples

    # Ideal probability = 0.5
    absolute_bias = abs(
        p - 0.5
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": nbits,
        "samples": samples,

        "empirical_probability": float(
            p
        ),

        "absolute_bias": float(
            absolute_bias
        ),

        "bias_percent": float(
            absolute_bias * 100
        ),

        "input_mask_hex": format(
            a,
            f"0{(nbits + 3)//4}x"
        ),

        "output_mask_hex": format(
            b,
            f"0{(nbits + 3)//4}x"
        ),
    }


# ============================================================
# Run linear screening for ALL algorithms
# ============================================================

linear_rows = []

for name, spec in PERMUTATIONS.items():

    result = linear_screen(
        name,
        spec,
        samples=CFG["linear_samples"]
    )

    linear_rows.append(
        result
    )


# ============================================================
# Create DataFrame
# ============================================================

linear_df = pd.DataFrame(
    linear_rows
)


linear_df = linear_df.sort_values(
    "absolute_bias"
).reset_index(drop=True)

display(linear_df)

# Projected Algebraic Degree and Boolean Nonlinearity



def mobius(values):

    a = np.asarray(
        values,
        dtype=np.uint8
    ).copy()

    length = len(a)

    if length == 0:
        raise ValueError(
            "Truth table cannot be empty."
        )

    # Number of Boolean variables
    n = int(
        math.log2(length)
    )

    # Ensure length is a power of two
    if (1 << n) != length:
        raise ValueError(
            "Truth-table length must be a power of two."
        )

    for i in range(n):

        step = 1 << i

        for base in range(
            0,
            length,
            2 * step
        ):

            for j in range(
                base,
                base + step
            ):
                a[j + step] ^= a[j]

    return a


def walsh_hadamard_transform(values):
    h = np.asarray(
        [
            1 if v == 0 else -1
            for v in values
        ],
        dtype=np.int64
    )

    length = len(h)

    step = 1

    while step < length:

        for i in range(
            0,
            length,
            2 * step
        ):

            left = h[
                i:i + step
            ].copy()

            right = h[
                i + step:i + 2 * step
            ].copy()

            h[
                i:i + step
            ] = left + right

            h[
                i + step:i + 2 * step
            ] = left - right

        step *= 2

    return h


def projected_algebraic(
    name,
    spec,
    input_bits=8,
    output_bit=0
):



    fn = BENCHMARK_FUNCTIONS[name]

    state_bits = spec["state_bits"]

    if input_bits < 1:
        raise ValueError(
            "input_bits must be >= 1."
        )

    if input_bits > state_bits:
        raise ValueError(
            f"{name} has only {state_bits} state bits."
        )

    if output_bit < 0 or output_bit >= state_bits:
        raise ValueError(
            f"output_bit must be in [0,{state_bits-1}]."
        )


    base = bytes(
        spec["state_bytes"]
    )

    truth = []

    # --------------------------------------------------------
    # Construct complete truth table
    # --------------------------------------------------------

    for xint in range(
        1 << input_bits
    ):

        x = bytearray(
            base
        )

        # Set selected input variables
        for i in range(
            input_bits
        ):

            if (xint >> i) & 1:

                x[
                    i // 8
                ] ^= (
                    1 << (i % 8)
                )

        # ----------------------------------------------------
        # Evaluate primitive
        # ----------------------------------------------------

        y = fn(
            bytes(x),
            spec["rounds"]
        )

        # ----------------------------------------------------
        # Extract selected output bit
        # ----------------------------------------------------

        output_value = (
            y[output_bit // 8]
            >>
            (output_bit % 8)
        ) & 1

        truth.append(
            output_value
        )

    # ========================================================
    # Algebraic Normal Form
    # ========================================================

    anf = mobius(
        truth
    )

    degree = 0

    for index, coefficient in enumerate(
        anf
    ):

        if coefficient:

            # Number of variables in this monomial
            monomial_degree = (
                index.bit_count()
            )

            degree = max(
                degree,
                monomial_degree
            )


    spectrum = walsh_hadamard_transform(
        truth
    )

    max_walsh = int(
        np.max(
            np.abs(spectrum)
        )
    )

    nonlinearity = (
        (1 << (input_bits - 1))
        -
        max_walsh // 2
    )

    # ========================================================
    # Return
    # ========================================================

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": state_bits,
        "projection_input_bits": input_bits,
        "projection_output_bit": output_bit,
        "ANF_degree": int(degree),
        "Boolean_nonlinearity": int(
            nonlinearity
        ),
        "maximum_Walsh_magnitude": max_walsh,
    }


# ============================================================
# Run for ALL algorithms
# ============================================================

degree_rows = []

for name, spec in PERMUTATIONS.items():

    result = projected_algebraic(
        name,
        spec,
        input_bits=8,
        output_bit=0
    )

    degree_rows.append(
        result
    )


# ============================================================
# Create DataFrame
# ============================================================

degree_df = pd.DataFrame(
    degree_rows
)

display(
    degree_df
)

# Symmetry analysis (NEW)


def symmetry_test(name, spec, samples=500):
    fn = BENCHMARK_FUNCTIONS[name]
    n = spec["state_bytes"]

    reversal_hits = 0
    rotation_hits = 0
    self_inverse_hits = 0

    for _ in range(samples):
        x = random_bytes(n)
        y = fn(x, spec["rounds"])

        # 1. byte-reversal symmetry
        x_rev = bytes(reversed(x))
        y_from_rev = fn(x_rev, spec["rounds"])
        if y_from_rev == bytes(reversed(y)):
            reversal_hits += 1

        # 2. one-byte cyclic rotation symmetry
        x_rot = x[1:] + x[:1]
        y_from_rot = fn(x_rot, spec["rounds"])
        if y_from_rot == (y[1:] + y[:1]):
            rotation_hits += 1

        # 3. self-inverse tendency  F(F(X)) == X
        # (only meaningful when output size == input size)
        if len(y) == n:
            y2 = fn(y, spec["rounds"])
            if y2 == x:
                self_inverse_hits += 1

    trivial = (n < 2)

    return {
        "algorithm": name,
        "type": spec["type"],
        "samples": samples,
        "byte_reversal_symmetry_rate": (float("nan") if trivial else reversal_hits / samples),
        "cyclic_rotation_symmetry_rate": (float("nan") if trivial else rotation_hits / samples),
        "self_inverse_rate_F_F_X_eq_X": self_inverse_hits / samples,
        "note": ("1-byte state: reversal/rotation trivially identity" if trivial else ""),
    }

symmetry_rows = []
for name, spec in PERMUTATIONS.items():
    symmetry_rows.append(symmetry_test(name, spec, samples=CFG.get("symmetry_samples", 500)))

symmetry_df = pd.DataFrame(symmetry_rows)
display(symmetry_df)

# Structural Tests



def structural_test(
    name,
    spec,
    samples=2000
):


    # --------------------------------------------------------
    # Use the normalized wrapper.
    # --------------------------------------------------------

    fn = BENCHMARK_FUNCTIONS[name]

    fixed = 0
    comp = 0
    same = 0

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    for _ in range(samples):

        # Random input
        x = random_bytes(
            spec["state_bytes"]
        )

        # Original output
        y = fn(
            x,
            spec["rounds"]
        )

        # ----------------------------------------------------
        # Complement input
        # ----------------------------------------------------

        xc = bytes(
            v ^ 0xFF
            for v in x
        )

        # Output of complemented input
        yc = fn(
            xc,
            spec["rounds"]
        )

        # ----------------------------------------------------
        # 1. Fixed-point test
        # ----------------------------------------------------

        if y == x:
            fixed += 1

        # ----------------------------------------------------
        # 2. Complementarity test
        # ----------------------------------------------------

        complement_y = bytes(
            v ^ 0xFF
            for v in y
        )

        if yc == complement_y:
            comp += 1

        # ----------------------------------------------------
        # 3. Same-output-under-complement test
        # ----------------------------------------------------

        if yc == y:
            same += 1

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    return {
        "algorithm": name,
        "type": spec["type"],
        "state_bits": spec["state_bits"],
        "samples": samples,

        "fixed_point_rate": (
            fixed / samples
        ),

        "complementarity_rate": (
            comp / samples
        ),

        "same_output_under_complement_rate": (
            same / samples
        ),
    }


# ============================================================
# Run structural tests for ALL algorithms
# ============================================================

structural_rows = []

for name, spec in PERMUTATIONS.items():

    result = structural_test(
        name,
        spec,
        samples=CFG["structural_samples"]
    )

    structural_rows.append(
        result
    )


# ============================================================
# Create DataFrame
# ============================================================

structural_df = pd.DataFrame(
    structural_rows
)

display(
    structural_df
)

# Invariant output-bit structure


def invariant_bit_test(name, spec, samples=500, max_bits=256):
    fn = BENCHMARK_FUNCTIONS[name]

    y0 = fn(random_bytes(spec["state_bytes"]), spec["rounds"])
    out_bits = min(len(y0) * 8, max_bits)

    ones_count = np.zeros(out_bits, dtype=np.int64)

    for _ in range(samples):
        x = random_bytes(spec["state_bytes"])
        y = fn(x, spec["rounds"])
        for j in range(out_bits):
            ones_count[j] += (y[j // 8] >> (j % 8)) & 1

    frac_ones = ones_count / samples
    invariant_bits = int(np.sum((frac_ones == 0.0) | (frac_ones == 1.0)))

    return {
        "algorithm": name,
        "type": spec["type"],
        "samples": samples,
        "output_bits_tested": out_bits,
        "invariant_bit_count": invariant_bits,
        "invariant_bit_fraction": invariant_bits / out_bits,
        "max_abs_deviation_from_50pct": float(np.max(np.abs(frac_ones - 0.5))),
    }

invariant_rows = []
for name, spec in PERMUTATIONS.items():
    invariant_rows.append(invariant_bit_test(name, spec, samples=CFG.get("structural_samples", 500)))

invariant_df = pd.DataFrame(invariant_rows).sort_values("invariant_bit_count", ascending=False).reset_index(drop=True)
display(invariant_df)

# GBS-1600 sampled permutation structure


rng=np.random.default_rng(SEED+5)
sampled_orbits=[]
for i in range(min(100, CFG["structural_samples"])):
    x=bytes(rng.bytes(GBS_P.state_bytes))
    y=GBS_P.permutation_bytes(x)
    inv=GBS_P.inverse_bytes(y)
    orbit_seen={x}
    cur=y
    length=1
    # Limit trajectory length; this is only a structural sample.
    for _ in range(15):
        if cur in orbit_seen:
            break
        orbit_seen.add(cur)
        cur=GBS_P.permutation_bytes(cur)
        length+=1
    sampled_orbits.append({
        "sample":i,
        "inverse_correct":inv==x,
        "sampled_trajectory_length":length,
        "returned_to_start_within_limit":cur==x,
    })

gbs_cycle_df=pd.DataFrame(sampled_orbits)
display(gbs_cycle_df)
print("GBS-1600 inverse correct for all sampled states:",bool(gbs_cycle_df["inverse_correct"].all()))

def byte_frequency_test(data):
    observed = np.bincount(
        np.frombuffer(data, dtype=np.uint8),
        minlength=256
    )

    expected = np.full(
        256,
        len(data) / 256.0
    )

    stat, p = chisquare(
        observed,
        expected
    )

    return {
        "bytes": len(data),
        "chi_square": float(stat),
        "p_value": float(p),
        "alpha": 0.01,
        "passes_alpha_0.01": bool(p >= 0.01),
    }

STREAM_DIR = Path("stat_streams_all_9")
STREAM_DIR.mkdir(
    exist_ok=True
)


def generate_stream(
    name,
    nbytes=1_000_000
):


    spec = PERMUTATIONS[name]
    fn = BENCHMARK_FUNCTIONS[name]

    out = bytearray()
    counter = 0

    modulus = 1 << (
        8 * spec["state_bytes"]
    )

    while len(out) < nbytes:
        value = counter % modulus

        x = value.to_bytes(
            spec["state_bytes"],
            "little"
        )

        y = fn(
            x,
            spec["rounds"]
        )

        out.extend(y)
        counter += 1

    return bytes(
        out[:nbytes]
    )


frequency_rows = []

for name in PERMUTATIONS:

    stream = generate_stream(
        name,
        CFG["frequency_bytes"]
    )

    result = byte_frequency_test(
        stream
    )

    result["algorithm"] = name

    frequency_rows.append(
        result
    )


frequency_df = pd.DataFrame(
    frequency_rows
)[
    [
        "algorithm",
        "bytes",
        "chi_square",
        "p_value",
        "alpha",
        "passes_alpha_0.01",
    ]
]

display(frequency_df)

# Runs test and Shannon entropy


def bit_runs_test(data: bytes):

    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    n = len(bits)
    pi = bits.mean()

    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return {"applicable": False, "proportion_ones": float(pi)}

    v_obs = 1 + np.sum(bits[1:] != bits[:-1])
    expected = 2 * n * pi * (1 - pi)
    variance = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    z = 0.0 if variance == 0 else (v_obs - expected) / (2 * math.sqrt(2 * n) * pi * (1 - pi))

    from math import erfc, sqrt
    p_value = erfc(abs(z) / sqrt(2))

    return {
        "applicable": True,
        "proportion_ones": float(pi),
        "observed_runs": int(v_obs),
        "expected_runs": float(expected),
        "p_value": float(p_value),
        "passes_alpha_0.01": bool(p_value >= 0.01),
    }


def shannon_entropy_bytes(data: bytes):

    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts / counts.sum()
    p_nonzero = p[p > 0]
    h = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(h)


stat_rows = []
for name in PERMUTATIONS:
    stream = generate_stream(name, CFG["frequency_bytes"])
    runs_result = bit_runs_test(stream)
    entropy_bits_per_byte = shannon_entropy_bytes(stream)

    row = {
        "algorithm": name,
        "stream_bytes": len(stream),
        "shannon_entropy_bits_per_byte": entropy_bits_per_byte,
        "entropy_deficit_from_8": 8.0 - entropy_bits_per_byte,
    }
    row.update({f"runs_{k}": v for k, v in runs_result.items()})
    stat_rows.append(row)

statistical_df = pd.DataFrame(stat_rows)
display(statistical_df)

# Statistical Test Stream Generation


STREAM_DIR = Path("stat_streams_all_9")
STREAM_DIR.mkdir(
    exist_ok=True
)

# Generate exactly one stream per algorithm.
for name in PERMUTATIONS:

    safe = "".join(
        c if c.isalnum() else "_"
        for c in name
    )

    path = (
        STREAM_DIR /
        (safe + ".bin")
    )

    stream = generate_stream(
        name,
        CFG["frequency_bytes"]
    )

    path.write_bytes(
        stream
    )

    print(
        f"{name}: "
        f"{path.stat().st_size:,} bytes"
    )

# NIST SP 800-22 (NIST STS)

NIST_STS_AVAILABLE = False
nist_sts_df = None

try:
    import nistrng

    nist_rows = []
    for name in PERMUTATIONS:
        stream = generate_stream(name, min(CFG["frequency_bytes"], 200_000))
        bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8)).astype(np.int8)

        eligible = nistrng.check_eligibility_all_battery(
            bits, nistrng.SP800_22R1A_BATTERY
        )
        results = nistrng.run_all_battery(bits, eligible, False)

        n_pass = sum(1 for r, _ in results if r.passed)
        n_total = len(results)

        for r, elapsed in results:
            nist_rows.append({
                "algorithm": name,
                "nist_test": r.name,
                "score": r.score,
                "passed": bool(r.passed),
            })

        print(f"{name}: {n_pass}/{n_total} NIST SP800-22 sub-tests passed")

    nist_sts_df = pd.DataFrame(nist_rows)
    NIST_STS_AVAILABLE = True
    display(nist_sts_df)

except ImportError:
    print("NIST STS SKIPPED: 'nistrng' package not installed.")
    print("Run '%pip install nistrng' in a cell above, then re-run this cell.")
except Exception as e:
    print("NIST STS SKIPPED (unexpected error):", repr(e))

# D4b — Dieharder

import subprocess, shutil, re

DIEHARDER_AVAILABLE = shutil.which("dieharder") is not None
dieharder_rows = []

if DIEHARDER_AVAILABLE:
    for name in PERMUTATIONS:
        safe = "".join(c if c.isalnum() else "_" for c in name)
        path = STREAM_DIR / (safe + ".bin")

        try:
            proc = subprocess.run(
                ["dieharder", "-g", "201", "-f", str(path), "-a"],
                capture_output=True, text=True, timeout=600
            )
            for line in proc.stdout.splitlines():
                m = re.match(
                    r"\s*(\S.*?)\|\s*\S+\s*\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|(\S+)",
                    line
                )
                if m:
                    test_name, ntup, p_value, assessment = m.groups()
                    dieharder_rows.append({
                        "algorithm": name,
                        "dieharder_test": test_name.strip(),
                        "p_value": float(p_value),
                        "assessment": assessment.strip(),
                    })
            print(f"{name}: dieharder run complete "
                  f"({sum(1 for r in dieharder_rows if r['algorithm']==name)} sub-tests parsed)")
        except Exception as e:
            print(f"{name}: dieharder run FAILED: {repr(e)}")

    dieharder_df = pd.DataFrame(dieharder_rows)
    if len(dieharder_df):
        display(dieharder_df)
else:
    dieharder_df = None
    print("Dieharder SKIPPED: 'dieharder' binary not found on PATH.")

# D4c — PractRand

import subprocess, shutil

PRACTRAND_AVAILABLE = shutil.which("RNG_test") is not None
practrand_rows = []

if PRACTRAND_AVAILABLE:
    for name in PERMUTATIONS:
        safe = "".join(c if c.isalnum() else "_" for c in name)
        path = STREAM_DIR / (safe + ".bin")

        try:
            with open(path, "rb") as fh:
                proc = subprocess.run(
                    ["RNG_test", "stdin8"],
                    stdin=fh, capture_output=True, text=True, timeout=600
                )
            tail = proc.stdout.strip().splitlines()[-15:]
            practrand_rows.append({
                "algorithm": name,
                "practrand_tail_output": "\n".join(tail),
            })
            print(f"{name}: PractRand run complete")
        except Exception as e:
            print(f"{name}: PractRand run FAILED: {repr(e)}")

    practrand_df = pd.DataFrame(practrand_rows)
    if len(practrand_df):
        display(practrand_df)
else:
    practrand_df = None
    print("PractRand SKIPPED: 'RNG_test' binary not found on PATH.")

def trunc(v,bits):
    return v & ((1<<bits)-1)

def sha3_t(m,bits):
    return trunc(int.from_bytes(hashlib.sha3_256(m).digest(),"little"),bits)

def blake2_t(m,bits):
    return trunc(int.from_bytes(hashlib.blake2b(m,digest_size=32).digest(),"little"),bits)

def blake3_t(m,bits):
    padded = m + bytes(64-len(m)) if len(m) < 64 else m[:64]
    digest = blake3_eval(padded)  # first 32 bytes = chaining value output
    return trunc(int.from_bytes(digest,"little"),bits)

def collision_search(fn,bits,max_queries):
    seen={}

    for q in range(max_queries):
        m=q.to_bytes(8,"little")
        d=fn(m,bits)

        if d in seen:
            return {
                "found":True,
                "queries":q+1,
                "message1":seen[d].hex(),
                "message2":m.hex(),
                "digest":hex(d),
            }

        seen[d]=m

    return {"found":False,"queries":max_queries}

attack_bits=CFG["truncated_attack_bits"]

for name,fn in [
    ("SHA3-256",sha3_t),
    ("BLAKE2b",blake2_t),
    ("BLAKE3",blake3_t),
]:
    print(name,collision_search(
        fn,attack_bits,CFG["truncated_attack_max_queries"]
    ))

BOUNDARY_LENGTHS=[
    0,1,2,7,8,15,16,31,32,
    63,64,127,128,135,136,137,255,256,1024
]

boundary_rows=[]

for n in BOUNDARY_LENGTHS:
    m=bytes(i%256 for i in range(n))

    t0=time.perf_counter()
    d=sha3_256_reference(m)
    elapsed=time.perf_counter()-t0

    blocks=math.ceil((n+2)/136)

    boundary_rows.append({
        "message_bytes":n,
        "absorbed_blocks":blocks,
        "runtime_seconds":elapsed,
        "digest_hex":d.hex(),
    })

boundary_df=pd.DataFrame(boundary_rows)
display(boundary_df)

plt.figure(figsize=(10,5))
plt.plot(
    boundary_df["message_bytes"],
    boundary_df["runtime_seconds"],
    marker="o"
)
plt.xlabel("Message length (bytes)")
plt.ylabel("Reference Python runtime (seconds)")
plt.title("SHA3-256 boundary-message runtime")
plt.tight_layout()
plt.show()

# ============================================================
# F4 — Long-message sponge test
# ============================================================


LONG_MESSAGE_LENGTHS = [10_000, 100_000, 1_000_000]

long_message_rows = []
for n in LONG_MESSAGE_LENGTHS:
    m = (np.arange(n, dtype=np.uint8) % 251).tobytes()

    t0 = time.perf_counter()
    ours = sha3_256_reference(m)
    elapsed = time.perf_counter() - t0

    ref = hashlib.sha3_256(m).digest()
    blocks = math.ceil((n + 2) / 136)

    long_message_rows.append({
        "message_bytes": n,
        "absorbed_blocks": blocks,
        "runtime_seconds": elapsed,
        "matches_hashlib": ours == ref,
    })

long_message_df = pd.DataFrame(long_message_rows)
display(long_message_df)
print("All long messages match hashlib SHA3-256:", bool(long_message_df["matches_hashlib"].all()))

summary_df=pd.DataFrame([
    {
        "algorithm":name,
        "category":spec["type"],
        "state_bits":spec["state_bits"],
        "rounds":spec["rounds"],
        "has_GBS_candidate_implementation":name=="GBS-photonic",
    }
    for name,spec in PERMUTATIONS.items()
])

display(summary_df)

# ============================================================
# Final Test-Case Index & Scorecard
# ============================================================


TEST_INDEX = [
    ("A. Performance", "A1", "Latency (per-call)",            "performance_df",  True),
    ("A. Performance", "A2", "Throughput (MB/s)",              "performance_df",  True),
    ("A. Performance", "A4", "Memory footprint (tracemalloc)", "memory_df",       True),
    ("A. Performance", "A5", "Code size / complexity",         "codesize_df",     True),

    ("B. Diffusion",   "B1", "Avalanche effect",                "avalanche_df",   True),
    ("B. Diffusion",   "B2", "Strict Avalanche Criterion (SAC)", "sac_df",        True),
    ("B. Diffusion",   "B3", "Bit Independence Criterion (BIC)", "bic_df",        True),
    ("B. Diffusion",   "B4", "Diffusion per round (first-r-round)", "diffusion_df", True),

    ("C. Cryptanalysis","C1","Differential screening",          "differential_df",True),
    ("C. Cryptanalysis","C2","Linear screening",                "linear_df",      True),
    ("C. Cryptanalysis","C3","Projected algebraic degree",      "degree_df",      True),
    ("C. Cryptanalysis","C4","Projected Boolean nonlinearity",  "degree_df",      True),
    ("C. Cryptanalysis","C5","Symmetry analysis",               "symmetry_df",    True),

    ("D. Statistical", "D1", "Byte-frequency chi-square",        "frequency_df",  True),
    ("D. Statistical", "D2", "Runs test (bitstream)",            "statistical_df",True),
    ("D. Statistical", "D3", "Shannon entropy",                  "statistical_df",True),
    ("D. Statistical", "D4a", "NIST STS (nistrng, real execution)", "nist_sts_df", NIST_STS_AVAILABLE),
    ("D. Statistical", "D4b", "Dieharder (real execution if binary present)", "dieharder_df", DIEHARDER_AVAILABLE),
    ("E. Structural",  "E1", "Fixed points",                    "structural_df",  True),
    ("E. Structural",  "E2", "Complementarity",                 "structural_df",  True),
    ("E. Structural",  "E3", "Invariant output-bit structure",  "invariant_df",   True),
    ("E. Structural",  "E4", "Low-round behavior (r=1,2 summary)","low_round_df",  True),
    ("E. Structural",  "E5", "GBS-1600 sampled permutation/inverse structure", "gbs_cycle_df", True),

    ("F. Hash/Sponge", "F1", "SHA3-256 sponge KAT",              None,             True),
    ("F. Hash/Sponge", "F2", "SHAKE-256 XOF validation",         "shake_df",       True),
    ("F. Hash/Sponge", "F3", "SHA3 boundary messages",           "boundary_df",    True),
    ("F. Hash/Sponge", "F4", "Long-message sponge test",         "long_message_df",True),
]

scorecard_rows = []
for category, test_id, description, df_name, internally_computed in TEST_INDEX:
    n_algorithms = None
    if df_name is not None and df_name in globals():
        df = globals()[df_name]
        if df is not None and hasattr(df, "columns") and "algorithm" in df.columns:
            n_algorithms = df["algorithm"].nunique()
    scorecard_rows.append({
        "category": category,
        "test_id": test_id,
        "test": description,
        "internally_computed": internally_computed,
        "algorithms_covered": n_algorithms,
    })

scorecard_df = pd.DataFrame(scorecard_rows)
display(scorecard_df)

print()
print("Total distinct test cases in this notebook:", len(scorecard_df))
print("Internally computed (executable in this notebook):",
      int(scorecard_df["internally_computed"].sum()))
print("Documented-only / externally dependent:",
      int((~scorecard_df["internally_computed"]).sum()))
print()
print("Tests per category:")
print(scorecard_df.groupby("category").size().to_string())
print()
print("Final suite: 27 test cases.")

RESULT_DIR=Path("permutation_results_all_9_updated")
RESULT_DIR.mkdir(exist_ok=True)

tables={
    "registry.csv":registry_df,
    "correctness.csv":correctness_df,
    "performance.csv":performance_df,
    "avalanche.csv":avalanche_df,
    "sac.csv":sac_df,
    "bic.csv":bic_df,
    "diffusion_first_rounds.csv":diffusion_df,
    "differential.csv":differential_df,
    "linear.csv":linear_df,
    "algebraic_projection.csv":degree_df,
    "structural.csv":structural_df,
        "sha3_boundary.csv":boundary_df,
    "long_message.csv":long_message_df,
    "memory.csv":memory_df,
    "codesize.csv":codesize_df,
    "symmetry.csv":symmetry_df,
    "invariant_bits.csv":invariant_df,
    "low_round_summary.csv":low_round_df,
    "statistical_runs_entropy.csv":statistical_df,
    "shake256_validation.csv":shake_df,
    "scorecard.csv":scorecard_df,
}

for filename,df in tables.items():
    df.to_csv(RESULT_DIR/filename,index=False)

gbs_perm_table.to_csv(RESULT_DIR/"gbs_1600_sample_mapping.csv",index=False)

print("Saved:",RESULT_DIR.resolve())
print("Number of result files:",len(list(RESULT_DIR.iterdir())))
if GBS_SAMPLING_AVAILABLE:
    gbs_sampling_df.to_csv(RESULT_DIR/"gbs_sampling.csv",index=False)
else:
    print("gbs_sampling.csv SKIPPED (probabilistic sampling unavailable)")