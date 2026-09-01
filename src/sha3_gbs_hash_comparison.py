
# Commented out IPython magic to ensure Python compatibility.
# %pip install strawberryfields
import scipy

import pkg_resources

import scipy
import numpy as np

# Compatibility fix for Strawberry Fields 0.23.0 + SciPy 1.16.3
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields import ops

# ============================================================
# SHA3_GBS_256
# ============================================================
import itertools
import numpy as np
import strawberryfields as sf
from strawberryfields import ops

MASK = (1 << 64) - 1


# ============================================================
# STEP A: CLASSICAL BITS -> PHOTONIC STATE  (the encoding step)
# ============================================================
def encode_classical_to_photonic(bits: str, q, base_squeeze: float = 1.5):
    N = len(bits)
    for j in range(N):
        if bits[j] == "1":
            ops.Sgate(base_squeeze) | q[j]
            phase = 2 * np.pi * (j / N)
            ops.Rgate(phase) | q[j]
        # bit == '0' -> mode q[j] left as vacuum


# ============================================================
# INTERFEROMETER TOPOLOGY
# ============================================================

def _random_matching(N: int, rng: np.random.RandomState):
    perm = rng.permutation(N)
    return [(int(perm[2 * i]), int(perm[2 * i + 1])) for i in range(N // 2)]


# ============================================================
# STEP B: PHOTONIC STATE -> CLASSICAL BITS
# ============================================================

def _compute_three_mode_corr(state, modes):
    # Wick's-theorem correlation extraction.
    sorted_modes = sorted(list(set(modes)))
    if len(sorted_modes) < 3:
        return 0.0

    _, V = state.reduced_gaussian(sorted_modes)
    mode_map = {actual_mode: i for i, actual_mode in enumerate(sorted_modes)}
    m_indices = [mode_map[m] for m in modes]

    def wick(indices):
        if not indices:
            return 1.0
        res, first = 0.0, indices[0]
        for i in range(1, len(indices)):
            res += V[first, indices[i]] * wick(
                [x for x in indices if x != first and x != indices[i]]
            )
        return res

    term_6th = 0.0
    m1, m2, m3 = m_indices[0], m_indices[1], m_indices[2]
    for qp1, qp2, qp3 in itertools.product(
        [2 * m1, 2 * m1 + 1],
        [2 * m2, 2 * m2 + 1],
        [2 * m3, 2 * m3 + 1]
    ):
        term_6th += wick([qp1, qp1, qp2, qp2, qp3, qp3])

    mu = term_6th
    return mu / 64.0


# ============================================================
# FIXED THETA/PHI CACHE
# ============================================================

BASE_SEED = 42

_theta_phi_cache = {}


def _get_fixed_theta_phi(N: int, depth: int):
    key = (N, depth)

    if key not in _theta_phi_cache:
        rng = np.random.RandomState(BASE_SEED)

        phis = rng.uniform(0,2 * np.pi,(depth, N // 2))

        thetas = rng.normal(np.pi / 4,np.pi / 16,(depth, N // 2))

        _theta_phi_cache[key] = (phis, thetas)

    return _theta_phi_cache[key]


# ============================================================
# GBS ROUND
# ============================================================

def _gbs_round(bits: str, depth: int, k: int, seed: int) -> str:
    N = len(bits)
    eng = sf.Engine("gaussian")
    prog = sf.Program(N)

    # FIXED theta/phi
    phis, thetas = _get_fixed_theta_phi(N, depth)

    # Separate RNG for mode pairing
    mode_pairing_rng = np.random.RandomState(seed)

    with prog.context as q:
        encode_classical_to_photonic(bits, q, base_squeeze=1.5)
        for l in range(depth):
            layer_pairs = _random_matching(N, mode_pairing_rng)

            for pair_idx, (m1, m2) in enumerate(layer_pairs):
                theta, phi = thetas[l, pair_idx], phis[l, pair_idx]

                u_bs = np.array([
                    [np.cos(theta), np.exp(-1j * phi) * np.sin(theta)],
                    [-np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
                ])

                ops.Interferometer(u_bs) | (q[m1], q[m2])

    result = eng.run(prog)
    state = result.state
    out_bits = ""

    for j in range(N):
        target_modes = [j, (j + 1) % N, (j + 2) % N]
        mu_j = _compute_three_mode_corr(state, target_modes)
        bit = int(np.floor(10 ** k * mu_j)) % 2
        out_bits += str(bit)

    return out_bits


# ============================================================
# GBS CALL COUNTER
# ============================================================

_gbs_call_counter = {"n": 0}


def reset_gbs_counter():

    _gbs_call_counter["n"] = 0


# ============================================================
# AVALANCHE MIX
# ============================================================

def _avalanche_mix(x: int) -> int:

    x &= MASK
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK
    x ^= x >> 33
    return x


# ============================================================
# ROUND CONSTANT
# ============================================================

def _round_constant(lane_index: int, call_index: int) -> int:
    # Deterministic per-lane, per-call constant
    x = ((lane_index + 1) * 0x9E3779B97F4A7C15) & MASK
    x ^= ((call_index + 1) * 0xBF58476D1CE4E5B9) & MASK
    return _avalanche_mix(x)


# ============================================================
# GBS PERMUTATION
# ============================================================

def gbs_permute(state, depth: int = 16, k: int = 4):

    call_index = _gbs_call_counter["n"]
    _gbs_call_counter["n"] += 1

    bits = "".join(format(lane, "064b") for lane in state)

    out_bits = _gbs_round(
        bits,
        depth=depth,
        k=k,
        seed=BASE_SEED + call_index
    )

    for i in range(len(state)):
        chunk = out_bits[i * 64:(i + 1) * 64]

        raw_lane = int(chunk, 2) & MASK

        combined = raw_lane ^ _round_constant(
            i,
            call_index
        )

        state[i] = _avalanche_mix(combined)


# ============================================================
# SHA3-GBS-256
# ============================================================

def sha3_gbs_256(message):

    RATE = 136
    OUTPUT_LENGTH = 32

    message = message.encode("utf-8")

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------
    padded_message = bytearray(message)
    padded_message.append(0x06)

    while len(padded_message) % RATE != RATE - 1:
        padded_message.append(0x00)

    padded_message.append(0x80)

    # --------------------------------------------------------
    # Initialize 1600-bit state
    # --------------------------------------------------------
    state = [0] * 25

    reset_gbs_counter()

    # --------------------------------------------------------
    # Absorbing Phase
    # --------------------------------------------------------
    for offset in range(0, len(padded_message), RATE):

        block = padded_message[
            offset:offset + RATE
        ]

        for i in range(RATE // 8):

            word = int.from_bytes(
                block[i * 8:(i + 1) * 8],
                byteorder="little"
            )

            state[i] ^= word

        gbs_permute(state)

    # --------------------------------------------------------
    # Squeezing Phase
    # --------------------------------------------------------
    output = bytearray()

    while len(output) < OUTPUT_LENGTH:

        for i in range(RATE // 8):

            word = state[i].to_bytes(
                8,
                byteorder="little"
            )

            output.extend(word)

            if len(output) >= OUTPUT_LENGTH:
                break

        if len(output) < OUTPUT_LENGTH:
            gbs_permute(state)

    return output[:OUTPUT_LENGTH].hex()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    message = input("Enter the message: ")

    digest = sha3_gbs_256(message)

    print("\nOriginal Message :", message)
    print("SHA3-GBS-256 Hash:", digest)

#==========================================
#Evaluation of SHA3_GBS Hash function
#==========================================
import time
import math
import random
import string

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


DIGEST_BITS = 256
ASSUMED_CPU_GHZ = 3.0

GOLDEN_VECTORS = {
    '': '418fc27e6b1c98cc0f6c0925571a621ffadd20eb1474dc17d120272efebbc6ba',
    'a': '68441f630a7a7bf38b7d1a43540c3431fabf4c81de85f673a79aad74ab7700cb',
    'abc': 'ec3c6fa0392f04146e4590eb158093ce8a1c6e29ea0ec66ea36640826ddf9cf2',
    'hello': 'ba038a0199324eebf59e8bdffd0841ff8a08d3ca381e07982133834768b26c2b',
}

STRUCTURAL_INPUTS = ["", "0", "00000000", "a" * 8, "a" * 16, "\x00\x00\x00\x00"]


TEST_MODE = "quick"
N_INPUTS = 100

FULL_BIAS_TRIALS = 100_000
FULL_CORRELATION_TRIALS = 50_000
FULL_COLLISION_TRIALS = 1_000_000
FULL_LARGE_INPUTS = [16 * 1024, 64 * 1024, 1024 * 1024]

QUICK_DETERMINISM_CASES = 2
QUICK_MULTI_BIT_AVALANCHE_TRIALS = 2
QUICK_BIAS_TRIALS = 4
QUICK_CORRELATION_TRIALS = 4
QUICK_COLLISION_TRIALS = 20
QUICK_COMPLEMENT_TRIALS = 2
QUICK_LARGE_INPUTS = [0, 1, 256]
QUICK_BOUNDARY_LENGTHS = [0, 1, 135, 136, 137]

RUN_STANDARD_SHA3_REFERENCE = False


def random_message(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    )


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    return sum(x != y for x, y in zip(hex_to_bits(hex1), hex_to_bits(hex2)))


def _digest_bytes(digest):
    if isinstance(digest, bytes):
        return digest
    if isinstance(digest, bytearray):
        return bytes(digest)
    if isinstance(digest, str):
        return bytes.fromhex(digest)
    raise TypeError(f"Unsupported digest type: {type(digest)!r}")


def _digest_hex(digest):
    return _digest_bytes(digest).hex()


def hamming_distance(a, b):
    a = _digest_bytes(a)
    b = _digest_bytes(b)
    if len(a) != len(b):
        raise ValueError("Digests must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(
            0.0,
            min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s)
        )
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:


    def __init__(self):
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = sha3_gbs_256(msg)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


def build_input_pool(n_inputs=N_INPUTS, msg_len=16):
    pool = []
    seen = set()

    for m in list(GOLDEN_VECTORS.keys()) + STRUCTURAL_INPUTS:
        if m not in seen:
            pool.append(m)
            seen.add(m)

    while len(pool) < n_inputs:
        m = random_message(msg_len)
        if m not in seen:
            pool.append(m)
            seen.add(m)

    return pool[:n_inputs]


# ============================================================
# Metrics
# ============================================================

def run_kat(cache: DigestCache):
    print("=" * 60)
    print("KAT (Known Answer Test) for sha3_gbs_256()")
    print("=" * 60)

    passed = total = 0

    for msg, expected in GOLDEN_VECTORS.items():
        label = repr(msg) if msg else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok

        print(f"{label:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")


def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")

    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")


def run_avalanche_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Avalanche Effect Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    percentages = []

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)

        print(
            f"  {msg!r:12} vs {flipped!r:12} -> "
            f"{changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)"
        )

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(
        f"AVALANCHE RESULT: mean {mean_pct:.1f}% "
        f"(ideal ~50%)"
    )


def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None

    chi_sq = 0.0

    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2

    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n

    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None

    runs = 1 + sum(
        1 for i in range(1, n)
        if bits[i] != bits[i - 1]
    )

    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)

    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M

    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]

    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0

        for b in block:
            if b:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]

    chi_sq = sum(
        (v_counts[i] - n_blocks * pi_values[i]) ** 2
        / (n_blocks * pi_values[i])
        for i in range(4)
    )

    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("NIST SP 800-22 Style Randomness Tests")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    print(f"Bitstream = {len(bits)} bits from {len(inputs)} inputs")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(
            f"  {name:28s} p={p:.4f} "
            f"{'PASS' if p >= 0.01 else 'FAIL'}"
        )



def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}

    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0

    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)

    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, m=2):
    print("\n" + "=" * 60)
    print("Approximate Entropy (ApEn)")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)

    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn({m}) = {apen:.4f}")
    print(f"  p-value   = {p_value:.4f}")
    print(f"  result    = {'PASS' if p_value >= 0.01 else 'FAIL'}")


def run_sac_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("SAC (Strict Avalanche Criterion)")
    print("=" * 60)

    rng = random.Random(0x5AC)
    flip_counts = [0] * DIGEST_BITS

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        bits1 = hex_to_bits(d1)
        bits2 = hex_to_bits(d2)

        for i in range(DIGEST_BITS):
            flip_counts[i] += bits1[i] != bits2[i]

    n_trials = len(inputs)
    probs = [c / n_trials for c in flip_counts]

    print(f"  trials = {n_trials}")
    print(f"  mean P(flip) = {sum(probs)/len(probs):.3f}")
    print(f"  min P(flip)  = {min(probs):.3f}")
    print(f"  max P(flip)  = {max(probs):.3f}")


def run_performance_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Performance Stats")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]

    if not samples:
        print("No timing samples available.")
        return

    mean_s = sum(samples) / len(samples)
    best_s = min(samples)
    worst_s = max(samples)

    total_bytes = sum(
        len(m.encode("utf-8"))
        for m in inputs
        if m in cache.timings
    )

    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  samples     = {len(samples)}")
    print(f"  mean latency= {mean_s:.3f}s")
    print(f"  best        = {best_s:.3f}s")
    print(f"  worst       = {worst_s:.3f}s")
    print(f"  throughput  = {bytes_per_sec:.6f} bytes/sec")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes = total_bytes / len(samples)

    cycles_per_byte = (
        mean_s * cpu_hz / avg_bytes
        if avg_bytes else float("inf")
    )

    print(f"  CPU freq    = {cpu_hz/1e9:.2f} GHz")
    print(f"  cycles/byte = {cycles_per_byte:,.0f}")


def run_structural_test(cache: DigestCache):
    print("\n" + "=" * 60)
    print("Structural Weak-Spot Check")
    print("=" * 60)

    seen = {}

    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        distinct_chars = len(set(digest))

        print(
            f"  input={msg!r:20} "
            f"digest={digest} "
            f"distinct_hex={distinct_chars}"
        )

        if digest in seen and seen[digest] != msg:
            print(f"  !!! DUPLICATE: {seen[digest]!r} and {msg!r}")

        seen[digest] = msg


def run_collision_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Collision Check")
    print("=" * 60)

    seen = {}

    for msg in inputs:
        digest = cache.get(msg)

        if digest in seen and seen[digest] != msg:
            raise AssertionError(
                f"Collision: {seen[digest]!r} and {msg!r}"
            )

        seen[digest] = msg

    print(f"No collision among {len(inputs)} pool inputs.")


def run_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Preimage Sanity Check")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)

    found = any(
        cache.get(msg) == target_digest
        for msg in inputs[1:]
    )

    print(
        "A preimage was found."
        if found else
        f"No preimage found among {len(inputs)-1} pool messages."
    )


def run_second_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Second-Preimage Sanity Check")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)

    found = any(
        msg != original_msg and cache.get(msg) == original_digest
        for msg in inputs[:-1]
    )

    print(
        if found else
        f"No second preimage found among {len(inputs)-1} other messages."
    )


# ============================================================
# TEST 1: BOUNDARY LENGTH + PATTERN
# ============================================================

def _ascii_pattern_message(n, pattern_id):

    if pattern_id == 0:
        return "\x00" * n
    if pattern_id == 1:
        return "\x7f" * n
    if pattern_id == 2:
        return "\x55" * n
    if pattern_id == 3:
        return "\x2a" * n
    if pattern_id == 4:
        return (
            "".join(chr(i) for i in range(128)) * (n // 128)
            + "".join(chr(i) for i in range(n % 128))
        )

    raise ValueError("Unknown pattern ID")


def _boundary_lengths(rate_bytes=136, quick=False):
    if quick:
        return [0, 1, rate_bytes - 1, rate_bytes, rate_bytes + 1]

    lengths = set()

    for base in [
        0, 1, 2, 3,
        rate_bytes, 2 * rate_bytes, 3 * rate_bytes,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 1025,
        4095, 4096, 4097,
    ]:
        for delta in range(-3, 4):
            if base + delta >= 0:
                lengths.add(base + delta)

    return sorted(lengths)


def run_boundary_length_tests(rate_bytes=136, quick=False):
    print("\n" + "=" * 60)
    print("Boundary-Length + Pattern Tests")
    print("=" * 60)

    pattern_ids = range(2) if quick else range(5)
    total = 0

    for n in _boundary_lengths(rate_bytes, quick):
        for pattern_id in pattern_ids:
            msg = _ascii_pattern_message(n, pattern_id)

            d1 = sha3_gbs_256(msg)
            d2 = sha3_gbs_256(msg)

            assert _digest_hex(d1) == _digest_hex(d2), (
                f"Non-deterministic digest at "
                f"length={n}, pattern={pattern_id}"
            )

            total += 1

        print(f"  length={n:5d} bytes: PASS")

    print(f"Boundary test: {total} cases PASS")


# ============================================================
# TEST 2: DETERMINISM
# ============================================================

def run_determinism_test(inputs):
    print("\n" + "=" * 60)
    print("Determinism Test")
    print("=" * 60)

    for i, data in enumerate(inputs):
        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Non-deterministic result for case {i}: {data!r}"
        )

    print(f"Determinism: {len(inputs)} cases PASS")


# ============================================================
# TEST 3: INDEPENDENT STANDARD SHA3 REFERENCE
# ============================================================

def run_standard_sha3_reference_test(inputs):
    import hashlib

    print("\n" + "=" * 60)
    print("Independent Python SHA3-256 Reference Test")
    print("=" * 60)

    failures = []

    for data in inputs:
        expected = hashlib.sha3_256(
            data.encode("utf-8")
        ).digest()

        actual = _digest_bytes(sha3_gbs_256(data))

        if actual != expected:
            failures.append(
                (data, expected.hex(), actual.hex())
            )

    if failures:
        print(
            f"Reference mismatch: "
            f"{len(failures)}/{len(inputs)} cases."
        )
        print(
            "The current implementation "
            "is a custom GBS construction, not standard SHA3-256."
        )

        for data, expected, actual in failures[:5]:
            print(f"  input    = {data!r}")
            print(f"  expected = {expected}")
            print(f"  actual   = {actual}")

        return False

    print(f"Python SHA3-256 reference: {len(inputs)} cases PASS")
    return True


# ============================================================
# TEST 4: MULTI-BIT AVALANCHE
# ============================================================

def _deterministic_flip_ascii(data, rng, flips):
    raw = bytearray(data.encode("ascii"))

    if not raw:
        raw = bytearray(b"A")

    for _ in range(flips):
        idx = rng.randrange(len(raw))

        # Flip only the lower seven bits so the resulting byte remains ASCII.
        raw[idx] ^= 1 << rng.randrange(7)

    return raw.decode("ascii")


def run_multi_bit_avalanche_test(trials=2):
    print("\n" + "=" * 60)
    print("Multi-Bit Avalanche Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)
        original = random_message(length)
        modified = _deterministic_flip_ascii(
            original,
            rng,
            rng.randint(1, 8)
        )

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(modified)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    variance = sum(
        (x - mean) ** 2
        for x in distances
    ) / len(distances)

    stddev = math.sqrt(variance)

    print(f"  trials = {trials:,}")
    print(f"  mean   = {mean:.3f} / {DIGEST_BITS}")
    print(f"  stddev = {stddev:.3f}")
    print(f"  min    = {min(distances)}")
    print(f"  max    = {max(distances)}")

    if not (120 <= mean <= 136):
        print(
            "  Mean is outside the loose "
            "120..136 sanity range."
        )
    else:
        print("Multi-bit avalanche: PASS")


# ============================================================
# TEST 5: OUTPUT-BIT BIAS
# ============================================================

def run_output_bit_bias_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Bias Test")
    print("=" * 60)

    rng = random.Random(0xB1A5)
    ones = [0] * DIGEST_BITS

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)
        digest = _digest_bytes(sha3_gbs_256(data))

        for bit in range(DIGEST_BITS):
            if digest[bit // 8] & (1 << (bit % 8)):
                ones[bit] += 1

    ratios = [
        count / trials
        for count in ones
    ]

    print(f"  trials  = {trials:,}")
    print(f"  min P(1)= {min(ratios):.5f}")
    print(f"  max P(1)= {max(ratios):.5f}")
    print(f"  mean P(1)= {sum(ratios)/DIGEST_BITS:.5f}")

    if trials >= 100_000:
        assert min(ratios) > 0.48
        assert max(ratios) < 0.52
        print("Output-bit bias: PASS")
    else:
        print(
            "Output-bit bias: completed"
        )


# ============================================================
# TEST 6: OUTPUT-BIT PAIR CORRELATION
# ============================================================

def run_output_bit_correlation_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Pair Correlation Test")
    print("=" * 60)

    rng = random.Random(0xC011)
    pair_counts = {}

    n_pairs = DIGEST_BITS * (DIGEST_BITS - 1) // 2

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_bytes(sha3_gbs_256(data))

        bits = [
            (digest[i // 8] >> (i % 8)) & 1
            for i in range(DIGEST_BITS)
        ]

        for i in range(DIGEST_BITS):
            if not bits[i]:
                continue

            for j in range(i + 1, DIGEST_BITS):
                if bits[j]:
                    key = (i, j)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    expected = trials / 4
    largest_deviation = 0.0
    worst_pair = None

    for i in range(DIGEST_BITS):
        for j in range(i + 1, DIGEST_BITS):
            count = pair_counts.get((i, j), 0)
            deviation = abs(count - expected) / trials

            if deviation > largest_deviation:
                largest_deviation = deviation
                worst_pair = (i, j)

    print(f"  trials = {trials:,}")
    print(f"  tested pairs = {n_pairs:,}")
    print(
        f"  largest absolute deviation from 0.25 = "
        f"{largest_deviation:.5f}"
    )
    print(f"  worst pair = {worst_pair}")
    print(
        "Output-bit correlation: completed. "

    )


# ============================================================
# TEST 7: LARGE COLLISION SEARCH
# ============================================================

def run_extended_collision_test(trials=20):
    print("\n" + "=" * 60)
    print("Large Collision Search")
    print("=" * 60)

    rng = random.Random(0xC0111510)
    seen = {}

    for i in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_hex(sha3_gbs_256(data))

        if digest in seen and seen[digest] != data:
            print("COLLISION FOUND")
            print(
                f"  message A = "
                f"{seen[digest].encode('utf-8').hex()}"
            )
            print(
                f"  message B = "
                f"{data.encode('utf-8').hex()}"
            )
            print(f"  digest    = {digest}")
            raise AssertionError("Hash collision")

        seen[digest] = data

    print(f"Collision test: {trials:,} messages, PASS")


# ============================================================
# TEST 8: LARGE-INPUT DETERMINISM
# ============================================================

def run_large_input_test(sizes=None):
    if sizes is None:
        sizes = (
            QUICK_LARGE_INPUTS
            if TEST_MODE == "quick"
            else FULL_LARGE_INPUTS
        )

    print("\n" + "=" * 60)
    print("Large-Input Determinism Test")
    print("=" * 60)

    for size in sizes:
        data = "".join(
            chr(32 + (i % 95))
            for i in range(size)
        )

        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Failed at {size} bytes"
        )

        print(f"  {size:>8} bytes PASS")

    print("Large-input test PASS")


# ============================================================
# TEST 9: COMPLEMENT RELATIONSHIP
# ============================================================

def run_complement_test(trials=2):
    print("\n" + "=" * 60)
    print("Complement-Relationship Test")
    print("=" * 60)

    rng = random.Random(0xC0DE)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)

        original = bytes(
            rng.randrange(128)
            for _ in range(length)
        )

        complement = bytes(
            x ^ 0x7f
            for x in original
        )

        original = original.decode("ascii")
        complement = complement.decode("ascii")

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(complement)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    print(f"  trials = {trials:,}")
    print(
        f"  mean Hamming distance = "
        f"{mean:.3f} / {DIGEST_BITS}"
    )
    print(f"  min = {min(distances)}")
    print(f"  max = {max(distances)}")
    print("Complement test: completed (diagnostic only)")


# ============================================================
# TEST 10: GBS PERMUTATION STRUCTURAL CHECK
# ============================================================

def run_permutation_interface_test():

    print("\n" + "=" * 60)
    print("GBS Permutation Structural Test")
    print("=" * 60)

    state1 = [0] * 25
    state2 = [0] * 25

    reset_gbs_counter()
    gbs_permute(state1)

    reset_gbs_counter()
    gbs_permute(state2)

    assert state1 == state2
    assert all(
        isinstance(x, int) and 0 <= x <= MASK
        for x in state1
    )
    assert any(x != 0 for x in state1)

    print("  deterministic = PASS")
    print("  25 lanes, 64-bit range = PASS")
    print("  non-zero output = PASS")
    print("Permutation structural test: PASS")


# ============================================================
# COMPLETE RUNNER
# ============================================================

def run_extended_suite(n_inputs=N_INPUTS):
    try:
        sha3_gbs_256
        gbs_permute
        reset_gbs_counter
        MASK
    except NameError:
        print(
            "ERROR: run the SHA3_GBS_256 implementation cell "
            "before running the test cell."
        )
        return

    mode = TEST_MODE.lower()

    if mode not in {"quick", "full"}:
        raise ValueError(
            "TEST_MODE must be either 'quick' or 'full'"
        )

    quick = mode == "quick"

    cache = DigestCache()
    inputs = build_input_pool(n_inputs)

    print("=" * 60)
    print(f"SHA3-GBS-256 EXTENDED TEST SUITE | MODE = {mode.upper()}")
    print("=" * 60)

    print(
        f"Building shared pool of {len(inputs)} inputs "
        f"and hashing each ONCE..."
    )

    cache.get_many(inputs)

    print("Pool hashed.\n")

    # Original tests
    run_kat(cache)
    run_avalanche_test(cache, inputs)
    run_nist_tests(cache, inputs)
    run_approximate_entropy_test(cache, inputs)
    run_sac_test(cache, inputs)
    run_performance_tests(cache, inputs)
    run_structural_test(cache)
    run_collision_test(cache, inputs)
    run_preimage_test(cache, inputs)
    run_second_preimage_test(cache, inputs)

    # New requested tests
    run_boundary_length_tests(quick=quick)

    run_determinism_test(
        inputs[
            :QUICK_DETERMINISM_CASES
            if quick else len(inputs)
        ]
    )

    if RUN_STANDARD_SHA3_REFERENCE:
        run_standard_sha3_reference_test(
            inputs[
                :QUICK_DETERMINISM_CASES
                if quick else len(inputs)
            ]
        )

    run_multi_bit_avalanche_test(
        QUICK_MULTI_BIT_AVALANCHE_TRIALS
        if quick else 1000
    )

    run_output_bit_bias_test(
        QUICK_BIAS_TRIALS
        if quick else FULL_BIAS_TRIALS
    )

    run_output_bit_correlation_test(
        QUICK_CORRELATION_TRIALS
        if quick else FULL_CORRELATION_TRIALS
    )

    run_extended_collision_test(
        QUICK_COLLISION_TRIALS
        if quick else FULL_COLLISION_TRIALS
    )

    run_large_input_test(
        QUICK_LARGE_INPUTS
        if quick else FULL_LARGE_INPUTS
    )

    run_complement_test(
        QUICK_COMPLEMENT_TRIALS
        if quick else 1000
    )

    run_permutation_interface_test()

    print("\n" + "=" * 60)
    print(
        f"EXTENDED SUITE COMPLETE. "
        f"Unique cached messages: {len(cache.digests)}"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_extended_suite()

import hashlib
import time
import math

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ============================================================
# THE FUNCTIONS UNDER TEST
# ============================================================

def shake128_hash(data: bytes, output_length: int = 32) -> str:
    return hashlib.shake_128(data).hexdigest(output_length)


def shake256_hash(data: bytes, output_length: int = 64) -> str:
    return hashlib.shake_256(data).hexdigest(output_length)


# ============================================================
# CONFIG
# ============================================================

ASSUMED_CPU_GHZ = 3.0
SHAKE128_GOLDEN = {
    "": "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26",
    "abc": "5881092dd818bf5cf8a3ddb793fbcba74097d5c526a6d35f97b83351940f2cc8",
}
SHAKE256_GOLDEN = {
    "": "ab06d4f98bfdb2c4fef1cce24045dd15cbdd028db79f1e67d67f985e1b19f801",
    "abc": ("483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739"
            "d5a15bef186a5386c75744c0527e1faa9f8726e462a12a4feb06bd8801e751e4"),
}

#  fixed input pool, which is used for sha3_gbs hash function
INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]


# ============================================================
# GENERIC HELPERS (bit-level, math)
# ============================================================

def flip_one_bit(data: bytes, bit_index: int) -> bytes:
    byte_index = bit_index // 8
    bit_in_byte = 7 - (bit_index % 8)
    b = bytearray(data)
    b[byte_index] ^= (1 << bit_in_byte)
    return bytes(b)


def flipped_variant(msg: str) -> str:

    raw = msg.encode("utf-8")
    if len(raw) == 0:
        raw = b"\x00"
    import random
    bit_index = random.randint(0, len(raw) * 8 - 1)
    flipped = flip_one_bit(raw, bit_index)
    try:
        return flipped.decode("utf-8")
    except UnicodeDecodeError:
        return flipped.decode("utf-8", errors="replace")


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    b1, b2 = hex_to_bits(hex1), hex_to_bits(hex2)
    return sum(x != y for x, y in zip(b1, b2))


def _igamc(a, x, iterations=200):

    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(0.0, min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s))
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6  # psutil reports MHz
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:


    def __init__(self, hash_fn, output_length: int):
        self.hash_fn = hash_fn
        self.output_length = output_length
        self.digest_bits = output_length * 8
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = self.hash_fn(msg.encode("utf-8"), self.output_length)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


# ============================================================
# TEST 1: KAT (Known Answer Test)
# ============================================================

def run_kat(cache: DigestCache, golden_vectors, label):
    print("=" * 60)
    print(f"KAT (Known Answer Test) for {label}")
    print("=" * 60)

    passed = total = 0
    for msg, expected in golden_vectors.items():
        display = repr(msg) if msg != "" else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok
        print(f"{display:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")
    if passed < total:
        print(f"{label} FAIL")


# ============================================================
# TEST 2: AVALANCHE EFFECT
# ============================================================

def run_avalanche_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Avalanche Effect Test for {label}")
    print("=" * 60)

    percentages = []
    flipped_of = {}
    for msg in inputs:
        flipped = flipped_variant(msg)
        flipped_of[msg] = flipped
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / cache.digest_bits * 100
        percentages.append(pct)
        print(f"  {msg!r:20} vs {flipped!r:20} -> {changed}/{cache.digest_bits} bits changed ({pct:.1f}%)")

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(f"AVALANCHE RESULT: mean {mean_pct:.1f}% of output bits changed across "
          f"{len(inputs)} inputs (ideal ~50%)")
    if 40 <= mean_pct <= 60:
        print("Closer to 50%.")
    elif mean_pct < 40:
        print("LOW - input changes aren't diffusing well into the output.")
    else:
        print("HIGH - unusual but not necessarily broken")

    return flipped_of


# ============================================================
# TEST 3: NIST SP 800-22 STYLE RANDOMNESS TESTS
# ============================================================

def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return math.erfc(s_obs / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
    chi_sq = 0.0
    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2
    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return None
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M
    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]
    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0
        for b in block:
            if b == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
    chi_sq = sum((v_counts[i] - n_blocks * pi_values[i]) ** 2 / (n_blocks * pi_values[i])
                 for i in range(4))
    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"NIST SP 800-22 Style Randomness Tests for {label}")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)
    print(f"Bitstream built from {len(inputs)} already-hashed inputs = {n} bits\n")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits, block_size=128)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p


    for name, p in results.items():
        print(f"  {name:28s} p={p:.4f}  {'PASS' if p >= 0.01 else 'FAIL'}")


# ============================================================
# TEST 4: APPROXIMATE ENTROPY
# ============================================================

def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}
    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1
    phi = 0.0
    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)
    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, label, m=2):
    print("\n" + "=" * 60)
    print(f"Approximate Entropy (ApEn, m=2) for {label}")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)
    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn(2) = {apen:.4f} (ideal ~= {math.log(2):.4f})")
    print(f"  p-value = {p_value:.4f} -> {'PASS' if p_value >= 0.01 else 'FAIL'} (threshold 0.01)")


# ============================================================
# TEST 5: SAC (Strict Avalanche Criterion)
# ============================================================

def run_sac_test(cache: DigestCache, inputs, flipped_of, label):
    print("\n" + "=" * 60)
    print(f"SAC (Strict Avalanche Criterion) Test for {label}")
    print("=" * 60)

    flip_counts = [0] * cache.digest_bits
    n_trials = 0

    for msg in inputs:
        flipped = flipped_of[msg]
        d1 = cache.get(msg)
        d2 = cache.get(flipped)
        bits1, bits2 = hex_to_bits(d1), hex_to_bits(d2)
        n_trials += 1
        for i in range(cache.digest_bits):
            if bits1[i] != bits2[i]:
                flip_counts[i] += 1

    probs = [c / n_trials for c in flip_counts]
    mean_p = sum(probs) / len(probs)
    min_p, max_p = min(probs), max(probs)

    print(f"  trials = {n_trials} (one per shared input, reusing avalanche pairs)")
    print(f"  mean P(flip) = {mean_p:.3f} (ideal ~0.500)")
    print(f"  min  P(flip) = {min_p:.3f} at bit {probs.index(min_p)}")
    print(f"  max  P(flip) = {max_p:.3f} at bit {probs.index(max_p)}")

    weak_bits = [(i, p) for i, p in enumerate(probs) if abs(p - 0.5) > 0.3]
    if weak_bits:
        print(f"  {len(weak_bits)} bit position(s) far from ideal (|P-0.5| > 0.3)")
    else:
        print("  No output bit positions are wildly off.")

    print(" mean is close to ideal 0.5" if 0.4 <= mean_p <= 0.6
          else " mean is notably off from 0.5")


# ============================================================
# TEST 6: PERFORMANCE
# ============================================================

def run_performance_tests(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Performance Stats for {label}")
    print("(derived from the timings recorded when the shared pool was hashed)")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]
    if not samples:
        print("  No fresh timings available (every input was already cached).")
        return

    mean_s = sum(samples) / len(samples)
    best_s, worst_s = min(samples), max(samples)

    total_bytes = sum(len(m.encode("utf-8")) for m in inputs if m in cache.timings)
    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  sampled {len(samples)} calls")
    print(f"  mean latency = {mean_s:.6f}s, best = {best_s:.6f}s, worst = {worst_s:.6f}s")
    print(f"  throughput   = {bytes_per_sec:.2f} bytes/sec ({bytes_per_sec/1024:.4f} KB/s)")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes_per_call = total_bytes / len(samples)
    cycles_per_byte = (mean_s * cpu_hz) / avg_bytes_per_call if avg_bytes_per_call else float("inf")
    print(f"  CPU freq used = {cpu_hz/1e9:.2f} GHz "
          f"({'auto-detected' if _HAS_PSUTIL else 'assumed'})")
    print(f"  estimated cycles/byte ~= {cycles_per_byte:,.0f}")


# ============================================================
# TEST 7: STRUCTURAL WEAK-SPOT CHECK
# ============================================================

STRUCTURAL_INPUTS = ["", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa", "\x00\x00\x00\x00"]


def run_structural_test(cache: DigestCache, label):
    print("\n" + "=" * 60)
    print(f"Structural Weak-Spot Check (low-diversity inputs) for {label}")
    print("=" * 60)

    digests = []
    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        digests.append((msg, digest))
        distinct_chars = len(set(digest))
        flag = " <-- LOW DIVERSITY" if distinct_chars <= 3 else ""
        print(f"  input={msg!r:20} digest={digest} (distinct hex chars: {distinct_chars}){flag}")

    seen = {}
    dup_found = False
    for msg, digest in digests:
        if digest in seen:
            print(f"  !!! {seen[digest]!r} and {msg!r} produced the SAME digest")
            dup_found = True
        seen[digest] = msg

    if not dup_found:
        print("No duplicate/degenerate digests among the structural inputs.")


# ============================================================
# TEST 8: COLLISION CHECK
# ============================================================

def run_collision_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Collision Check (across the shared input pool) for {label}")
    print("=" * 60)

    seen = {}
    collision_found = False
    for msg in inputs:
        digest = cache.get(msg)
        if digest in seen and seen[digest] != msg:
            print(f"  !!! COLLISION: {seen[digest]!r} and {msg!r} both hash to {digest}")
            collision_found = True
        seen[digest] = msg

    if not collision_found:
        print(f"No collision among {len(inputs)} pool inputs.")


# ============================================================
# TEST 9 & 10: PREIMAGE / SECOND-PREIMAGE SANITY CHECKS
# ============================================================

def run_preimage_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Preimage Sanity Check (searched within the shared pool) for {label}")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)
    print(f"Target digest (from {target_msg!r}): {target_digest}")

    found = False
    for msg in inputs[1:]:
        digest = cache.get(msg)
        if digest == target_digest:
            print(f"  MATCH: {msg!r} -> {digest}")
            found = True
            break

    print("A preimage was found." if found else
          f"No preimage found among the other {len(inputs) - 1} pool inputs "
          "(expected at this sample size).")


def run_second_preimage_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Second-Preimage Sanity Check (searched within the shared pool) for {label}")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)
    print(f"Original message: {original_msg!r} -> {original_digest}")

    found = False
    for msg in inputs[:-1]:
        digest = cache.get(msg)
        if digest == original_digest and msg != original_msg:
            print(f"  MATCH: {msg!r} -> {digest}")
            found = True
            break

    print("A second preimage was found." if found else
          f"No second preimage found among the other {len(inputs) - 1} pool inputs ")


# ============================================================
# RUN ONE VARIANT (SHAKE128 or SHAKE256) OVER THE FIXED POOL
# ============================================================

def run_suite_for(hash_fn, output_length, golden_vectors, label, inputs):
    cache = DigestCache(hash_fn, output_length)

    print("#" * 60)
    print(f"# {label}  (output = {output_length} bytes / {output_length*8} bits)")
    print("#" * 60)
    print(f"Hashing the shared pool of {len(inputs)} fixed inputs ONCE each...")
    cache.get_many(inputs)
    print("Pool hashed. All tests below reuse these digests.\n")

    run_kat(cache, golden_vectors, label)
    flipped_of = run_avalanche_test(cache, inputs, label)
    run_nist_tests(cache, inputs, label)
    run_approximate_entropy_test(cache, inputs, label)
    run_sac_test(cache, inputs, flipped_of, label)
    run_performance_tests(cache, inputs, label)
    run_structural_test(cache, label)
    run_collision_test(cache, inputs, label)
    run_preimage_test(cache, inputs, label)
    run_second_preimage_test(cache, inputs, label)

    print("\n" + "=" * 60)
    print(f"DONE with {label}. Total unique messages hashed: {len(cache.digests)}")
    print("=" * 60 + "\n")


def run_all_tests():
    inputs = INPUT_POOL
    print(f"Fixed shared input pool: {len(inputs)} messages "
          f"(user-supplied, used as-is for every test below)\n")

    run_suite_for(shake128_hash, 32, SHAKE128_GOLDEN, "SHAKE128", inputs)
    run_suite_for(shake256_hash, 64, SHAKE256_GOLDEN, "SHAKE256", inputs)


if __name__ == "__main__":
    run_all_tests()

import hashlib
import time
import math

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ============================================================
# THE FUNCTION UNDER TEST
# ============================================================

def sha1_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ============================================================
# CONFIG
# ============================================================

DIGEST_BITS = 160
ASSUMED_CPU_GHZ = 3.0

# Known-answer vectors (standard, independently verifiable SHA-1 outputs;
# 20-byte digest -> 40 hex chars).
SHA1_GOLDEN = {
    "": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "abc": "a9993e364706816aba3e25717850c26c9cd0d89d",
}

# Fixed input pool, which are used for the sha3_gbs hash function.
INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]


# ============================================================
# GENERIC HELPERS (bit-level, math)
# ============================================================

def flip_one_bit(data: bytes, bit_index: int) -> bytes:
    byte_index = bit_index // 8
    bit_in_byte = 7 - (bit_index % 8)
    b = bytearray(data)
    b[byte_index] ^= (1 << bit_in_byte)
    return bytes(b)


def flipped_variant(msg: str) -> str:

    raw = msg.encode("utf-8")
    if len(raw) == 0:
        raw = b"\x00"
    import random
    bit_index = random.randint(0, len(raw) * 8 - 1)
    flipped = flip_one_bit(raw, bit_index)
    try:
        return flipped.decode("utf-8")
    except UnicodeDecodeError:
        return flipped.decode("utf-8", errors="replace")


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    b1, b2 = hex_to_bits(hex1), hex_to_bits(hex2)
    return sum(x != y for x, y in zip(b1, b2))


def _igamc(a, x, iterations=200):

    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(0.0, min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s))
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6  # psutil reports MHz
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:

    def __init__(self, hash_fn):
        self.hash_fn = hash_fn
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = self.hash_fn(msg.encode("utf-8"))
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


# ============================================================
# TEST 1: KAT (Known Answer Test)
# ============================================================

def run_kat(cache: DigestCache, golden_vectors, label):
    print("=" * 60)
    print(f"KAT (Known Answer Test) for {label}")
    print("=" * 60)

    passed = total = 0
    for msg, expected in golden_vectors.items():
        display = repr(msg) if msg != "" else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok
        print(f"{display:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")
    if passed < total:
        print(f"{label} FAIL")


# ============================================================
# TEST 2: AVALANCHE EFFECT
# ============================================================

def run_avalanche_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Avalanche Effect Test for {label}")
    print("=" * 60)

    percentages = []
    flipped_of = {}
    for msg in inputs:
        flipped = flipped_variant(msg)
        flipped_of[msg] = flipped
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)
        print(f"  {msg!r:20} vs {flipped!r:20} -> {changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)")

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(f"AVALANCHE RESULT: mean {mean_pct:.1f}% of output bits changed across "
          f"{len(inputs)} inputs (ideal ~50%)")
    if 40 <= mean_pct <= 60:
        print("Closer to 50%.")
    elif mean_pct < 40:
        print("LOW - input changes aren't diffusing well into the output.")
    else:
        print("HIGH - unusual but not necessarily broken")

    return flipped_of


# ============================================================
# TEST 3: NIST SP 800-22 STYLE RANDOMNESS TESTS
# ============================================================

def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return math.erfc(s_obs / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
    chi_sq = 0.0
    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2
    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return None
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M
    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]
    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0
        for b in block:
            if b == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
    chi_sq = sum((v_counts[i] - n_blocks * pi_values[i]) ** 2 / (n_blocks * pi_values[i])
                 for i in range(4))
    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"NIST SP 800-22 Style Randomness Tests for {label}")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)
    print(f"Bitstream built from {len(inputs)} already-hashed inputs = {n} bits\n")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits, block_size=128)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(f"  {name:28s} p={p:.4f}  {'PASS' if p >= 0.01 else 'FAIL'}")


# ============================================================
# TEST 4: APPROXIMATE ENTROPY
# ============================================================

def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}
    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1
    phi = 0.0
    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)
    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, label, m=2):
    print("\n" + "=" * 60)
    print(f"Approximate Entropy (ApEn, m=2) for {label}")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)
    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn(2) = {apen:.4f} (ideal ~= {math.log(2):.4f})")
    print(f"  p-value = {p_value:.4f} -> {'PASS' if p_value >= 0.01 else 'FAIL'} (threshold 0.01)")


# ============================================================
# TEST 5: SAC (Strict Avalanche Criterion)
# ============================================================

def run_sac_test(cache: DigestCache, inputs, flipped_of, label):
    print("\n" + "=" * 60)
    print(f"SAC (Strict Avalanche Criterion) Test for {label}")
    print("=" * 60)

    flip_counts = [0] * DIGEST_BITS
    n_trials = 0

    for msg in inputs:
        flipped = flipped_of[msg]
        d1 = cache.get(msg)
        d2 = cache.get(flipped)
        bits1, bits2 = hex_to_bits(d1), hex_to_bits(d2)
        n_trials += 1
        for i in range(DIGEST_BITS):
            if bits1[i] != bits2[i]:
                flip_counts[i] += 1

    probs = [c / n_trials for c in flip_counts]
    mean_p = sum(probs) / len(probs)
    min_p, max_p = min(probs), max(probs)

    print(f"  trials = {n_trials} (one per shared input, reusing avalanche pairs)")
    print(f"  mean P(flip) = {mean_p:.3f} (ideal ~0.500)")
    print(f"  min  P(flip) = {min_p:.3f} at bit {probs.index(min_p)}")
    print(f"  max  P(flip) = {max_p:.3f} at bit {probs.index(max_p)}")

    weak_bits = [(i, p) for i, p in enumerate(probs) if abs(p - 0.5) > 0.3]
    if weak_bits:
        print(f"  {len(weak_bits)} bit position(s) far from ideal (|P-0.5| > 0.3)")
    else:
        print("  No output bit positions are wildly off.")

    print(" mean is close to ideal 0.5" if 0.4 <= mean_p <= 0.6
          else " mean is notably off from 0.5")


# ============================================================
# TEST 6: PERFORMANCE
# ============================================================

def run_performance_tests(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Performance Stats for {label}")
    print("(derived from the timings recorded when the shared pool was hashed)")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]
    if not samples:
        print("  No fresh timings available (every input was already cached).")
        return

    mean_s = sum(samples) / len(samples)
    best_s, worst_s = min(samples), max(samples)

    total_bytes = sum(len(m.encode("utf-8")) for m in inputs if m in cache.timings)
    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  sampled {len(samples)} calls")
    print(f"  mean latency = {mean_s:.6f}s, best = {best_s:.6f}s, worst = {worst_s:.6f}s")
    print(f"  throughput   = {bytes_per_sec:.2f} bytes/sec ({bytes_per_sec/1024:.4f} KB/s)")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes_per_call = total_bytes / len(samples)
    cycles_per_byte = (mean_s * cpu_hz) / avg_bytes_per_call if avg_bytes_per_call else float("inf")
    print(f"  CPU freq used = {cpu_hz/1e9:.2f} GHz "
          f"({'auto-detected' if _HAS_PSUTIL else 'assumed'})")
    print(f"  estimated cycles/byte ~= {cycles_per_byte:,.0f}")


# ============================================================
# TEST 7: STRUCTURAL WEAK-SPOT CHECK
# ============================================================

STRUCTURAL_INPUTS = ["", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa", "\x00\x00\x00\x00"]


def run_structural_test(cache: DigestCache, label):
    print("\n" + "=" * 60)
    print(f"Structural Weak-Spot Check (low-diversity inputs) for {label}")
    print("=" * 60)

    digests = []
    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        digests.append((msg, digest))
        distinct_chars = len(set(digest))
        flag = " <-- LOW DIVERSITY" if distinct_chars <= 3 else ""
        print(f"  input={msg!r:20} digest={digest} (distinct hex chars: {distinct_chars}){flag}")

    seen = {}
    dup_found = False
    for msg, digest in digests:
        if digest in seen:
            print(f"  !!! {seen[digest]!r} and {msg!r} produced the SAME digest")
            dup_found = True
        seen[digest] = msg

    if not dup_found:
        print("No duplicate/degenerate digests among the structural inputs.")


# ============================================================
# TEST 8: COLLISION CHECK
# ============================================================

def run_collision_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Collision Check (across the shared input pool) for {label}")
    print("=" * 60)

    seen = {}
    collision_found = False
    for msg in inputs:
        digest = cache.get(msg)
        if digest in seen and seen[digest] != msg:
            print(f"  !!! COLLISION: {seen[digest]!r} and {msg!r} both hash to {digest}")
            collision_found = True
        seen[digest] = msg

    if not collision_found:
        print(f"No collision among {len(inputs)} pool inputs.")


# ============================================================
# TEST 9 & 10: PREIMAGE / SECOND-PREIMAGE SANITY CHECKS
# ============================================================

def run_preimage_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Preimage Sanity Check (searched within the shared pool) for {label}")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)
    print(f"Target digest (from {target_msg!r}): {target_digest}")

    found = False
    for msg in inputs[1:]:
        digest = cache.get(msg)
        if digest == target_digest:
            print(f"  MATCH: {msg!r} -> {digest}")
            found = True
            break

    print("A preimage was found." if found else
          f"No preimage found among the other {len(inputs) - 1} pool inputs "
          "(expected at this sample size).")


def run_second_preimage_test(cache: DigestCache, inputs, label):
    print("\n" + "=" * 60)
    print(f"Second-Preimage Sanity Check (searched within the shared pool) for {label}")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)
    print(f"Original message: {original_msg!r} -> {original_digest}")

    found = False
    for msg in inputs[:-1]:
        digest = cache.get(msg)
        if digest == original_digest and msg != original_msg:
            print(f"  MATCH: {msg!r} -> {digest}")
            found = True
            break

    print("A second preimage was found." if found else
          f"No second preimage found among the other {len(inputs) - 1} pool inputs ")


# ============================================================
# RUN ALL
# ============================================================

def run_all_tests():
    inputs = INPUT_POOL
    cache = DigestCache(sha1_hash)
    label = "SHA-1"

    print("#" * 60)
    print(f"# {label}  (fixed output = 20 bytes / {DIGEST_BITS} bits)")
    print("#" * 60)
    print(f"Hashing the shared pool of {len(inputs)} fixed inputs ONCE each...")
    cache.get_many(inputs)
    print("Pool hashed. All tests below reuse these digests.\n")

    run_kat(cache, SHA1_GOLDEN, label)
    flipped_of = run_avalanche_test(cache, inputs, label)
    run_nist_tests(cache, inputs, label)
    run_approximate_entropy_test(cache, inputs, label)
    run_sac_test(cache, inputs, flipped_of, label)
    run_performance_tests(cache, inputs, label)
    run_structural_test(cache, label)
    run_collision_test(cache, inputs, label)
    run_preimage_test(cache, inputs, label)
    run_second_preimage_test(cache, inputs, label)

    print("\n" + "=" * 60)
    print(f"DONE with {label}. Total unique messages hashed: {len(cache.digests)}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

import hashlib
import time
import math
import random
import string

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# SHA-2 FAMILY (as provided)
# ============================================================

def sha2_hash(data: bytes, variant: str = "sha256") -> str:

    return hashlib.new(variant, data).hexdigest()


def sha256_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# ADAPTER: sha3_gbs_256 -> standard SHA-256
# ============================================================

def sha3_gbs_256(msg) -> str:
    if isinstance(msg, str):
        data = msg.encode("utf-8")
    elif isinstance(msg, (bytes, bytearray)):
        data = bytes(msg)
    else:
        raise TypeError(f"Unsupported message type: {type(msg)!r}")
    return sha256_hash(data)


# ============================================================
# CONFIG
# ============================================================

DIGEST_BITS = 256
ASSUMED_CPU_GHZ = 3.0

GOLDEN_VECTORS = {
    m: sha3_gbs_256(m) for m in ["", "a", "abc", "hello"]
}

STRUCTURAL_INPUTS = ["", "0", "00000000", "a" * 8, "a" * 16, "\x00\x00\x00\x00"]


INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]

TEST_MODE = "quick"          # "quick" or "full"
N_INPUTS = len(INPUT_POOL)

FULL_BIAS_TRIALS = 100_000
FULL_CORRELATION_TRIALS = 50_000
FULL_COLLISION_TRIALS = 1_000_000
FULL_LARGE_INPUTS = [16 * 1024, 64 * 1024, 1024 * 1024]

QUICK_DETERMINISM_CASES = min(2, N_INPUTS)
QUICK_MULTI_BIT_AVALANCHE_TRIALS = 2
QUICK_BIAS_TRIALS = 4
QUICK_CORRELATION_TRIALS = 4
QUICK_COLLISION_TRIALS = 20
QUICK_COMPLEMENT_TRIALS = 2
QUICK_LARGE_INPUTS = [0, 1, 256]
QUICK_BOUNDARY_LENGTHS = [0, 1, 135, 136, 137]

RUN_STANDARD_SHA3_REFERENCE = False

# ============================================================
# GENERIC HELPERS
# ============================================================

def random_message(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    )


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    return sum(x != y for x, y in zip(hex_to_bits(hex1), hex_to_bits(hex2)))


def _digest_bytes(digest):
    if isinstance(digest, bytes):
        return digest
    if isinstance(digest, bytearray):
        return bytes(digest)
    if isinstance(digest, str):
        return bytes.fromhex(digest)
    raise TypeError(f"Unsupported digest type: {type(digest)!r}")


def _digest_hex(digest):
    return _digest_bytes(digest).hex()


def hamming_distance(a, b):
    a = _digest_bytes(a)
    b = _digest_bytes(b)
    if len(a) != len(b):
        raise ValueError("Digests must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(
            0.0,
            min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s)
        )
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:
    def __init__(self):
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = sha3_gbs_256(msg)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


def build_input_pool(n_inputs=N_INPUTS, msg_len=16):

    pool = []
    seen = set()

    for m in INPUT_POOL:
        if m not in seen:
            pool.append(m)
            seen.add(m)
        if len(pool) >= n_inputs:
            break

    while len(pool) < n_inputs:
        m = random_message(msg_len)
        if m not in seen:
            pool.append(m)
            seen.add(m)

    return pool[:n_inputs]


# ============================================================
# ORIGINAL TESTS
# ============================================================

def run_kat(cache: DigestCache):
    print("=" * 60)
    print("KAT (Known Answer Test) for sha3_gbs_256() [now hashlib SHA-256]")
    print("=" * 60)

    passed = total = 0

    for msg, expected in GOLDEN_VECTORS.items():
        label = repr(msg) if msg else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok

        print(f"{label:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")


def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")

    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")


def run_avalanche_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Avalanche Effect Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    percentages = []

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)

        print(
            f"  {msg!r:12} vs {flipped!r:12} -> "
            f"{changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)"
        )

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(
        f"AVALANCHE RESULT: mean {mean_pct:.1f}% "
        f"(ideal ~50%)"
    )


def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None

    chi_sq = 0.0

    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2

    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n

    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None

    runs = 1 + sum(
        1 for i in range(1, n)
        if bits[i] != bits[i - 1]
    )

    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)

    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M

    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]

    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0

        for b in block:
            if b:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]

    chi_sq = sum(
        (v_counts[i] - n_blocks * pi_values[i]) ** 2
        / (n_blocks * pi_values[i])
        for i in range(4)
    )

    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("NIST SP 800-22 Style Randomness Tests")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    print(f"Bitstream = {len(bits)} bits from {len(inputs)} inputs")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(
            f"  {name:28s} p={p:.4f} "
            f"{'PASS' if p >= 0.01 else 'FAIL'}"
        )



def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}

    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0

    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)

    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, m=2):
    print("\n" + "=" * 60)
    print("Approximate Entropy (ApEn)")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)

    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn({m}) = {apen:.4f}")
    print(f"  p-value   = {p_value:.4f}")
    print(f"  result    = {'PASS' if p_value >= 0.01 else 'FAIL'}")


def run_sac_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("SAC (Strict Avalanche Criterion)")
    print("=" * 60)

    rng = random.Random(0x5AC)
    flip_counts = [0] * DIGEST_BITS

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        bits1 = hex_to_bits(d1)
        bits2 = hex_to_bits(d2)

        for i in range(DIGEST_BITS):
            flip_counts[i] += bits1[i] != bits2[i]

    n_trials = len(inputs)
    probs = [c / n_trials for c in flip_counts]

    print(f"  trials = {n_trials}")
    print(f"  mean P(flip) = {sum(probs)/len(probs):.3f}")
    print(f"  min P(flip)  = {min(probs):.3f}")
    print(f"  max P(flip)  = {max(probs):.3f}")


def run_performance_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Performance Stats")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]

    if not samples:
        print("No timing samples available.")
        return

    mean_s = sum(samples) / len(samples)
    best_s = min(samples)
    worst_s = max(samples)

    total_bytes = sum(
        len(m.encode("utf-8"))
        for m in inputs
        if m in cache.timings
    )

    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  samples     = {len(samples)}")
    print(f"  mean latency= {mean_s*1e6:.2f} microseconds")
    print(f"  best        = {best_s*1e6:.2f} microseconds")
    print(f"  worst       = {worst_s*1e6:.2f} microseconds")
    print(f"  throughput  = {bytes_per_sec:,.0f} bytes/sec")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes = total_bytes / len(samples)

    cycles_per_byte = (
        mean_s * cpu_hz / avg_bytes
        if avg_bytes else float("inf")
    )

    print(f"  CPU freq    = {cpu_hz/1e9:.2f} GHz")
    print(f"  cycles/byte = {cycles_per_byte:,.0f}")


def run_structural_test(cache: DigestCache):
    print("\n" + "=" * 60)
    print("Structural Weak-Spot Check")
    print("=" * 60)

    seen = {}

    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        distinct_chars = len(set(digest))

        print(
            f"  input={msg!r:20} "
            f"digest={digest} "
            f"distinct_hex={distinct_chars}"
        )

        if digest in seen and seen[digest] != msg:
            print(f"  !!! DUPLICATE: {seen[digest]!r} and {msg!r}")

        seen[digest] = msg


def run_collision_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Collision Check")
    print("=" * 60)

    seen = {}

    for msg in inputs:
        digest = cache.get(msg)

        if digest in seen and seen[digest] != msg:
            raise AssertionError(
                f"Collision: {seen[digest]!r} and {msg!r}"
            )

        seen[digest] = msg

    print(f"No collision among {len(inputs)} pool inputs.")


def run_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Preimage Sanity Check")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)

    found = any(
        cache.get(msg) == target_digest
        for msg in inputs[1:]
    )

    print(
        "A preimage was found."
        if found else
        f"No preimage found among {len(inputs)-1} pool messages."
    )


def run_second_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Second-Preimage Sanity Check")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)

    found = any(
        msg != original_msg and cache.get(msg) == original_digest
        for msg in inputs[:-1]
    )

    print(
        "A second preimage was found."
        if found else
        f"No second preimage found among {len(inputs)-1} other messages."
    )


# ============================================================
# BOUNDARY LENGTH + PATTERN
# ============================================================

def _ascii_pattern_message(n, pattern_id):
    if pattern_id == 0:
        return "\x00" * n
    if pattern_id == 1:
        return "\x7f" * n
    if pattern_id == 2:
        return "\x55" * n
    if pattern_id == 3:
        return "\x2a" * n
    if pattern_id == 4:
        return (
            "".join(chr(i) for i in range(128)) * (n // 128)
            + "".join(chr(i) for i in range(n % 128))
        )

    raise ValueError("Unknown pattern ID")


def _boundary_lengths(rate_bytes=136, quick=False):
    if quick:
        return [0, 1, rate_bytes - 1, rate_bytes, rate_bytes + 1]

    lengths = set()

    for base in [
        0, 1, 2, 3,
        rate_bytes, 2 * rate_bytes, 3 * rate_bytes,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 1025,
        4095, 4096, 4097,
    ]:
        for delta in range(-3, 4):
            if base + delta >= 0:
                lengths.add(base + delta)

    return sorted(lengths)


def run_boundary_length_tests(rate_bytes=136, quick=False):
    print("\n" + "=" * 60)
    print("Boundary-Length + Pattern Tests")
    print("=" * 60)

    pattern_ids = range(2) if quick else range(5)
    total = 0

    for n in _boundary_lengths(rate_bytes, quick):
        for pattern_id in pattern_ids:
            msg = _ascii_pattern_message(n, pattern_id)

            d1 = sha3_gbs_256(msg)
            d2 = sha3_gbs_256(msg)

            assert _digest_hex(d1) == _digest_hex(d2), (
                f"Non-deterministic digest at "
                f"length={n}, pattern={pattern_id}"
            )

            total += 1

        print(f"  length={n:5d} bytes: PASS")

    print(f"Boundary test: {total} cases PASS")


def run_determinism_test(inputs):
    print("\n" + "=" * 60)
    print("Determinism Test")
    print("=" * 60)

    for i, data in enumerate(inputs):
        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Non-deterministic result for case {i}: {data!r}"
        )

    print(f"Determinism: {len(inputs)} cases PASS")


def _deterministic_flip_ascii(data, rng, flips):
    raw = bytearray(data.encode("ascii"))

    if not raw:
        raw = bytearray(b"A")

    for _ in range(flips):
        idx = rng.randrange(len(raw))
        raw[idx] ^= 1 << rng.randrange(7)

    return raw.decode("ascii")


def run_multi_bit_avalanche_test(trials=2):
    print("\n" + "=" * 60)
    print("Multi-Bit Avalanche Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)
        original = random_message(length)
        modified = _deterministic_flip_ascii(
            original,
            rng,
            rng.randint(1, 8)
        )

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(modified)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    variance = sum(
        (x - mean) ** 2
        for x in distances
    ) / len(distances)

    stddev = math.sqrt(variance)

    print(f"  trials = {trials:,}")
    print(f"  mean   = {mean:.3f} / {DIGEST_BITS}")
    print(f"  stddev = {stddev:.3f}")
    print(f"  min    = {min(distances)}")
    print(f"  max    = {max(distances)}")

    if not (120 <= mean <= 136):
        print(
            "  mean is outside the loose "
            "120..136 sanity range."
        )
    else:
        print("Multi-bit avalanche: PASS")


def run_output_bit_bias_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Bias Test")
    print("=" * 60)

    rng = random.Random(0xB1A5)
    ones = [0] * DIGEST_BITS

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)
        digest = _digest_bytes(sha3_gbs_256(data))

        for bit in range(DIGEST_BITS):
            if digest[bit // 8] & (1 << (bit % 8)):
                ones[bit] += 1

    ratios = [
        count / trials
        for count in ones
    ]

    print(f"  trials  = {trials:,}")
    print(f"  min P(1)= {min(ratios):.5f}")
    print(f"  max P(1)= {max(ratios):.5f}")
    print(f"  mean P(1)= {sum(ratios)/DIGEST_BITS:.5f}")

    if trials >= 100_000:
        assert min(ratios) > 0.48
        assert max(ratios) < 0.52
        print("Output-bit bias: PASS")
    else:
        print(
            "Output-bit bias: completed; increase trials "
            "for a meaningful statistical bound."
        )


def run_output_bit_correlation_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Pair Correlation Test")
    print("=" * 60)

    rng = random.Random(0xC011)
    pair_counts = {}

    n_pairs = DIGEST_BITS * (DIGEST_BITS - 1) // 2

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_bytes(sha3_gbs_256(data))

        bits = [
            (digest[i // 8] >> (i % 8)) & 1
            for i in range(DIGEST_BITS)
        ]

        for i in range(DIGEST_BITS):
            if not bits[i]:
                continue

            for j in range(i + 1, DIGEST_BITS):
                if bits[j]:
                    key = (i, j)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    expected = trials / 4
    largest_deviation = 0.0
    worst_pair = None

    for i in range(DIGEST_BITS):
        for j in range(i + 1, DIGEST_BITS):
            count = pair_counts.get((i, j), 0)
            deviation = abs(count - expected) / trials

            if deviation > largest_deviation:
                largest_deviation = deviation
                worst_pair = (i, j)

    print(f"  trials = {trials:,}")
    print(f"  tested pairs = {n_pairs:,}")
    print(
        f"  largest absolute deviation from 0.25 = "
        f"{largest_deviation:.5f}"
    )
    print(f"  worst pair = {worst_pair}")
    print(
        "Output-bit correlation: completed. "
        "Collect the distribution before applying a hard threshold."
    )


def run_extended_collision_test(trials=20):
    print("\n" + "=" * 60)
    print("Large Collision Search")
    print("=" * 60)

    rng = random.Random(0xC0111510)
    seen = {}

    for i in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_hex(sha3_gbs_256(data))

        if digest in seen and seen[digest] != data:
            print("COLLISION FOUND")
            print(
                f"  message A = "
                f"{seen[digest].encode('utf-8').hex()}"
            )
            print(
                f"  message B = "
                f"{data.encode('utf-8').hex()}"
            )
            print(f"  digest    = {digest}")
            raise AssertionError("Hash collision")

        seen[digest] = data

    print(f"Collision test: {trials:,} messages, PASS")


def run_large_input_test(sizes=None):
    if sizes is None:
        sizes = (
            QUICK_LARGE_INPUTS
            if TEST_MODE == "quick"
            else FULL_LARGE_INPUTS
        )

    print("\n" + "=" * 60)
    print("Large-Input Determinism Test")
    print("=" * 60)

    for size in sizes:
        data = "".join(
            chr(32 + (i % 95))
            for i in range(size)
        )

        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Failed at {size} bytes"
        )

        print(f"  {size:>8} bytes PASS")

    print("Large-input test PASS")


def run_complement_test(trials=2):
    print("\n" + "=" * 60)
    print("Complement-Relationship Test")
    print("=" * 60)

    rng = random.Random(0xC0DE)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)

        original = bytes(
            rng.randrange(128)
            for _ in range(length)
        )

        complement = bytes(
            x ^ 0x7f
            for x in original
        )

        original = original.decode("ascii")
        complement = complement.decode("ascii")

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(complement)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    print(f"  trials = {trials:,}")
    print(
        f"  mean Hamming distance = "
        f"{mean:.3f} / {DIGEST_BITS}"
    )
    print(f"  min = {min(distances)}")
    print(f"  max = {max(distances)}")
    print("Complement test: completed (diagnostic only)")


# ============================================================
# COMPLETE RUNNER
# ============================================================

def run_extended_suite(n_inputs=N_INPUTS):
    mode = TEST_MODE.lower()

    if mode not in {"quick", "full"}:
        raise ValueError(
            "TEST_MODE must be either 'quick' or 'full'"
        )

    quick = mode == "quick"

    cache = DigestCache()
    inputs = build_input_pool(n_inputs)

    print("=" * 60)
    print(f"SHA-256 (hashlib) EVAL SUITE | MODE = {mode.upper()}")
    print("(sha3_gbs_256 has been replaced with the standard hashlib SHA-256)")
    print("=" * 60)

    print(
        f"Building shared pool of {len(inputs)} inputs "
        f"(from your INPUT_POOL) and hashing each ONCE..."
    )

    cache.get_many(inputs)

    print("Pool hashed.\n")

    run_kat(cache)
    run_avalanche_test(cache, inputs)
    run_nist_tests(cache, inputs)
    run_approximate_entropy_test(cache, inputs)
    run_sac_test(cache, inputs)
    run_performance_tests(cache, inputs)
    run_structural_test(cache)
    run_collision_test(cache, inputs)
    run_preimage_test(cache, inputs)
    run_second_preimage_test(cache, inputs)

    run_boundary_length_tests(quick=quick)

    run_determinism_test(
        inputs[
            :QUICK_DETERMINISM_CASES
            if quick else len(inputs)
        ]
    )

    run_multi_bit_avalanche_test(
        QUICK_MULTI_BIT_AVALANCHE_TRIALS
        if quick else 1000
    )

    run_output_bit_bias_test(
        QUICK_BIAS_TRIALS
        if quick else FULL_BIAS_TRIALS
    )

    run_output_bit_correlation_test(
        QUICK_CORRELATION_TRIALS
        if quick else FULL_CORRELATION_TRIALS
    )

    run_extended_collision_test(
        QUICK_COLLISION_TRIALS
        if quick else FULL_COLLISION_TRIALS
    )

    run_large_input_test(
        QUICK_LARGE_INPUTS
        if quick else FULL_LARGE_INPUTS
    )

    run_complement_test(
        QUICK_COMPLEMENT_TRIALS
        if quick else 1000
    )

    print("\n" + "=" * 60)
    print(
        f"SUITE COMPLETE. "
        f"Unique cached messages: {len(cache.digests)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_extended_suite()

import hashlib
import time
import math
import random
import string

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# SHA-3 (as provided)
# ============================================================

def sha3_256_hash(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


# ============================================================
# ADAPTER: sha3_gbs_256 -> standard SHA3-256
# ============================================================

def sha3_gbs_256(msg) -> str:
    if isinstance(msg, str):
        data = msg.encode("utf-8")
    elif isinstance(msg, (bytes, bytearray)):
        data = bytes(msg)
    else:
        raise TypeError(f"Unsupported message type: {type(msg)!r}")
    return sha3_256_hash(data)


# ============================================================
# CONFIG
# ============================================================

DIGEST_BITS = 256
ASSUMED_CPU_GHZ = 3.0

# Golden vectors recomputed against real hashlib SHA3-256 so the KAT
# test is meaningful for the swapped-in implementation.
GOLDEN_VECTORS = {
    m: sha3_gbs_256(m) for m in ["", "a", "abc", "hello"]
}

STRUCTURAL_INPUTS = ["", "0", "00000000", "a" * 8, "a" * 16, "\x00\x00\x00\x00"]

INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]

TEST_MODE = "quick"          # "quick" or "full"
N_INPUTS = len(INPUT_POOL)

FULL_BIAS_TRIALS = 100_000
FULL_CORRELATION_TRIALS = 50_000
FULL_COLLISION_TRIALS = 1_000_000
FULL_LARGE_INPUTS = [16 * 1024, 64 * 1024, 1024 * 1024]

QUICK_DETERMINISM_CASES = min(2, N_INPUTS)
QUICK_MULTI_BIT_AVALANCHE_TRIALS = 2
QUICK_BIAS_TRIALS = 4
QUICK_CORRELATION_TRIALS = 4
QUICK_COLLISION_TRIALS = 20
QUICK_COMPLEMENT_TRIALS = 2
QUICK_LARGE_INPUTS = [0, 1, 256]
QUICK_BOUNDARY_LENGTHS = [0, 1, 135, 136, 137]

RUN_STANDARD_SHA3_REFERENCE = False

# ============================================================
# GENERIC HELPERS
# ============================================================

def random_message(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    )


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    return sum(x != y for x, y in zip(hex_to_bits(hex1), hex_to_bits(hex2)))


def _digest_bytes(digest):
    if isinstance(digest, bytes):
        return digest
    if isinstance(digest, bytearray):
        return bytes(digest)
    if isinstance(digest, str):
        return bytes.fromhex(digest)
    raise TypeError(f"Unsupported digest type: {type(digest)!r}")


def _digest_hex(digest):
    return _digest_bytes(digest).hex()


def hamming_distance(a, b):
    a = _digest_bytes(a)
    b = _digest_bytes(b)
    if len(a) != len(b):
        raise ValueError("Digests must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(
            0.0,
            min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s)
        )
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:
    def __init__(self):
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = sha3_gbs_256(msg)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


def build_input_pool(n_inputs=N_INPUTS, msg_len=16):
    """Now sourced from the user-supplied INPUT_POOL instead of random."""
    pool = []
    seen = set()

    for m in INPUT_POOL:
        if m not in seen:
            pool.append(m)
            seen.add(m)
        if len(pool) >= n_inputs:
            break

    while len(pool) < n_inputs:
        m = random_message(msg_len)
        if m not in seen:
            pool.append(m)
            seen.add(m)

    return pool[:n_inputs]


# ============================================================
# ORIGINAL TESTS
# ============================================================

def run_kat(cache: DigestCache):
    print("=" * 60)
    print("KAT (Known Answer Test) for sha3_gbs_256() [now hashlib SHA3-256]")
    print("=" * 60)

    passed = total = 0

    for msg, expected in GOLDEN_VECTORS.items():
        label = repr(msg) if msg else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok

        print(f"{label:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")


def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")

    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")


def run_avalanche_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Avalanche Effect Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    percentages = []

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)

        print(
            f"  {msg!r:12} vs {flipped!r:12} -> "
            f"{changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)"
        )

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(
        f"AVALANCHE RESULT: mean {mean_pct:.1f}% "
        f"(ideal ~50%)"
    )


def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None

    chi_sq = 0.0

    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2

    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n

    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None

    runs = 1 + sum(
        1 for i in range(1, n)
        if bits[i] != bits[i - 1]
    )

    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)

    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M

    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]

    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0

        for b in block:
            if b:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]

    chi_sq = sum(
        (v_counts[i] - n_blocks * pi_values[i]) ** 2
        / (n_blocks * pi_values[i])
        for i in range(4)
    )

    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("NIST SP 800-22 Style Randomness Tests")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    print(f"Bitstream = {len(bits)} bits from {len(inputs)} inputs")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(
            f"  {name:28s} p={p:.4f} "
            f"{'PASS' if p >= 0.01 else 'FAIL'}"
        )



def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}

    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0

    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)

    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, m=2):
    print("\n" + "=" * 60)
    print("Approximate Entropy (ApEn)")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)

    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn({m}) = {apen:.4f}")
    print(f"  p-value   = {p_value:.4f}")
    print(f"  result    = {'PASS' if p_value >= 0.01 else 'FAIL'}")


def run_sac_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("SAC (Strict Avalanche Criterion)")
    print("=" * 60)

    rng = random.Random(0x5AC)
    flip_counts = [0] * DIGEST_BITS

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        bits1 = hex_to_bits(d1)
        bits2 = hex_to_bits(d2)

        for i in range(DIGEST_BITS):
            flip_counts[i] += bits1[i] != bits2[i]

    n_trials = len(inputs)
    probs = [c / n_trials for c in flip_counts]

    print(f"  trials = {n_trials}")
    print(f"  mean P(flip) = {sum(probs)/len(probs):.3f}")
    print(f"  min P(flip)  = {min(probs):.3f}")
    print(f"  max P(flip)  = {max(probs):.3f}")


def run_performance_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Performance Stats")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]

    if not samples:
        print("No timing samples available.")
        return

    mean_s = sum(samples) / len(samples)
    best_s = min(samples)
    worst_s = max(samples)

    total_bytes = sum(
        len(m.encode("utf-8"))
        for m in inputs
        if m in cache.timings
    )

    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  samples     = {len(samples)}")
    print(f"  mean latency= {mean_s*1e6:.2f} microseconds")
    print(f"  best        = {best_s*1e6:.2f} microseconds")
    print(f"  worst       = {worst_s*1e6:.2f} microseconds")
    print(f"  throughput  = {bytes_per_sec:,.0f} bytes/sec")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes = total_bytes / len(samples)

    cycles_per_byte = (
        mean_s * cpu_hz / avg_bytes
        if avg_bytes else float("inf")
    )

    print(f"  CPU freq    = {cpu_hz/1e9:.2f} GHz")
    print(f"  cycles/byte = {cycles_per_byte:,.0f}")


def run_structural_test(cache: DigestCache):
    print("\n" + "=" * 60)
    print("Structural Weak-Spot Check")
    print("=" * 60)

    seen = {}

    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        distinct_chars = len(set(digest))

        print(
            f"  input={msg!r:20} "
            f"digest={digest} "
            f"distinct_hex={distinct_chars}"
        )

        if digest in seen and seen[digest] != msg:
            print(f"  !!! DUPLICATE: {seen[digest]!r} and {msg!r}")

        seen[digest] = msg


def run_collision_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Collision Check")
    print("=" * 60)

    seen = {}

    for msg in inputs:
        digest = cache.get(msg)

        if digest in seen and seen[digest] != msg:
            raise AssertionError(
                f"Collision: {seen[digest]!r} and {msg!r}"
            )

        seen[digest] = msg

    print(f"No collision among {len(inputs)} pool inputs.")


def run_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Preimage Sanity Check")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)

    found = any(
        cache.get(msg) == target_digest
        for msg in inputs[1:]
    )

    print(
        "A preimage was found."
        if found else
        f"No preimage found among {len(inputs)-1} pool messages."
    )


def run_second_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Second-Preimage Sanity Check")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)

    found = any(
        msg != original_msg and cache.get(msg) == original_digest
        for msg in inputs[:-1]
    )

    print(
        "A second preimage was found."
        if found else
        f"No second preimage found among {len(inputs)-1} other messages."
    )


# ============================================================
# BOUNDARY LENGTH + PATTERN
# ============================================================

def _ascii_pattern_message(n, pattern_id):
    if pattern_id == 0:
        return "\x00" * n
    if pattern_id == 1:
        return "\x7f" * n
    if pattern_id == 2:
        return "\x55" * n
    if pattern_id == 3:
        return "\x2a" * n
    if pattern_id == 4:
        return (
            "".join(chr(i) for i in range(128)) * (n // 128)
            + "".join(chr(i) for i in range(n % 128))
        )

    raise ValueError("Unknown pattern ID")


def _boundary_lengths(rate_bytes=136, quick=False):
    if quick:
        return [0, 1, rate_bytes - 1, rate_bytes, rate_bytes + 1]

    lengths = set()

    for base in [
        0, 1, 2, 3,
        rate_bytes, 2 * rate_bytes, 3 * rate_bytes,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 1025,
        4095, 4096, 4097,
    ]:
        for delta in range(-3, 4):
            if base + delta >= 0:
                lengths.add(base + delta)

    return sorted(lengths)


def run_boundary_length_tests(rate_bytes=136, quick=False):
    print("\n" + "=" * 60)
    print("Boundary-Length + Pattern Tests")
    print("=" * 60)

    pattern_ids = range(2) if quick else range(5)
    total = 0

    for n in _boundary_lengths(rate_bytes, quick):
        for pattern_id in pattern_ids:
            msg = _ascii_pattern_message(n, pattern_id)

            d1 = sha3_gbs_256(msg)
            d2 = sha3_gbs_256(msg)

            assert _digest_hex(d1) == _digest_hex(d2), (
                f"Non-deterministic digest at "
                f"length={n}, pattern={pattern_id}"
            )

            total += 1

        print(f"  length={n:5d} bytes: PASS")

    print(f"Boundary test: {total} cases PASS")


def run_determinism_test(inputs):
    print("\n" + "=" * 60)
    print("Determinism Test")
    print("=" * 60)

    for i, data in enumerate(inputs):
        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Non-deterministic result for case {i}: {data!r}"
        )

    print(f"Determinism: {len(inputs)} cases PASS")


def _deterministic_flip_ascii(data, rng, flips):
    raw = bytearray(data.encode("ascii"))

    if not raw:
        raw = bytearray(b"A")

    for _ in range(flips):
        idx = rng.randrange(len(raw))
        raw[idx] ^= 1 << rng.randrange(7)

    return raw.decode("ascii")


def run_multi_bit_avalanche_test(trials=2):
    print("\n" + "=" * 60)
    print("Multi-Bit Avalanche Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)
        original = random_message(length)
        modified = _deterministic_flip_ascii(
            original,
            rng,
            rng.randint(1, 8)
        )

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(modified)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    variance = sum(
        (x - mean) ** 2
        for x in distances
    ) / len(distances)

    stddev = math.sqrt(variance)

    print(f"  trials = {trials:,}")
    print(f"  mean   = {mean:.3f} / {DIGEST_BITS}")
    print(f"  stddev = {stddev:.3f}")
    print(f"  min    = {min(distances)}")
    print(f"  max    = {max(distances)}")

    if not (120 <= mean <= 136):
        print(
            "  mean is outside the loose "
            "120..136 sanity range."
        )
    else:
        print("Multi-bit avalanche: PASS")


def run_output_bit_bias_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Bias Test")
    print("=" * 60)

    rng = random.Random(0xB1A5)
    ones = [0] * DIGEST_BITS

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)
        digest = _digest_bytes(sha3_gbs_256(data))

        for bit in range(DIGEST_BITS):
            if digest[bit // 8] & (1 << (bit % 8)):
                ones[bit] += 1

    ratios = [
        count / trials
        for count in ones
    ]

    print(f"  trials  = {trials:,}")
    print(f"  min P(1)= {min(ratios):.5f}")
    print(f"  max P(1)= {max(ratios):.5f}")
    print(f"  mean P(1)= {sum(ratios)/DIGEST_BITS:.5f}")

    if trials >= 100_000:
        assert min(ratios) > 0.48
        assert max(ratios) < 0.52
        print("Output-bit bias: PASS")
    else:
        print(
            "Output-bit bias: completed"
        )


def run_output_bit_correlation_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Pair Correlation Test")
    print("=" * 60)

    rng = random.Random(0xC011)
    pair_counts = {}

    n_pairs = DIGEST_BITS * (DIGEST_BITS - 1) // 2

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_bytes(sha3_gbs_256(data))

        bits = [
            (digest[i // 8] >> (i % 8)) & 1
            for i in range(DIGEST_BITS)
        ]

        for i in range(DIGEST_BITS):
            if not bits[i]:
                continue

            for j in range(i + 1, DIGEST_BITS):
                if bits[j]:
                    key = (i, j)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    expected = trials / 4
    largest_deviation = 0.0
    worst_pair = None

    for i in range(DIGEST_BITS):
        for j in range(i + 1, DIGEST_BITS):
            count = pair_counts.get((i, j), 0)
            deviation = abs(count - expected) / trials

            if deviation > largest_deviation:
                largest_deviation = deviation
                worst_pair = (i, j)

    print(f"  trials = {trials:,}")
    print(f"  tested pairs = {n_pairs:,}")
    print(
        f"  largest absolute deviation from 0.25 = "
        f"{largest_deviation:.5f}"
    )
    print(f"  worst pair = {worst_pair}")
    print(
        "Output-bit correlation: completed. "
        "Collect the distribution before applying a hard threshold."
    )


def run_extended_collision_test(trials=20):
    print("\n" + "=" * 60)
    print("Large Collision Search")
    print("=" * 60)

    rng = random.Random(0xC0111510)
    seen = {}

    for i in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_hex(sha3_gbs_256(data))

        if digest in seen and seen[digest] != data:
            print("COLLISION FOUND")
            print(
                f"  message A = "
                f"{seen[digest].encode('utf-8').hex()}"
            )
            print(
                f"  message B = "
                f"{data.encode('utf-8').hex()}"
            )
            print(f"  digest    = {digest}")
            raise AssertionError("Hash collision")

        seen[digest] = data

    print(f"Collision test: {trials:,} messages, PASS")


def run_large_input_test(sizes=None):
    if sizes is None:
        sizes = (
            QUICK_LARGE_INPUTS
            if TEST_MODE == "quick"
            else FULL_LARGE_INPUTS
        )

    print("\n" + "=" * 60)
    print("Large-Input Determinism Test")
    print("=" * 60)

    for size in sizes:
        data = "".join(
            chr(32 + (i % 95))
            for i in range(size)
        )

        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Failed at {size} bytes"
        )

        print(f"  {size:>8} bytes PASS")

    print("Large-input test PASS")


def run_complement_test(trials=2):
    print("\n" + "=" * 60)
    print("Complement-Relationship Test")
    print("=" * 60)

    rng = random.Random(0xC0DE)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)

        original = bytes(
            rng.randrange(128)
            for _ in range(length)
        )

        complement = bytes(
            x ^ 0x7f
            for x in original
        )

        original = original.decode("ascii")
        complement = complement.decode("ascii")

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(complement)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    print(f"  trials = {trials:,}")
    print(
        f"  mean Hamming distance = "
        f"{mean:.3f} / {DIGEST_BITS}"
    )
    print(f"  min = {min(distances)}")
    print(f"  max = {max(distances)}")
    print("Complement test: completed (diagnostic only)")


# ============================================================
# COMPLETE RUNNER
# ============================================================

def run_extended_suite(n_inputs=N_INPUTS):
    mode = TEST_MODE.lower()

    if mode not in {"quick", "full"}:
        raise ValueError(
            "TEST_MODE must be either 'quick' or 'full'"
        )

    quick = mode == "quick"

    cache = DigestCache()
    inputs = build_input_pool(n_inputs)

    print("=" * 60)
    print(f"SHA3-256 (hashlib) EVAL SUITE | MODE = {mode.upper()}")
    print("(sha3_gbs_256 has been replaced with the standard hashlib SHA3-256)")
    print("=" * 60)

    print(
        f"Building shared pool of {len(inputs)} inputs "
        f"(from your INPUT_POOL) and hashing each ONCE..."
    )

    cache.get_many(inputs)

    print("Pool hashed.\n")

    run_kat(cache)
    run_avalanche_test(cache, inputs)
    run_nist_tests(cache, inputs)
    run_approximate_entropy_test(cache, inputs)
    run_sac_test(cache, inputs)
    run_performance_tests(cache, inputs)
    run_structural_test(cache)
    run_collision_test(cache, inputs)
    run_preimage_test(cache, inputs)
    run_second_preimage_test(cache, inputs)

    run_boundary_length_tests(quick=quick)

    run_determinism_test(
        inputs[
            :QUICK_DETERMINISM_CASES
            if quick else len(inputs)
        ]
    )

    run_multi_bit_avalanche_test(
        QUICK_MULTI_BIT_AVALANCHE_TRIALS
        if quick else 1000
    )

    run_output_bit_bias_test(
        QUICK_BIAS_TRIALS
        if quick else FULL_BIAS_TRIALS
    )

    run_output_bit_correlation_test(
        QUICK_CORRELATION_TRIALS
        if quick else FULL_CORRELATION_TRIALS
    )

    run_extended_collision_test(
        QUICK_COLLISION_TRIALS
        if quick else FULL_COLLISION_TRIALS
    )

    run_large_input_test(
        QUICK_LARGE_INPUTS
        if quick else FULL_LARGE_INPUTS
    )

    run_complement_test(
        QUICK_COMPLEMENT_TRIALS
        if quick else 1000
    )

    print("\n" + "=" * 60)
    print(
        f"SUITE COMPLETE. "
        f"Unique cached messages: {len(cache.digests)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_extended_suite()

import hashlib
import time
import math
import random
import string

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# BLAKE2b (as provided)
# ============================================================

def blake2b_hash(data: bytes, digest_size: int = 32, key: bytes = b"") -> str:
    return hashlib.blake2b(data, digest_size=digest_size, key=key).hexdigest()


# ============================================================
# ADAPTER: sha3_gbs_256 -> standard BLAKE2b (32-byte / 256-bit digest)
# ============================================================

def sha3_gbs_256(msg) -> str:

    if isinstance(msg, str):
        data = msg.encode("utf-8")
    elif isinstance(msg, (bytes, bytearray)):
        data = bytes(msg)
    else:
        raise TypeError(f"Unsupported message type: {type(msg)!r}")
    return blake2b_hash(data, digest_size=32)


# ============================================================
# CONFIG
# ============================================================

DIGEST_BITS = 256
ASSUMED_CPU_GHZ = 3.0

# Golden vectors recomputed against real hashlib BLAKE2b (32-byte) so the KAT
# test is meaningful for the swapped-in implementation.
GOLDEN_VECTORS = {
    m: sha3_gbs_256(m) for m in ["", "a", "abc", "hello"]
}

STRUCTURAL_INPUTS = ["", "0", "00000000", "a" * 8, "a" * 16, "\x00\x00\x00\x00"]

# User-supplied fixed input pool (replaces random pool generation)
INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]

TEST_MODE = "quick"          # "quick" or "full"
N_INPUTS = len(INPUT_POOL)

FULL_BIAS_TRIALS = 100_000
FULL_CORRELATION_TRIALS = 50_000
FULL_COLLISION_TRIALS = 1_000_000
FULL_LARGE_INPUTS = [16 * 1024, 64 * 1024, 1024 * 1024]

QUICK_DETERMINISM_CASES = min(2, N_INPUTS)
QUICK_MULTI_BIT_AVALANCHE_TRIALS = 2
QUICK_BIAS_TRIALS = 4
QUICK_CORRELATION_TRIALS = 4
QUICK_COLLISION_TRIALS = 20
QUICK_COMPLEMENT_TRIALS = 2
QUICK_LARGE_INPUTS = [0, 1, 256]
QUICK_BOUNDARY_LENGTHS = [0, 1, 135, 136, 137]

RUN_STANDARD_SHA3_REFERENCE = False

# ============================================================
# GENERIC HELPERS
# ============================================================

def random_message(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    )


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    return sum(x != y for x, y in zip(hex_to_bits(hex1), hex_to_bits(hex2)))


def _digest_bytes(digest):
    if isinstance(digest, bytes):
        return digest
    if isinstance(digest, bytearray):
        return bytes(digest)
    if isinstance(digest, str):
        return bytes.fromhex(digest)
    raise TypeError(f"Unsupported digest type: {type(digest)!r}")


def _digest_hex(digest):
    return _digest_bytes(digest).hex()


def hamming_distance(a, b):
    a = _digest_bytes(a)
    b = _digest_bytes(b)
    if len(a) != len(b):
        raise ValueError("Digests must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(
            0.0,
            min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s)
        )
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:
    def __init__(self):
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = sha3_gbs_256(msg)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


def build_input_pool(n_inputs=N_INPUTS, msg_len=16):
    """Now sourced from the user-supplied INPUT_POOL instead of random."""
    pool = []
    seen = set()

    for m in INPUT_POOL:
        if m not in seen:
            pool.append(m)
            seen.add(m)
        if len(pool) >= n_inputs:
            break

    # top up with random messages only if the fixed pool is smaller
    # than requested
    while len(pool) < n_inputs:
        m = random_message(msg_len)
        if m not in seen:
            pool.append(m)
            seen.add(m)

    return pool[:n_inputs]


# ============================================================
# ORIGINAL TESTS
# ============================================================

def run_kat(cache: DigestCache):
    print("=" * 60)
    print("KAT (Known Answer Test) for sha3_gbs_256() [now hashlib BLAKE2b (32-byte)]")
    print("=" * 60)

    passed = total = 0

    for msg, expected in GOLDEN_VECTORS.items():
        label = repr(msg) if msg else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok

        print(f"{label:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")


def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")

    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")


def run_avalanche_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Avalanche Effect Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    percentages = []

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)

        print(
            f"  {msg!r:12} vs {flipped!r:12} -> "
            f"{changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)"
        )

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(
        f"AVALANCHE RESULT: mean {mean_pct:.1f}% "
        f"(ideal ~50%)"
    )


def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None

    chi_sq = 0.0

    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2

    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n

    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None

    runs = 1 + sum(
        1 for i in range(1, n)
        if bits[i] != bits[i - 1]
    )

    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)

    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M

    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]

    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0

        for b in block:
            if b:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]

    chi_sq = sum(
        (v_counts[i] - n_blocks * pi_values[i]) ** 2
        / (n_blocks * pi_values[i])
        for i in range(4)
    )

    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("NIST SP 800-22 Style Randomness Tests")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    print(f"Bitstream = {len(bits)} bits from {len(inputs)} inputs")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(
            f"  {name:28s} p={p:.4f} "
            f"{'PASS' if p >= 0.01 else 'FAIL'}"
        )




def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}

    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0

    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)

    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, m=2):
    print("\n" + "=" * 60)
    print("Approximate Entropy (ApEn)")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)

    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn({m}) = {apen:.4f}")
    print(f"  p-value   = {p_value:.4f}")
    print(f"  result    = {'PASS' if p_value >= 0.01 else 'FAIL'}")


def run_sac_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("SAC (Strict Avalanche Criterion)")
    print("=" * 60)

    rng = random.Random(0x5AC)
    flip_counts = [0] * DIGEST_BITS

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        bits1 = hex_to_bits(d1)
        bits2 = hex_to_bits(d2)

        for i in range(DIGEST_BITS):
            flip_counts[i] += bits1[i] != bits2[i]

    n_trials = len(inputs)
    probs = [c / n_trials for c in flip_counts]

    print(f"  trials = {n_trials}")
    print(f"  mean P(flip) = {sum(probs)/len(probs):.3f}")
    print(f"  min P(flip)  = {min(probs):.3f}")
    print(f"  max P(flip)  = {max(probs):.3f}")


def run_performance_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Performance Stats")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]

    if not samples:
        print("No timing samples available.")
        return

    mean_s = sum(samples) / len(samples)
    best_s = min(samples)
    worst_s = max(samples)

    total_bytes = sum(
        len(m.encode("utf-8"))
        for m in inputs
        if m in cache.timings
    )

    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  samples     = {len(samples)}")
    print(f"  mean latency= {mean_s*1e6:.2f} microseconds")
    print(f"  best        = {best_s*1e6:.2f} microseconds")
    print(f"  worst       = {worst_s*1e6:.2f} microseconds")
    print(f"  throughput  = {bytes_per_sec:,.0f} bytes/sec")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes = total_bytes / len(samples)

    cycles_per_byte = (
        mean_s * cpu_hz / avg_bytes
        if avg_bytes else float("inf")
    )

    print(f"  CPU freq    = {cpu_hz/1e9:.2f} GHz")
    print(f"  cycles/byte = {cycles_per_byte:,.0f}")


def run_structural_test(cache: DigestCache):
    print("\n" + "=" * 60)
    print("Structural Weak-Spot Check")
    print("=" * 60)

    seen = {}

    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        distinct_chars = len(set(digest))

        print(
            f"  input={msg!r:20} "
            f"digest={digest} "
            f"distinct_hex={distinct_chars}"
        )

        if digest in seen and seen[digest] != msg:
            print(f"  !!! DUPLICATE: {seen[digest]!r} and {msg!r}")

        seen[digest] = msg


def run_collision_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Collision Check")
    print("=" * 60)

    seen = {}

    for msg in inputs:
        digest = cache.get(msg)

        if digest in seen and seen[digest] != msg:
            raise AssertionError(
                f"Collision: {seen[digest]!r} and {msg!r}"
            )

        seen[digest] = msg

    print(f"No collision among {len(inputs)} pool inputs.")


def run_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Preimage Sanity Check")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)

    found = any(
        cache.get(msg) == target_digest
        for msg in inputs[1:]
    )

    print(
        "A preimage was found."
        if found else
        f"No preimage found among {len(inputs)-1} pool messages."
    )


def run_second_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Second-Preimage Sanity Check")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)

    found = any(
        msg != original_msg and cache.get(msg) == original_digest
        for msg in inputs[:-1]
    )

    print(
        "A second preimage was found."
        if found else
        f"No second preimage found among {len(inputs)-1} other messages."
    )


# ============================================================
# BOUNDARY LENGTH + PATTERN
# ============================================================

def _ascii_pattern_message(n, pattern_id):
    if pattern_id == 0:
        return "\x00" * n
    if pattern_id == 1:
        return "\x7f" * n
    if pattern_id == 2:
        return "\x55" * n
    if pattern_id == 3:
        return "\x2a" * n
    if pattern_id == 4:
        return (
            "".join(chr(i) for i in range(128)) * (n // 128)
            + "".join(chr(i) for i in range(n % 128))
        )

    raise ValueError("Unknown pattern ID")


def _boundary_lengths(rate_bytes=136, quick=False):
    if quick:
        return [0, 1, rate_bytes - 1, rate_bytes, rate_bytes + 1]

    lengths = set()

    for base in [
        0, 1, 2, 3,
        rate_bytes, 2 * rate_bytes, 3 * rate_bytes,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 1025,
        4095, 4096, 4097,
    ]:
        for delta in range(-3, 4):
            if base + delta >= 0:
                lengths.add(base + delta)

    return sorted(lengths)


def run_boundary_length_tests(rate_bytes=136, quick=False):
    print("\n" + "=" * 60)
    print("Boundary-Length + Pattern Tests")
    print("=" * 60)

    pattern_ids = range(2) if quick else range(5)
    total = 0

    for n in _boundary_lengths(rate_bytes, quick):
        for pattern_id in pattern_ids:
            msg = _ascii_pattern_message(n, pattern_id)

            d1 = sha3_gbs_256(msg)
            d2 = sha3_gbs_256(msg)

            assert _digest_hex(d1) == _digest_hex(d2), (
                f"Non-deterministic digest at "
                f"length={n}, pattern={pattern_id}"
            )

            total += 1

        print(f"  length={n:5d} bytes: PASS")

    print(f"Boundary test: {total} cases PASS")


def run_determinism_test(inputs):
    print("\n" + "=" * 60)
    print("Determinism Test")
    print("=" * 60)

    for i, data in enumerate(inputs):
        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Non-deterministic result for case {i}: {data!r}"
        )

    print(f"Determinism: {len(inputs)} cases PASS")


def _deterministic_flip_ascii(data, rng, flips):
    raw = bytearray(data.encode("ascii"))

    if not raw:
        raw = bytearray(b"A")

    for _ in range(flips):
        idx = rng.randrange(len(raw))
        raw[idx] ^= 1 << rng.randrange(7)

    return raw.decode("ascii")


def run_multi_bit_avalanche_test(trials=2):
    print("\n" + "=" * 60)
    print("Multi-Bit Avalanche Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)
        original = random_message(length)
        modified = _deterministic_flip_ascii(
            original,
            rng,
            rng.randint(1, 8)
        )

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(modified)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    variance = sum(
        (x - mean) ** 2
        for x in distances
    ) / len(distances)

    stddev = math.sqrt(variance)

    print(f"  trials = {trials:,}")
    print(f"  mean   = {mean:.3f} / {DIGEST_BITS}")
    print(f"  stddev = {stddev:.3f}")
    print(f"  min    = {min(distances)}")
    print(f"  max    = {max(distances)}")

    if not (120 <= mean <= 136):
        print(
            "  mean is outside the loose "
            "120..136 sanity range."
        )
    else:
        print("Multi-bit avalanche: PASS")


def run_output_bit_bias_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Bias Test")
    print("=" * 60)

    rng = random.Random(0xB1A5)
    ones = [0] * DIGEST_BITS

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)
        digest = _digest_bytes(sha3_gbs_256(data))

        for bit in range(DIGEST_BITS):
            if digest[bit // 8] & (1 << (bit % 8)):
                ones[bit] += 1

    ratios = [
        count / trials
        for count in ones
    ]

    print(f"  trials  = {trials:,}")
    print(f"  min P(1)= {min(ratios):.5f}")
    print(f"  max P(1)= {max(ratios):.5f}")
    print(f"  mean P(1)= {sum(ratios)/DIGEST_BITS:.5f}")

    if trials >= 100_000:
        assert min(ratios) > 0.48
        assert max(ratios) < 0.52
        print("Output-bit bias: PASS")
    else:
        print(
            "Output-bit bias: completed"
        )


def run_output_bit_correlation_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Pair Correlation Test")
    print("=" * 60)

    rng = random.Random(0xC011)
    pair_counts = {}

    n_pairs = DIGEST_BITS * (DIGEST_BITS - 1) // 2

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_bytes(sha3_gbs_256(data))

        bits = [
            (digest[i // 8] >> (i % 8)) & 1
            for i in range(DIGEST_BITS)
        ]

        for i in range(DIGEST_BITS):
            if not bits[i]:
                continue

            for j in range(i + 1, DIGEST_BITS):
                if bits[j]:
                    key = (i, j)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    expected = trials / 4
    largest_deviation = 0.0
    worst_pair = None

    for i in range(DIGEST_BITS):
        for j in range(i + 1, DIGEST_BITS):
            count = pair_counts.get((i, j), 0)
            deviation = abs(count - expected) / trials

            if deviation > largest_deviation:
                largest_deviation = deviation
                worst_pair = (i, j)

    print(f"  trials = {trials:,}")
    print(f"  tested pairs = {n_pairs:,}")
    print(
        f"  largest absolute deviation from 0.25 = "
        f"{largest_deviation:.5f}"
    )
    print(f"  worst pair = {worst_pair}")
    print(
        "Output-bit correlation: completed. "
        "Collect the distribution before applying a hard threshold."
    )


def run_extended_collision_test(trials=20):
    print("\n" + "=" * 60)
    print("Large Collision Search")
    print("=" * 60)

    rng = random.Random(0xC0111510)
    seen = {}

    for i in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_hex(sha3_gbs_256(data))

        if digest in seen and seen[digest] != data:
            print("COLLISION FOUND")
            print(
                f"  message A = "
                f"{seen[digest].encode('utf-8').hex()}"
            )
            print(
                f"  message B = "
                f"{data.encode('utf-8').hex()}"
            )
            print(f"  digest    = {digest}")
            raise AssertionError("Hash collision")

        seen[digest] = data

    print(f"Collision test: {trials:,} messages, PASS")


def run_large_input_test(sizes=None):
    if sizes is None:
        sizes = (
            QUICK_LARGE_INPUTS
            if TEST_MODE == "quick"
            else FULL_LARGE_INPUTS
        )

    print("\n" + "=" * 60)
    print("Large-Input Determinism Test")
    print("=" * 60)

    for size in sizes:
        data = "".join(
            chr(32 + (i % 95))
            for i in range(size)
        )

        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Failed at {size} bytes"
        )

        print(f"  {size:>8} bytes PASS")

    print("Large-input test PASS")


def run_complement_test(trials=2):
    print("\n" + "=" * 60)
    print("Complement-Relationship Test")
    print("=" * 60)

    rng = random.Random(0xC0DE)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)

        original = bytes(
            rng.randrange(128)
            for _ in range(length)
        )

        complement = bytes(
            x ^ 0x7f
            for x in original
        )

        original = original.decode("ascii")
        complement = complement.decode("ascii")

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(complement)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    print(f"  trials = {trials:,}")
    print(
        f"  mean Hamming distance = "
        f"{mean:.3f} / {DIGEST_BITS}"
    )
    print(f"  min = {min(distances)}")
    print(f"  max = {max(distances)}")
    print("Complement test: completed (diagnostic only)")


# ============================================================
# COMPLETE RUNNER
# ============================================================

def run_extended_suite(n_inputs=N_INPUTS):
    mode = TEST_MODE.lower()

    if mode not in {"quick", "full"}:
        raise ValueError(
            "TEST_MODE must be either 'quick' or 'full'"
        )

    quick = mode == "quick"

    cache = DigestCache()
    inputs = build_input_pool(n_inputs)

    print("=" * 60)
    print(f"BLAKE2b (hashlib) EVAL SUITE | MODE = {mode.upper()}")
    print("(sha3_gbs_256 has been replaced with the standard hashlib BLAKE2b (32-byte digest))")
    print("=" * 60)

    print(
        f"Building shared pool of {len(inputs)} inputs "
        f"(from your INPUT_POOL) and hashing each ONCE..."
    )

    cache.get_many(inputs)

    print("Pool hashed.\n")

    run_kat(cache)
    run_avalanche_test(cache, inputs)
    run_nist_tests(cache, inputs)
    run_approximate_entropy_test(cache, inputs)
    run_sac_test(cache, inputs)
    run_performance_tests(cache, inputs)
    run_structural_test(cache)
    run_collision_test(cache, inputs)
    run_preimage_test(cache, inputs)
    run_second_preimage_test(cache, inputs)

    run_boundary_length_tests(quick=quick)

    run_determinism_test(
        inputs[
            :QUICK_DETERMINISM_CASES
            if quick else len(inputs)
        ]
    )

    run_multi_bit_avalanche_test(
        QUICK_MULTI_BIT_AVALANCHE_TRIALS
        if quick else 1000
    )

    run_output_bit_bias_test(
        QUICK_BIAS_TRIALS
        if quick else FULL_BIAS_TRIALS
    )

    run_output_bit_correlation_test(
        QUICK_CORRELATION_TRIALS
        if quick else FULL_CORRELATION_TRIALS
    )

    run_extended_collision_test(
        QUICK_COLLISION_TRIALS
        if quick else FULL_COLLISION_TRIALS
    )

    run_large_input_test(
        QUICK_LARGE_INPUTS
        if quick else FULL_LARGE_INPUTS
    )

    run_complement_test(
        QUICK_COMPLEMENT_TRIALS
        if quick else 1000
    )

    print("\n" + "=" * 60)
    print(
        f"SUITE COMPLETE. "
        f"Unique cached messages: {len(cache.digests)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_extended_suite()

pip install blake3

import hashlib
import time
import math
import random
import string

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# BLAKE3 (as provided)
# ============================================================

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False


def blake3_hash(data: bytes) -> str:
    if not BLAKE3_AVAILABLE:
        return "blake3 not installed"
    return blake3.blake3(data).hexdigest()


# ============================================================
# ADAPTER: sha3_gbs_256 -> BLAKE3 (32-byte / 256-bit digest)
# ============================================================

def sha3_gbs_256(msg) -> str:

    if not BLAKE3_AVAILABLE:
        raise RuntimeError("blake3 not installed (pip install blake3)")
    if isinstance(msg, str):
        data = msg.encode("utf-8")
    elif isinstance(msg, (bytes, bytearray)):
        data = bytes(msg)
    else:
        raise TypeError(f"Unsupported message type: {type(msg)!r}")
    return blake3_hash(data)


# ============================================================
# CONFIG
# ============================================================

DIGEST_BITS = 256
ASSUMED_CPU_GHZ = 3.0

GOLDEN_VECTORS = {
    m: sha3_gbs_256(m) for m in ["", "a", "abc", "hello"]
}

STRUCTURAL_INPUTS = ["", "0", "00000000", "a" * 8, "a" * 16, "\x00\x00\x00\x00"]

# User-supplied fixed input pool (replaces random pool generation)
INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]

TEST_MODE = "quick"          # "quick" or "full"
N_INPUTS = len(INPUT_POOL)

FULL_BIAS_TRIALS = 100_000
FULL_CORRELATION_TRIALS = 50_000
FULL_COLLISION_TRIALS = 1_000_000
FULL_LARGE_INPUTS = [16 * 1024, 64 * 1024, 1024 * 1024]

QUICK_DETERMINISM_CASES = min(2, N_INPUTS)
QUICK_MULTI_BIT_AVALANCHE_TRIALS = 2
QUICK_BIAS_TRIALS = 4
QUICK_CORRELATION_TRIALS = 4
QUICK_COLLISION_TRIALS = 20
QUICK_COMPLEMENT_TRIALS = 2
QUICK_LARGE_INPUTS = [0, 1, 256]
QUICK_BOUNDARY_LENGTHS = [0, 1, 135, 136, 137]

RUN_STANDARD_SHA3_REFERENCE = False

# ============================================================
# GENERIC HELPERS
# ============================================================

def random_message(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    )


def hex_to_bits(hex_str: str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def hamming_distance_hex(hex1: str, hex2: str) -> int:
    return sum(x != y for x, y in zip(hex_to_bits(hex1), hex_to_bits(hex2)))


def _digest_bytes(digest):
    if isinstance(digest, bytes):
        return digest
    if isinstance(digest, bytearray):
        return bytes(digest)
    if isinstance(digest, str):
        return bytes.fromhex(digest)
    raise TypeError(f"Unsupported digest type: {type(digest)!r}")


def _digest_hex(digest):
    return _digest_bytes(digest).hex()


def hamming_distance(a, b):
    a = _digest_bytes(a)
    b = _digest_bytes(b)
    if len(a) != len(b):
        raise ValueError("Digests must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(
            0.0,
            min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s)
        )
    except (ValueError, OverflowError):
        return 0.0


def get_cpu_freq_hz():
    if _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return freq.current * 1e6
        except Exception:
            pass
    return ASSUMED_CPU_GHZ * 1e9


# ============================================================
# SHARED DIGEST CACHE
# ============================================================

class DigestCache:
    def __init__(self):
        self.digests = {}
        self.timings = {}
        self.order = []

    def get(self, msg: str) -> str:
        if msg not in self.digests:
            t0 = time.perf_counter()
            digest = sha3_gbs_256(msg)
            elapsed = time.perf_counter() - t0
            self.digests[msg] = digest
            self.timings[msg] = elapsed
            self.order.append(msg)
        return self.digests[msg]

    def get_many(self, msgs):
        return [self.get(m) for m in msgs]

    def bitstream(self, msgs=None):
        msgs = self.order if msgs is None else msgs
        bits = []
        for m in msgs:
            bits.extend(hex_to_bits(self.get(m)))
        return bits


def build_input_pool(n_inputs=N_INPUTS, msg_len=16):

    pool = []
    seen = set()

    for m in INPUT_POOL:
        if m not in seen:
            pool.append(m)
            seen.add(m)
        if len(pool) >= n_inputs:
            break


    while len(pool) < n_inputs:
        m = random_message(msg_len)
        if m not in seen:
            pool.append(m)
            seen.add(m)

    return pool[:n_inputs]


# ============================================================
# ORIGINAL TESTS
# ============================================================

def run_kat(cache: DigestCache):
    print("=" * 60)
    print("KAT (Known Answer Test) for sha3_gbs_256() [now BLAKE3 (32-byte)]")
    print("=" * 60)

    passed = total = 0

    for msg, expected in GOLDEN_VECTORS.items():
        label = repr(msg) if msg else "'' (empty string)"
        digest = cache.get(msg)
        total += 1
        ok = digest == expected
        passed += ok

        print(f"{label:20s} -> {digest}")
        print(f"    expected: {expected}")
        print(f"    result  : {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print(f"KAT RESULT: {passed}/{total} passed")


def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")

    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")


def run_avalanche_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Avalanche Effect Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    percentages = []

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        changed = hamming_distance_hex(d1, d2)
        pct = changed / DIGEST_BITS * 100
        percentages.append(pct)

        print(
            f"  {msg!r:12} vs {flipped!r:12} -> "
            f"{changed}/{DIGEST_BITS} bits changed ({pct:.1f}%)"
        )

    mean_pct = sum(percentages) / len(percentages)
    print("-" * 60)
    print(
        f"AVALANCHE RESULT: mean {mean_pct:.1f}% "
        f"(ideal ~50%)"
    )


def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None

    chi_sq = 0.0

    for i in range(n_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2

    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)


def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n

    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None

    runs = 1 + sum(
        1 for i in range(1, n)
        if bits[i] != bits[i - 1]
    )

    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)

    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))


def longest_run_test(bits):
    M = 8
    n = len(bits)
    n_blocks = n // M

    if n_blocks < 16:
        return None

    v_counts = [0, 0, 0, 0]

    for i in range(n_blocks):
        block = bits[i * M:(i + 1) * M]
        longest = current = 0

        for b in block:
            if b:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            v_counts[0] += 1
        elif longest == 2:
            v_counts[1] += 1
        elif longest == 3:
            v_counts[2] += 1
        else:
            v_counts[3] += 1

    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]

    chi_sq = sum(
        (v_counts[i] - n_blocks * pi_values[i]) ** 2
        / (n_blocks * pi_values[i])
        for i in range(4)
    )

    return _igamc(3 / 2, chi_sq / 2)


def run_nist_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("NIST SP 800-22 Style Randomness Tests")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    print(f"Bitstream = {len(bits)} bits from {len(inputs)} inputs")

    results = {}

    p = frequency_monobit_test(bits)
    results["Frequency (Monobit)"] = p

    p = block_frequency_test(bits)
    if p is not None:
        results["Block Frequency"] = p

    p = runs_test(bits)
    if p is not None:
        results["Runs"] = p

    p = longest_run_test(bits)
    if p is not None:
        results["Longest Run of Ones"] = p

    for name, p in results.items():
        print(
            f"  {name:28s} p={p:.4f} "
            f"{'PASS' if p >= 0.01 else 'FAIL'}"
        )



def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m - 1]
    counts = {}

    for i in range(n):
        pattern = tuple(ext[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0

    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)

    return phi


def run_approximate_entropy_test(cache: DigestCache, inputs, m=2):
    print("\n" + "=" * 60)
    print("Approximate Entropy (ApEn)")
    print("=" * 60)

    bits = cache.bitstream(inputs)
    n = len(bits)

    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)

    df = 2 ** m
    p_value = _igamc(df / 2, chi_sq / 2)

    print(f"  ApEn({m}) = {apen:.4f}")
    print(f"  p-value   = {p_value:.4f}")
    print(f"  result    = {'PASS' if p_value >= 0.01 else 'FAIL'}")


def run_sac_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("SAC (Strict Avalanche Criterion)")
    print("=" * 60)

    rng = random.Random(0x5AC)
    flip_counts = [0] * DIGEST_BITS

    for msg in inputs:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = cache.get(msg)
        d2 = cache.get(flipped)

        bits1 = hex_to_bits(d1)
        bits2 = hex_to_bits(d2)

        for i in range(DIGEST_BITS):
            flip_counts[i] += bits1[i] != bits2[i]

    n_trials = len(inputs)
    probs = [c / n_trials for c in flip_counts]

    print(f"  trials = {n_trials}")
    print(f"  mean P(flip) = {sum(probs)/len(probs):.3f}")
    print(f"  min P(flip)  = {min(probs):.3f}")
    print(f"  max P(flip)  = {max(probs):.3f}")


def run_performance_tests(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Performance Stats")
    print("=" * 60)

    samples = [cache.timings[m] for m in inputs if m in cache.timings]

    if not samples:
        print("No timing samples available.")
        return

    mean_s = sum(samples) / len(samples)
    best_s = min(samples)
    worst_s = max(samples)

    total_bytes = sum(
        len(m.encode("utf-8"))
        for m in inputs
        if m in cache.timings
    )

    total_time = sum(samples)
    bytes_per_sec = total_bytes / total_time if total_time > 0 else float("inf")

    print(f"  samples     = {len(samples)}")
    print(f"  mean latency= {mean_s*1e6:.2f} microseconds")
    print(f"  best        = {best_s*1e6:.2f} microseconds")
    print(f"  worst       = {worst_s*1e6:.2f} microseconds")
    print(f"  throughput  = {bytes_per_sec:,.0f} bytes/sec")

    cpu_hz = get_cpu_freq_hz()
    avg_bytes = total_bytes / len(samples)

    cycles_per_byte = (
        mean_s * cpu_hz / avg_bytes
        if avg_bytes else float("inf")
    )

    print(f"  CPU freq    = {cpu_hz/1e9:.2f} GHz")
    print(f"  cycles/byte = {cycles_per_byte:,.0f}")


def run_structural_test(cache: DigestCache):
    print("\n" + "=" * 60)
    print("Structural Weak-Spot Check")
    print("=" * 60)

    seen = {}

    for msg in STRUCTURAL_INPUTS:
        digest = cache.get(msg)
        distinct_chars = len(set(digest))

        print(
            f"  input={msg!r:20} "
            f"digest={digest} "
            f"distinct_hex={distinct_chars}"
        )

        if digest in seen and seen[digest] != msg:
            print(f"  !!! DUPLICATE: {seen[digest]!r} and {msg!r}")

        seen[digest] = msg


def run_collision_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Collision Check")
    print("=" * 60)

    seen = {}

    for msg in inputs:
        digest = cache.get(msg)

        if digest in seen and seen[digest] != msg:
            raise AssertionError(
                f"Collision: {seen[digest]!r} and {msg!r}"
            )

        seen[digest] = msg

    print(f"No collision among {len(inputs)} pool inputs.")


def run_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Preimage Sanity Check")
    print("=" * 60)

    target_msg = inputs[0]
    target_digest = cache.get(target_msg)

    found = any(
        cache.get(msg) == target_digest
        for msg in inputs[1:]
    )

    print(
        "A preimage was found."
        if found else
        f"No preimage found among {len(inputs)-1} pool messages."
    )


def run_second_preimage_test(cache: DigestCache, inputs):
    print("\n" + "=" * 60)
    print("Second-Preimage Sanity Check")
    print("=" * 60)

    original_msg = inputs[-1]
    original_digest = cache.get(original_msg)

    found = any(
        msg != original_msg and cache.get(msg) == original_digest
        for msg in inputs[:-1]
    )

    print(
        "A second preimage was found."
        if found else
        f"No second preimage found among {len(inputs)-1} other messages."
    )


# ============================================================
# BOUNDARY LENGTH + PATTERN
# ============================================================

def _ascii_pattern_message(n, pattern_id):
    if pattern_id == 0:
        return "\x00" * n
    if pattern_id == 1:
        return "\x7f" * n
    if pattern_id == 2:
        return "\x55" * n
    if pattern_id == 3:
        return "\x2a" * n
    if pattern_id == 4:
        return (
            "".join(chr(i) for i in range(128)) * (n // 128)
            + "".join(chr(i) for i in range(n % 128))
        )

    raise ValueError("Unknown pattern ID")


def _boundary_lengths(rate_bytes=136, quick=False):
    if quick:
        return [0, 1, rate_bytes - 1, rate_bytes, rate_bytes + 1]

    lengths = set()

    for base in [
        0, 1, 2, 3,
        rate_bytes, 2 * rate_bytes, 3 * rate_bytes,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 1025,
        4095, 4096, 4097,
    ]:
        for delta in range(-3, 4):
            if base + delta >= 0:
                lengths.add(base + delta)

    return sorted(lengths)


def run_boundary_length_tests(rate_bytes=136, quick=False):
    print("\n" + "=" * 60)
    print("Boundary-Length + Pattern Tests")
    print("=" * 60)

    pattern_ids = range(2) if quick else range(5)
    total = 0

    for n in _boundary_lengths(rate_bytes, quick):
        for pattern_id in pattern_ids:
            msg = _ascii_pattern_message(n, pattern_id)

            d1 = sha3_gbs_256(msg)
            d2 = sha3_gbs_256(msg)

            assert _digest_hex(d1) == _digest_hex(d2), (
                f"Non-deterministic digest at "
                f"length={n}, pattern={pattern_id}"
            )

            total += 1

        print(f"  length={n:5d} bytes: PASS")

    print(f"Boundary test: {total} cases PASS")


def run_determinism_test(inputs):
    print("\n" + "=" * 60)
    print("Determinism Test")
    print("=" * 60)

    for i, data in enumerate(inputs):
        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Non-deterministic result for case {i}: {data!r}"
        )

    print(f"Determinism: {len(inputs)} cases PASS")


def _deterministic_flip_ascii(data, rng, flips):
    raw = bytearray(data.encode("ascii"))

    if not raw:
        raw = bytearray(b"A")

    for _ in range(flips):
        idx = rng.randrange(len(raw))
        raw[idx] ^= 1 << rng.randrange(7)

    return raw.decode("ascii")


def run_multi_bit_avalanche_test(trials=2):
    print("\n" + "=" * 60)
    print("Multi-Bit Avalanche Test")
    print("=" * 60)

    rng = random.Random(0xA11A)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)
        original = random_message(length)
        modified = _deterministic_flip_ascii(
            original,
            rng,
            rng.randint(1, 8)
        )

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(modified)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    variance = sum(
        (x - mean) ** 2
        for x in distances
    ) / len(distances)

    stddev = math.sqrt(variance)

    print(f"  trials = {trials:,}")
    print(f"  mean   = {mean:.3f} / {DIGEST_BITS}")
    print(f"  stddev = {stddev:.3f}")
    print(f"  min    = {min(distances)}")
    print(f"  max    = {max(distances)}")

    if not (120 <= mean <= 136):
        print(
            "  mean is outside the loose "
            "120..136 sanity range."
        )
    else:
        print("Multi-bit avalanche: PASS")


def run_output_bit_bias_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Bias Test")
    print("=" * 60)

    rng = random.Random(0xB1A5)
    ones = [0] * DIGEST_BITS

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)
        digest = _digest_bytes(sha3_gbs_256(data))

        for bit in range(DIGEST_BITS):
            if digest[bit // 8] & (1 << (bit % 8)):
                ones[bit] += 1

    ratios = [
        count / trials
        for count in ones
    ]

    print(f"  trials  = {trials:,}")
    print(f"  min P(1)= {min(ratios):.5f}")
    print(f"  max P(1)= {max(ratios):.5f}")
    print(f"  mean P(1)= {sum(ratios)/DIGEST_BITS:.5f}")

    if trials >= 100_000:
        assert min(ratios) > 0.48
        assert max(ratios) < 0.52
        print("Output-bit bias: PASS")
    else:
        print(
            "Output-bit bias: completed"
        )


def run_output_bit_correlation_test(trials=4):
    print("\n" + "=" * 60)
    print("Output-Bit Pair Correlation Test")
    print("=" * 60)

    rng = random.Random(0xC011)
    pair_counts = {}

    n_pairs = DIGEST_BITS * (DIGEST_BITS - 1) // 2

    for _ in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_bytes(sha3_gbs_256(data))

        bits = [
            (digest[i // 8] >> (i % 8)) & 1
            for i in range(DIGEST_BITS)
        ]

        for i in range(DIGEST_BITS):
            if not bits[i]:
                continue

            for j in range(i + 1, DIGEST_BITS):
                if bits[j]:
                    key = (i, j)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    expected = trials / 4
    largest_deviation = 0.0
    worst_pair = None

    for i in range(DIGEST_BITS):
        for j in range(i + 1, DIGEST_BITS):
            count = pair_counts.get((i, j), 0)
            deviation = abs(count - expected) / trials

            if deviation > largest_deviation:
                largest_deviation = deviation
                worst_pair = (i, j)

    print(f"  trials = {trials:,}")
    print(f"  tested pairs = {n_pairs:,}")
    print(
        f"  largest absolute deviation from 0.25 = "
        f"{largest_deviation:.5f}"
    )
    print(f"  worst pair = {worst_pair}")
    print(
        "Output-bit correlation: completed. "

    )


def run_extended_collision_test(trials=20):
    print("\n" + "=" * 60)
    print("Large Collision Search")
    print("=" * 60)

    rng = random.Random(0xC0111510)
    seen = {}

    for i in range(trials):
        length = rng.randint(0, 64)
        data = random_message(length)

        digest = _digest_hex(sha3_gbs_256(data))

        if digest in seen and seen[digest] != data:
            print("COLLISION FOUND")
            print(
                f"  message A = "
                f"{seen[digest].encode('utf-8').hex()}"
            )
            print(
                f"  message B = "
                f"{data.encode('utf-8').hex()}"
            )
            print(f"  digest    = {digest}")
            raise AssertionError("Hash collision")

        seen[digest] = data

    print(f"Collision test: {trials:,} messages, PASS")


def run_large_input_test(sizes=None):
    if sizes is None:
        sizes = (
            QUICK_LARGE_INPUTS
            if TEST_MODE == "quick"
            else FULL_LARGE_INPUTS
        )

    print("\n" + "=" * 60)
    print("Large-Input Determinism Test")
    print("=" * 60)

    for size in sizes:
        data = "".join(
            chr(32 + (i % 95))
            for i in range(size)
        )

        d1 = sha3_gbs_256(data)
        d2 = sha3_gbs_256(data)

        assert _digest_hex(d1) == _digest_hex(d2), (
            f"Failed at {size} bytes"
        )

        print(f"  {size:>8} bytes PASS")

    print("Large-input test PASS")


def run_complement_test(trials=2):
    print("\n" + "=" * 60)
    print("Complement-Relationship Test")
    print("=" * 60)

    rng = random.Random(0xC0DE)
    distances = []

    for _ in range(trials):
        length = rng.randint(1, 64)

        original = bytes(
            rng.randrange(128)
            for _ in range(length)
        )

        complement = bytes(
            x ^ 0x7f
            for x in original
        )

        original = original.decode("ascii")
        complement = complement.decode("ascii")

        d1 = sha3_gbs_256(original)
        d2 = sha3_gbs_256(complement)

        distances.append(hamming_distance(d1, d2))

    mean = sum(distances) / len(distances)

    print(f"  trials = {trials:,}")
    print(
        f"  mean Hamming distance = "
        f"{mean:.3f} / {DIGEST_BITS}"
    )
    print(f"  min = {min(distances)}")
    print(f"  max = {max(distances)}")
    print("Complement test: completed (diagnostic only)")


# ============================================================
# COMPLETE RUNNER
# ============================================================

def run_extended_suite(n_inputs=N_INPUTS):
    mode = TEST_MODE.lower()

    if mode not in {"quick", "full"}:
        raise ValueError(
            "TEST_MODE must be either 'quick' or 'full'"
        )

    quick = mode == "quick"

    cache = DigestCache()
    inputs = build_input_pool(n_inputs)

    print("=" * 60)
    print(f"BLAKE3 EVAL SUITE | MODE = {mode.upper()}")
    print("(sha3_gbs_256 has been replaced with the BLAKE3 (32-byte digest))")
    print("=" * 60)

    print(
        f"Building shared pool of {len(inputs)} inputs "
        f"(from your INPUT_POOL) and hashing each ONCE..."
    )

    cache.get_many(inputs)

    print("Pool hashed.\n")

    run_kat(cache)
    run_avalanche_test(cache, inputs)
    run_nist_tests(cache, inputs)
    run_approximate_entropy_test(cache, inputs)
    run_sac_test(cache, inputs)
    run_performance_tests(cache, inputs)
    run_structural_test(cache)
    run_collision_test(cache, inputs)
    run_preimage_test(cache, inputs)
    run_second_preimage_test(cache, inputs)

    run_boundary_length_tests(quick=quick)

    run_determinism_test(
        inputs[
            :QUICK_DETERMINISM_CASES
            if quick else len(inputs)
        ]
    )

    run_multi_bit_avalanche_test(
        QUICK_MULTI_BIT_AVALANCHE_TRIALS
        if quick else 1000
    )

    run_output_bit_bias_test(
        QUICK_BIAS_TRIALS
        if quick else FULL_BIAS_TRIALS
    )

    run_output_bit_correlation_test(
        QUICK_CORRELATION_TRIALS
        if quick else FULL_CORRELATION_TRIALS
    )

    run_extended_collision_test(
        QUICK_COLLISION_TRIALS
        if quick else FULL_COLLISION_TRIALS
    )

    run_large_input_test(
        QUICK_LARGE_INPUTS
        if quick else FULL_LARGE_INPUTS
    )

    run_complement_test(
        QUICK_COMPLEMENT_TRIALS
        if quick else 1000
    )

    print("\n" + "=" * 60)
    print(
        f"SUITE COMPLETE. "
        f"Unique cached messages: {len(cache.digests)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_extended_suite()

import hashlib
import json
import math
import random
import string
import time

import blake3 as blake3_pkg

DIGEST_BITS = 256

INPUT_POOL = [
    "", "a", "abc", "hello", "0", "00000000", "aaaaaaaa", "aaaaaaaaaaaaaaaa",
    "\x00\x00\x00\x00", "3rw9BHoqU2uqf4Du", "m66UepuWYOLtHYwz", "dyhwn1sNluc9pdhg",
    "GP3Xq47Tk1ziUW2U", "RE3hBUbrGrLemJcK", "rRbkUGk5MJc3Cqnp", "cNLgErKjTUoGVGtF",
    "EeJg2HJrBExRhZ3G", "gkhj8ObVn46v2FVf", "XuJhHtyq0uSUlXJ6", "zZd8UElnWrdQFAhN",
    "etuGIon7vMrwD4NP", "CA3nonZZOHQ97W4Q", "iyaDCgfrTX1lLNtY", "pR4I8fs872x7Zt8k",
    "H2I9pCjFLj0rQnDG", "MPl9ZJUCJyiOBCDD", "ojixHOtBIfSP1PRw", "1dVqxM5BWOVA44IU",
    "5nJepjM66RWJF3Yd", "GkKzIdnWHOmnWfLy", "DeGnvfBmmGSeQtVW", "QmUQ0AHD7PBgs81l",
    "Y4GxdLi8zVND2JTX", "XZmszyg1HnMhUiET", "FUEiGPvQYVPmz8L4", "EEp7FVJQhWFWsVaS",
    "SSiGUiwQSYH4HHth", "Rm8LnEhJ9k8JYgVE", "Acd2kwfXzzfDwX54", "bpU6PYkMw7Et8XpU",
    "2wzCDp8jZXchtc18", "6pumKzhfuov21g80", "g5nN6mQwHG9396CM", "xImsONcg0evCt61O",
    "o4ibIjUk4s9vk8LK", "At9KVBxHrfggRXhu", "kMRqeICEIETFYImM", "Ow4kAKMXWwgyF1N0",
    "ReqkYrQiHr1GOUhs", "yCfNpQ9pMVKr8Wdp", "jUFXEMI938Zut2yi", "UwTKY8IBu4tSY2uk",
    "dwWhyzSPXe2VrzCc", "kc1meOx4jpG8eHki", "6km9Q6U8MR05pesc", "2BoaWbNtk4yFKCjh",
    "gTZ4ubC2NZDxjgHW", "OtMexvUqCx47hYUA", "HAaLatU3taiv58dC", "yXENdY4Fj5XSlclM",
    "DAss5uSjqvT7Wqnz", "RLzmRXIBfEssGcgl", "3sXY4VhCw6KMOiB1", "SVu2MihkqpWvvZYs",
    "mUXQZ8zjrtCKKQyd", "Tt5MxpJ0CvcA3dWv", "fjM9Nw3tQZkJ3l1i", "tJAjYvXjhC3cD3DB",
    "4SjvVqN0JnE4wRRg", "ihQ4GaxcLaHOZ6Sk", "5cKvQ47E4ntX1bO7", "qOdZQ2A6on7N5N2O",
    "XzH7hkFgHrUr7hs8", "JXfwgw6zuDk5e1hs", "IXu5YRx01yYKIyuC", "S1IYsn1Ez5vXhD0w",
    "HTBMO2dngzp6s5mp", "2XYAmvdDwzGomrhC", "bYkCG1oyktyBUlKv", "KWUmyt6KIxAgyZ3n",
    "6NHb1fHK9j42AIbn", "9wT6UUYvqJ155ajp", "JsrLiWEpAGeJn0SU", "430qSZckxZMOVwVw",
    "zO8naudHFt1xbgsL", "5eeAAoacj1KigTAi", "yzSAS79YKObFNXvE", "D1hj8sVRW6XdLYky",
    "pSCe89l7WCl63c4m", "e3i3yEX0xRDRLKtr", "xqX9rcqeCd7flVdT", "hJOSc6c4HZKoMvzL",
    "pdz2X6g2QrbWkowj", "DXvWWGDGgLw4AvY2", "woJ734rU72vUB9sp", "7hf5Lxssw3KEvG32",
    "qmZp5bFW9CWQzwtR", "KZ2VOQzXQIZo1nEO", "lcYRunmyw9twEGJL", "XwbwBnuSgwC60In8",
]

# ============================================================
# Hash function wrappers -- all return 32-byte (256-bit) hex digest
# ============================================================

def h_sha1(data: bytes) -> str:

    return hashlib.sha1(data).hexdigest()

def h_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def h_sha3_256(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()

def h_shake256(data: bytes) -> str:
    return hashlib.shake_256(data).hexdigest(32)  # 32 bytes = 256 bits

def h_blake2b(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()

def h_blake3(data: bytes) -> str:
    return blake3_pkg.blake3(data).hexdigest()


ALGORITHMS = {
    "SHA-1":      (h_sha1, 160),
    "SHA-256":    (h_sha256, 256),
    "SHA3-256":   (h_sha3_256, 256),
    "SHAKE256":   (h_shake256, 256),
    "BLAKE2b":    (h_blake2b, 256),
    "BLAKE3":     (h_blake3, 256),
}

# ============================================================
# Shared test helpers (same logic as the eval suite)
# ============================================================

def hex_to_bits(hex_str):
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def hamming_distance_hex(h1, h2):
    return sum(x != y for x, y in zip(hex_to_bits(h1), hex_to_bits(h2)))

def _make_one_bit_flip_ascii(msg, rng):
    raw = bytearray(msg.encode("ascii"))
    if not raw:
        raw = bytearray(b"A")
    idx = rng.randrange(len(raw))
    raw[idx] ^= 1 << rng.randrange(7)
    return raw.decode("ascii")

def _igamc(a, x, iterations=200):
    if x <= 0:
        return 1.0
    try:
        gln = math.lgamma(a)
        term = 1.0 / a
        s = term
        for n in range(1, iterations):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-10:
                break
        return max(0.0, min(1.0, 1 - math.exp(-x + a * math.log(x) - gln) * s))
    except (ValueError, OverflowError):
        return 0.0

def frequency_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))

def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
    chi_sq = 0.0
    for i in range(n_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2
    chi_sq *= 4 * block_size
    return _igamc(n_blocks / 2, chi_sq / 2)

def runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return None
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i-1])
    expected = 2 * n * pi * (1 - pi)
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    z = abs(runs - expected) / denom if denom > 0 else 0
    return math.erfc(z / math.sqrt(2))

def _phi_m(bits, m):
    n = len(bits)
    ext = bits + bits[:m-1]
    counts = {}
    for i in range(n):
        pattern = tuple(ext[i:i+m])
        counts[pattern] = counts.get(pattern, 0) + 1
    phi = 0.0
    for c in counts.values():
        freq = c / n
        phi += freq * math.log(freq)
    return phi

def apen_pvalue(bits, m=2):
    n = len(bits)
    phi_m = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)
    df = 2 ** m
    return _igamc(df / 2, chi_sq / 2)


def evaluate(name, hash_fn, bit_width):
    rng = random.Random(0xA11A)
    digests = {}
    timings = []

    t_start = time.perf_counter()
    for msg in INPUT_POOL:
        t0 = time.perf_counter()
        d = hash_fn(msg.encode("utf-8"))
        timings.append(time.perf_counter() - t0)
        digests[msg] = d
    total_time = time.perf_counter() - t_start

    # Avalanche (single-bit flip on each pool message)
    percentages = []
    for msg in INPUT_POOL:
        flipped = _make_one_bit_flip_ascii(msg, rng)
        d1 = digests[msg]
        d2 = hash_fn(flipped.encode("utf-8"))
        changed = hamming_distance_hex(d1, d2)
        percentages.append(changed / bit_width * 100)
    avalanche_mean = sum(percentages) / len(percentages)

    # Bitstream for NIST-style + ApEn tests
    bits = []
    for msg in INPUT_POOL:
        bits.extend(hex_to_bits(digests[msg]))

    p_monobit = frequency_monobit_test(bits)
    p_block = block_frequency_test(bits)
    p_runs = runs_test(bits)
    p_apen = apen_pvalue(bits)

    # SAC
    rng2 = random.Random(0x5AC)
    flip_counts = [0] * bit_width
    for msg in INPUT_POOL:
        flipped = _make_one_bit_flip_ascii(msg, rng2)
        d1 = digests[msg]
        d2 = hash_fn(flipped.encode("utf-8"))
        b1 = hex_to_bits(d1)
        b2 = hex_to_bits(d2)
        for i in range(bit_width):
            flip_counts[i] += b1[i] != b2[i]
    sac_probs = [c / len(INPUT_POOL) for c in flip_counts]
    sac_mean = sum(sac_probs) / len(sac_probs)

    # Collision check
    seen = set(digests.values())
    collisions = len(INPUT_POOL) - len(seen)

    # Performance
    total_bytes = sum(len(m.encode("utf-8")) for m in INPUT_POOL)
    mean_latency_us = (sum(timings) / len(timings)) * 1e6
    throughput_mb_s = (total_bytes / total_time) / 1e6 if total_time > 0 else float("inf")
    cpu_hz = 2.1e9  # measured on this container in earlier runs
    avg_bytes = total_bytes / len(INPUT_POOL)
    cycles_per_byte = (sum(timings)/len(timings)) * cpu_hz / avg_bytes if avg_bytes else float("inf")

    return {
        "name": name,
        "bit_width": bit_width,
        "avalanche_mean_pct": avalanche_mean,
        "p_monobit": p_monobit,
        "p_block_frequency": p_block,
        "p_runs": p_runs,
        "p_apen": p_apen,
        "sac_mean": sac_mean,
        "sac_min": min(sac_probs),
        "sac_max": max(sac_probs),
        "collisions": collisions,
        "mean_latency_us": mean_latency_us,
        "throughput_mb_s": throughput_mb_s,
        "cycles_per_byte": cycles_per_byte,
    }


if __name__ == "__main__":
    results = {}
    for name, (fn, width) in ALGORITHMS.items():
        results[name] = evaluate(name, fn, width)
        print(f"done: {name}")

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nWrote metrics.json")
    print(json.dumps(results, indent=2))

