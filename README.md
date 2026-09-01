
# SHA-GBS: Gaussian Boson Sampling Based Hash and Permutation Analysis

This project explores a Gaussian Boson Sampling (GBS)-inspired approach for cryptographic permutation and hash-function experimentation.

The repository contains two main implementations:

- A GBS-based permutation comparison framework.
- A SHA3-style hash construction using a GBS-inspired permutation.

The project evaluates the proposed GBS-based approach alongside multiple existing cryptographic permutations and hash functions using diffusion, avalanche, SAC, BIC, statistical, structural, performance, and other tests.


---

## Project Structure

```text
gbs-sha/
│
├── src/
│   ├── gbs_permutation_comparison.py
│   └── sha3_gbs_hash_comparison.py
│
├── results/
│   ├── csv/
│   └── plots/
│
├── README.md
└── requirements.txt
````

---


# Main Files

## `src/gbs_permutation_comparison.py`

This is the main permutation comparison and evaluation program.

It implements and evaluates multiple cryptographic permutations, including a GBS-derived permutation approach.

The experiments include:

* Correctness and determinism checks
* Performance benchmarking
* Avalanche testing
* SAC testing
* BIC testing
* First-round diffusion analysis
* Differential screening
* Linear screening
* Algebraic projection analysis
* Boolean nonlinearity
* Statistical tests
* Shannon entropy
* Structural analysis
* Fixed-point analysis
* Complementarity analysis
* Invariant-bit analysis
* Low-round analysis
* SHA3 sponge validation
* SHAKE256 validation
* Long-message tests
* GBS sampling experiments
* GBS-1600 permutation and inverse validation

The generated results are saved as CSV files and visualized using plots.

---

## `src/sha3_gbs_hash_comparison.py`

This file contains the SHA3-GBS-256 experimental hash implementation.

The design follows a sponge-style structure with:

* A 1600-bit state
* 25 lanes of 64 bits
* SHA3-style padding
* Absorbing phase
* GBS-inspired permutation
* Squeezing phase
* 256-bit hexadecimal output

The GBS-inspired transformation uses:

* Classical bit encoding
* Photonic state construction
* Gaussian Boson Sampling components
* Interferometer operations
* Three-mode correlation extraction
* Deterministic round constants
* Avalanche mixing

The file also includes an extended evaluation suite for the hash implementation.

---



# GBS Experiments

The project also includes experiments specifically related to the GBS-based construction.

These include:

* GBS sampling
* 8-bit mapping experiments
* 1600-bit sample mapping
* Hamming-distance analysis
* Permutation/inverse validation

Related CSV files:

```text
results/csv/gbs_sampling.csv
results/csv/gbs_8bit_complete_mapping.csv
results/csv/gbs_1600_sample_mapping.csv
```

Related plots:

```text
results/plots/20_gbs_sampling.png
results/plots/21_gbs_8bit_mapping.png
results/plots/29_gbs_1600_mapping_hamming.png
```

---


# Usage

Run the permutation comparison:

```bash
python src/gbs_permutation_comparison.py
```

Run the SHA3-GBS hash implementation and evaluation:

```bash
python src/sha3_gbs_hash_comparison.py
```

Depending on the enabled experiments and installed dependencies, execution may take time because the project performs multiple statistical and cryptographic tests.

---

# Results

All numerical experiment outputs are stored in:

```text
results/csv/
```

All generated visualizations are stored in:

```text
results/plots/
```

The CSV files contain the raw or summarized measurements, while the plots provide visual comparisons between the tested algorithms and the GBS-derived approach.

The scorecard and registry files provide an overview of the implemented tests and evaluated algorithms:

```text
results/csv/scorecard.csv
results/csv/registry.csv
```

---

# Notes

This repository is an experimental implementation for studying GBS-inspired cryptographic transformations and comparing their observable properties with established cryptographic algorithms.

The results included in this repository should be interpreted as experimental measurements from the implemented test suite.


---

