# neo-fields
## What it is
This module provides field arithmetic wrappers, primarily around the Goldilocks prime field (2^64 - 2^32 + 1) and its quadratic extension for sum-check security.

## How it is used in the paper
Neo uses small primes like Goldilocks for efficient arithmetic (Section 1.2, Section 6). The quadratic extension (degree 2) ensures 128-bit security for sum-check (Section 8). Fields are used in CCS constraints (Definition 16-17), multilinear extensions (Appendix A), and sum-check over extensions (Section 4.1).

## What we have
- `F` alias for Goldilocks field with basic ops and inverse.
- `ExtF` quadratic extension (x^2 + 1 = 0) with add/sub/mul/neg/inv.
- Random sampling for both `F` and `ExtF`.
- Norm helpers for extension elements (e.g., abs_norm for bounding).
- Tests for inverse roundtrips and extension ops.

## What we are missing
- Full field trait implementations (e.g., `p3-field::ExtensionField` integration for better composability).
- Support for other small fields (e.g., M31 as mentioned in Section 1.2).
- Faster polynomial ops (needed for large multilinear extensions in Appendix A).
- Security proofs for non-square in extension (assumes -1 is non-square in Goldilocks).
- QuickCheck property tests for field laws.
