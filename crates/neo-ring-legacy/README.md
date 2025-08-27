# neo-ring
## What it is
Cyclotomic ring elements (Z_q[x] / (x^n + 1)) for lattice-based commitments, with arithmetic and sampling.

## How it is used in the paper
Rings embed decomposed vectors for commitments (Section 3.2, Definition 11). Operations are module homomorphisms (Section 3.3). Used in norm checks (Section 4.1) and pay-per-bit costs (Section 1.3). Power-of-2 cyclotomics for Goldilocks (Section 6).

## What we have
- `RingElement<C>` with add/mul/sub/neg, reduction mod x^n +1.
- From coeffs/scalar, coeffs access.
- Infinity norm (signed).
- Random uniform/small/Gaussian sampling.
- Rotation and automorphism operations.
- Coefficient decomposition integration with neo-decomp.
- Toy commit function.
- Tests for add/mul closure, norms, sampling, reductions, rotation/automorphism, and decomposition.
- QuickCheck properties for ring axioms, reductions, and coefficient decomposition.
- Benchmarks for ring multiplication and coefficient decomposition.

## What we are missing
- Efficient multiplication for large n (e.g., NTT for O(n log n)).
- Full invertibility checks in production paths.
