# neo-modint
## What it is
Modular integer arithmetic for the lattice modulus q (a Mersenne prime like 2^61 - 1), used as coefficients in rings.

## How it is used in the paper
q is the lattice modulus for Ajtai commitments (Section 3.2, Definition 10). Modints handle arithmetic in Z_q (Section 2.3), with operations preserving norms (Lemma 3). Supports small fields without emulation (Section 1.2).

## What we have
- `ModInt` with add/sub/mul/neg/add_assign/etc., using wrapping for efficiency.
- Conversions from/to u64/i128 (signed/unsigned).
- `Coeff` trait impl for use in polys/rings.
- Random sampling and inverse.
- Tests for ops and overflow.
- QuickCheck arbitrary instance (optional feature).

## What we are missing
- Batch operations or SIMD optimizations (for performance in large commitments).
- Support for other q values (paper mentions flexible q, Section 3.2).
- Property tests for modular arithmetic laws.
- Integration with external crates like `num-bigint` for larger moduli.
