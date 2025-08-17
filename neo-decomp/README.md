# neo-decomp
## What it is
Base-b decomposition of field vectors into low-norm matrices for lattice embedding.

## How it is used in the paper
Decomposition maps vectors to low-norm matrices (Section 3.2, Definition 11). Enables pay-per-bit commitments (Section 1.3) and linear homomorphism for folding (Section 3.3, Lemma 2). Used in Π_DEC reduction (Section 4.2).

## What we have
- `signed_decomp_b` to decompose vector into k layers of low-norm vectors (signed, ||digits||_inf <= (b-1)/2).
- `reconstruct_decomp` for roundtrip reconstruction.
- Norm bounds enforcement in decomposition.
- Signed decomposition variant (better norms, inspired by LatticeFold).
- Gadget vector support (full Definition 11, with g for MSIS binding).
- Direct matrix output (using RowMajorMatrix for decomposed digits).
- Property tests for decomposition invariants (e.g., reconstruction, norm bounds).
- Tests for roundtrip.

## What we are missing
- Multivariate or batch decomp for lookups (Section 1.4).
