# neo-poly
## What it is
A simple univariate polynomial library over generic coefficient types (e.g., fields or modints), supporting basic arithmetic.

## How it is used in the paper
Polynomials represent ring elements in cyclotomic rings (Section 3.2, Definition 5). Used for multilinear extensions (Appendix A), sum-check univariate reductions (Section 4.1), and constraint polynomials in CCS (Definition 16). The paper runs sum-check over extensions of small primes (Section 1.3).

## What we have
- `Polynomial<C>` with add/sub/mul, eval, degree, coeff access.
- Generic over `Coeff` trait (supports `F`, `ExtF`, `ModInt`).
- Auto-trimming of leading zeros.
- Division/quotient-remainder.
- Naive Lagrange interpolation.
- Karatsuba multiplication for efficiency.
- Tests for mul, eval, div_rem, and interpolation correctness.
- QuickCheck arbitrary instance and property tests for ring axioms (optional feature).

## What we are missing
- Efficient multiplication algorithms for large degrees (e.g., FFT/NTT).
- Multivariate support (CCS uses multivariate f over s vars, Section 4.1).
- Roots finding (for multilinear extensions in Appendix A).
