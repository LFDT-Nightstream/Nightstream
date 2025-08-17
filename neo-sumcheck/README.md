# neo-sumcheck
## What it is
Batched sum-check protocol for multivariate polynomials over field extensions, with specialized support for multilinear polynomials. This is a core building block for reducing claims about polynomial sums (e.g., over the Boolean hypercube) to random-point evaluations.
NOTE: This crate provides *generic* sum-check tools only. CCS-specific proving lives in the `neo-ccs` crate; keep this crate relation-agnostic for reuse in folding, lookups, etc.

## How it is used in the paper
Sum-check reduces claims to evaluation claims (Section 4.1). It supports batching for efficiency and is central to the folding scheme (Theorem 1, Section 4) and extensions like lookups/memory checks (Section 1.4). Runs over small-field extensions for security (Section 1.3).

## What we have
- Batched prover and verifier for general multivariate polynomials, sending univariate polynomials per round (supports multiple claims/polys via random linear combinations).
- Specialized efficient implementations for multilinear polynomials: multilinear_sumcheck_prover/verifier and batched_multilinear_sumcheck_prover/verifier, which use folding over evaluation tables (avoids exponential time for multilinear cases).
- Efficient prover for general/high-degree multivariate polynomials: Uses precomputed tables, folding, and cross-inner products (O(N log N) for N-sized instances).
- Flexible UnivPoly trait for representing polynomials (e.g., as closures for arbitrary multivariates or MultilinearEvals for evaluation tables over the Boolean hypercube).
- Fiat-Shamir transform using Poseidon2 hashing for full extension-field challenges, deriving randomness from transcripts and supporting batching with rho.
- ZK blinding with Gaussian samples.
- Tests covering roundtrip correctness for multilinear/low-degree/high-degree polynomials, prover rejection on invalid claims, verifier rejection on mismatches.

## What we are missing
- Optimizations for large degrees/instances: No degree reduction for high-degree unis, or batching across multiple independent sum-checks (e.g., for IVC recursion in Section 5).
- Property and soundness tests: Randomized tests for verifier rejection probability on invalid proofs, edge cases (e.g., zero polynomial, empty claims).
- Multivariate-specific features: No extensions for lookups (Shout/Twist in Section 1.4), but closures support general multivariate polynomials via `UnivPoly`.
- Error handling and robustness: Fixed MLE padding to always zero-pad up.
