# neo-main
## What it is
This is the entry point crate for the Neo lattice-based folding scheme implementation. It ties together all other modules to demonstrate end-to-end folding of CCS instances, including commitment, satisfiability checks, and folding reductions.

## How it is used in the paper
In the Neo paper (Section 4: Folding scheme for CCS, Section 5: Properties), this serves as the high-level protocol for folding two CCS instance-witness pairs into one using reductions like Π_CCS (CCS to evaluation claims), Π_RLC (random linear combinations), and Π_DEC (decomposition). It enables IVC/PCD constructions (Section 1.5) by recursively folding proofs without elliptic curves, leveraging lattice commitments over small fields.

## What we have
- Main executable that sets up a toy CCS structure, generates witnesses and instances, commits using Ajtai, checks satisfiability, and performs folding via `pi_ccs`, `pi_rlc`, and `pi_dec`.
- IVC recursion stub (simple loop for chaining folds).
- Integration tests for roundtrip commitment and folding flow.
- Dependencies on all other neo-* crates for a complete demo.

## What we are missing
- Full IVC/PCD recursion with verifier circuit proving (currently simple loop; needs Spartan+FRI for compression as in Section 1.5).
- Support for lookups and read-write memory (Section 1.4, using Shout/Twist).
- Fiat-Shamir transcript for non-interactive proofs (currently partial Poseidon2).
- Benchmarks and large-scale tests (e.g., 2^20 instances as in performance claims).
- Error handling and security parameter validation.
