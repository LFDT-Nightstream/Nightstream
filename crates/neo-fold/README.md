# neo-fold
## What it is
Folding reductions for CCS instances: Π_CCS, Π_RLC, Π_DEC.

## How it is used in the paper
Core folding protocol (Section 4: Reductions). Π_CCS to eval claims (Section 4.1), Π_RLC for batching (Section 4.2), Π_DEC for decomposition (Section 4.2). Composes to full scheme (Theorem 1, Section 5).

## What we have
- `FoldState` to track instances, generate proofs, and verify.
- `pi_ccs`, `pi_rlc`, `pi_dec` functions with integration to sum-check.
- Verifier protocols (`verify_ccs`, `verify_rlc`, `verify_dec`).
- Multilinear extension and Q poly as closures.
- FRI stubs for compression.
- IVC recursion stub (simple loop).
- Tests for individual reductions, full flow, invalid cases, and transcript mutations.

## What we are missing
- Full composition and recursion (Section 5 properties: restricted, composable); IVC needs Spartan+FRI.
- Extractors for knowledge soundness (Section 5, Definition 19).
- Lookup/memory extensions (Section 1.4).
- Committed oracle access in verifiers (currently direct; needs full ZK).
- Benchmarks for large chains (e.g., 10 folds).
