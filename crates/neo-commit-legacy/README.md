# neo-commit
## What it is
Ajtai-based lattice commitment scheme for vectors over rings, with linear homomorphism.

## How it is used in the paper
Core commitment for CCS witnesses (Section 3: Neo's folding-friendly commitments). Provides binding/hiding with low-norm openings (Definition 10). Supports folding multilinear claims (Section 3.3, Corollary 1) and pay-per-bit costs (Section 1.3).

## What we have
- `AjtaiCommitter` with setup, commit (deterministic/RNG variants), verify, random linear combo (scalar/rotation).
- Packing decomposed matrices into ring elements.

- ZK blinding with Gaussian errors (σ=3.2).
- Open/verify at multilinear points with proofs.
- Preset params for Goldilocks (n=64, etc.) and secure (n=8192).
- Pay-per-bit cost calculation.
- Tests for roundtrip, ZK randomization, and openings.

## What we are missing
- Full knowledge soundness extractor (Section 2.2, Definition 3).
- MSIS-based security reduction (Section 3.1, binding from MSIS).
- Sparse vector support (for lookups, Section 1.4).
- Parameter search/validation (Appendix B.10; partial via Sage).
- Integration with sum-check for efficient openings (currently toy).
