# TODO.md

## Overview

Updated TODO reflecting current state: Sum-check fully functional with Poseidon2 FS, ZK blinding, and extension fields; CCS supports structures/instances/witnesses with batched norm checks; folding implements Π_CCS/RLC/DEC with end-to-end proof/verify for two instances (toy params). PCS/openings now use full FRI (replaced FnOracle stubs). Focus on remaining security (full extractors), optimizations (faster mul for very large n), extensions (IVC/lookups), and production hardening (constant-time, audits). Aim: Secure, scalable IVC with 128-bit params.
This version marks items: [x] fully done, [~] partial, [ ] pending. Removed NTT/FFT references per design avoidance (replaced with alternatives like Karatsuba). Last updated: 2025-08-16.

## High Priority (Core Functionality Gaps)

- [x] Fix `construct_q` in `neo-fold` for proper CCS multilinear encoding (§4.4).
- [x] Complete `batch_norm_checks` in `neo-fold` for multivariate norm indicators (§4).
- [x] Finish verifier protocols in `neo-fold` (verify_ccs/rlc/dec).
- [x] Integrate commitment openings in `AjtaiCommitter` with folding verifiers (§3.2). (Wired PCS; replaced FnOracle.)
- [x] Unify extension-field plumbing across Π_CCS + sum-check + FRI.

## Medium Priority (Security & Correctness)

- [x] Add ZK blinding.
- [x] Implement Poseidon2 Fiat-Shamir in `neo-sumcheck`/transcripts.
- [ ] Support Mz linear transforms in sum-check batching (§3.3). (Partial: Tables in ccs_sumcheck_prover.)
- [x] Fix extension field handling in sum-check/evals (degree 2 for Goldilocks).
- [x] Add proper transcript handling.
- [x] Implement real GPV sampling in AjtaiCommitter (§3.2, Alg. 1).
- [x] Implement full multi-query FRI with RS encoding (for rate<0.5).
- [x] Enable ZK blinding in commitments (Gaussian errors, σ=3.2).
- [x] Add per-round FS labels and re-entrant transcripts.
- [~] Implement knowledge soundness extractors via rewinding. (Partial: Stub in place, needs full rewinding logic.)
- [x] Use `NeoChallenger` everywhere to avoid framing mistakes.
- [x] Fix phi_of_w denom for odd/even b consistency.
- [ ] Add full constant-time ops across crates (§8: no secret-dependent branches).
- [ ] Scale to secure params (n=8192, validate MSIS/RLWE >128 bits via Sage).

## Low Priority (Optimizations/Extensions)

- [~] Add IVC recursion loop in `neo-fold` (§1.5). (Partial: Stubbed; needs full verifier circuit folding.)
  - Wrap folder to prove verifier circuit.
  - Test: Chain 10 folds; verify final.
- [x] Optimize large instances in `neo-ring`.
  - Karatsuba multiplication for faster large-n ring ops.
- [ ] Add lookups (§1.4) and FRI compression (§1.5).
- [x] Optimize poly_mul in ccs_sumcheck_prover (O(d²) -> better with Karatsuba).
- [ ] Optimize RS to O(n log n) with FFT.
- [ ] Add Toom-3 or tuned Karatsuba for unis/rings (target O(n log n) without FFT).

## Tests

- [x] Full folding integration: Fold two CCS; assert final satisfies + verification passes.
- [x] Invalid witness: Fold with bad witness; assert fails.
- [x] Norm violation: High-norm; assert rejects.
- [ ] Large instance: Fold 2^10 CCS; time it.
- [~] RLC homomorphism: Folded commit = ρ-combo. (Partial: test_batch_norm_checks_poly.)
- [ ] Property (QuickCheck): Random CCS; folding preserves satisfiability.
- [ ] Soundness: Simulate bad proofs; verifier rejects high-prob.
- [ ] Cross-protocol: Folded instance valid in original CCS.
- [x] Basic unit tests for modules (fields, polys, rings, etc.).
- [x] QuickCheck properties for rings/polys.
- [x] Large CCS satisfiability (1024 constraints).
- [x] Ring operation benchmarks.
- [ ] Test scaled batching in pi_ccs (rho-norm combination).
- [ ] Soundness sim: Forge sum-check msgs; check verifier rejection prob >1-1/|F|.
- [ ] Add soundness simulations (forge proofs, rejection prob >1-1/|F|).
- [x] Ensure constant-time ops (audit norm_inf/sampling).
- [ ] Test full extractor rewinding on transcripts.

## Misc

- [x] Update params to paper samples (§6) for realism.
- [ ] Add paper-section comments in code.
- [x] Fix issues: Multilinear padding bugs; replace panics with Results; remove hardcoded RNG.
- [x] Document parameter choices (partial in READMEs).
- [x] Add domain separation labels to FS (e.g., "round\_{i}" for challenges).
- [x] Validate params with Sage for 128-bit MSIS/RLWE.
- [ ] Compute pay-per-bit metrics.
- [x] RNG policy: Remove hardcoded seeds outside tests; derive from transcripts.
- [ ] Proof objects: Replace raw transcripts with structured proofs.

## Changelog vs Previous

- Marked FRI/PCS as [x] (full impl in neo-sumcheck, integrated in neo-fold).
- Updated poly_mul opt to [x] (Karatsuba tuned).
- Error handling (panics to Results) [x].
- Unify extension fields [x].
- Verifiers in neo-fold [x].
- Tests: Added more, including error cases and long-running guards.
- Updated date to 2025-08-16.
- Added items for constant-time, secure params, and Toom-3 opts (still pending).

## Short Next-Steps

1. Complete extractors and IVC recursion for full security.
2. Add lookups for extended CCS.
3. Benchmark large folds (n=1024) with secure params.
4. Implement Toom-3 for further mul speedup.
