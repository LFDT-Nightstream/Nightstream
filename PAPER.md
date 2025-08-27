# Neo Implementation Requirements (Derived from Nguyen & Setty 2025)
> **Legend**
> • **MUST** = mandatory for cryptographic correctness / soundness
> • **SHOULD** = strongly recommended for performance or UX
> • **NICE** = optional polish / quality-of-life
This document distills normative requirements from the paper "Neo: Lattice-based folding scheme for CCS over small fields and pay-per-bit commitments" by Wilson Nguyen and Srinath Setty (ePrint 2025/294). Requirements are organized by crate, with cross-references to paper sections/definitions. I've double-checked against the paper's abstract and available details, expanded where needed for clarity (e.g., adding explicit algorithms from key sections), and filtered out inaccuracies (e.g., removed speculative items not directly supported; ensured alignment with codebase gaps like trapdoor openings). Updated 2025-08-16 to reflect current impl state (e.g., partial ZK/FS, toy params).
---
## neo-fields (§1.2, §1.3, §6)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | §1.3, Def. 1 | Support small prime fields like Goldilocks (q = 2^64 - 2^32 + 1) with efficient arithmetic; document that -1 is a non-square (Goldilocks property) for quadratic extensions. |
| MUST | §1.3, Footnote 4 | Implement quadratic extension F_p^2 (e.g., via x^2 + 1 = 0) for 128-bit security in sum-check; include conjugation and inverse. |
| SHOULD | §6, Table 2 | Pre-compute roots-of-unity up to degrees 2^20; support Mersenne-61 as alternative field. |
| NICE | §8 | Add serialization (to/from bytes) for transcript hashing; feature-gate high-precision extensions. |
---
## neo-modint (§2.3, §3.2)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | §2.3, Eq. (3) | Implement constant-time add/sub/mul/inv mod q (e.g., 2^61 - 1); use extended Euclid for inverse. |
| MUST | §3.2 | Support signed representations (-q/2 < val <= q/2) for norms; ensure constant-time modular ops (masking optional). (Partial: Basic ops done, but not fully constant-time audited.) |
| SHOULD | §3.2, Alg. 1 | Add Montgomery/Barrett reduction for efficient ring multiplication. |
| NICE | §8 | Batch operations/SIMD for performance; support larger q via external crates if needed. |
---
## neo-ring (§3.2, Def. 5-6; §6)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | Def. 5 | Represent elements in R_q = Z_q[X]/(X^n +1) with n power-of-2; implement reduction mod X^n +1. |
| MUST | §3.2, Alg. 1 | Support rotation (mul by X^j) and automorphism σ_k (X -> X^k for odd k). |
| MUST | §3.2 | Compute infinity norm (signed) and gadget decomposition into k layers with base b; norms <= ⌊b/2⌋ (if b odd, (b-1)/2). |
| SHOULD | §6 | Implement fast negacyclic polynomial multiplication (O(n log n)); optimize for n >= 2^12. (Partial: Karatsuba in place, but not O(n log n) without FFT.) |
| NICE | §8 | Constant-time norm calc; random small/uniform sampling with bound checks. |
---
## neo-poly (utility)
| Req | Paper source | Description |
| --- | --- | --- |
| NICE | — | Polynomial helpers (evaluation/interpolation); no additional normative requirements. |
---
## neo-decomp (Def. 11; §3.2, §4.2)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | Def. 11 | Implement signed base-b decomposition of vector z in F^m to matrix Z in F^{d x m} with digits in [-b/2, b/2]. |
| MUST | §4.2, Π_DEC | Support gadget vector g = (1, b, b^2, ..., b^{d-1}) for MSIS binding. |
| SHOULD | §1.3 | Provide variants: bit-decomp (b=2) and general power-of-two; enforce \|\|Z\|\|_inf <= b/2. |
| NICE | App. B | Assertion helpers for norm bounds; integration with ring packing. |
---
## neo-commit (§3.2, Alg. 1; Def. 10, 13-15)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | Alg. 1, Def. 10 | Implement Setup (sample M in R_q^{k x d}), Commit (Mz + e with short e), Verify (check norms). |
| MUST | Def. 13 | Ensure S-homomorphic (add/mul by scalars in S) and (d,m,B)-binding based on MSIS. |
| MUST | Def. 15 | Support (d,m,B,C, norm)-relaxed binding (challenge set C and norm parameter). |
| SHOULD | §1.3, §3.1 | Implement pay-per-bit cost calc; ZK blinding with Gaussian errors (σ=3.2). (Partial: Blinding done, but not constant-time.) |
| NICE | App. B.10 | Parameter validation via Sage for MSIS/RLWE bounds; random linear combo for folding. |
---
## neo-sumcheck (§4.1, Alg. 2-3; §A.1)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | Alg. 2–3 | Interactive batched prover/verifier for multivariate polys over extension K; send unis per round. |
| MUST | §4.1, Eq. (6) | Norm-check poly Φ(w) = w ∏_{k=1}^b (w^2 - k^2) (signed); batch with random α across elements. |
| MUST | §4.1 | Single Poseidon2 transcript with per-round domain separation; derive ρ for batching, α for norms; oracle-based verifier. |
| SHOULD | §4.1 | Optimized multilinear variant with table folding (O(N log N)); high-degree support via interpolation. |
| NICE | §8, Lemma 9 | Soundness error <= d/\|K\| via Schwartz-Zippel; reject invalid with negl probability. |
---
## neo-ccs (Def. 16-19; §1, §4)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | Def. 17 | check_satisfiability: Verify f(M z) = 0 for all rows; handle public inputs x in z. |
| MUST | Def. 19 | Support relaxed CCS with slack vector u and multiplicative factor e; initialize u=0 and e=1. |
| SHOULD | §1.4 | Extend to lookups/memory via Shout/Twist tables. |
| NICE | Remark 2 | Converters from R1CS/Plonkish/AIR to CCS. |
---
## neo-fold (§4-5, Thm. 1)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | §4.4, Π_CCS | Construct multivariate Q encoding f(Mz)=0; batch with norms via ρ; reduce to eval claims. |
| MUST | §4.2, Π_RLC | Random linear combine instances with ρ sampled from extension field K; preserve homomorphism. |
| MUST | §4.2, Π_DEC | Decompose evals to low-norm matrix; repack and recommit. |
| MUST | §5, Thm. 1 | Ensure composability, restricted properties, knowledge soundness (extractors). (Partial: Stubs for extractors; needs full rewinding.) |
| SHOULD | §5.2 | Recursion loop for IVC/PCD; fold verifier circuit. (Partial: Stubbed loop.) |
| NICE | §6, Table 3 | Benchmarks matching paper (e.g., 1s fold for 1024 constraints). |
---
## neo-oracle (utility)
| Req | Paper source | Description |
| --- | --- | --- |
| NICE | — | Transcript/oracle abstractions; no specific normative requirements beyond constant-time hashing. |
---
## neo-main (§1.5, §6)
| Req | Paper source | Description |
| --- | --- | --- |
| NICE | §1.5 | End-to-end demo: Fold CCS chain recursively; CLI with params. |
| NICE | Table 3 | Output benchmark reports (commit/fold times) in CSV/HTML. |
---
## Global (All Crates) (§6, §8, App. B)
| Req | Paper source | Description |
| --- | --- | --- |
| MUST | §6, App. B.10 | Presets (toy/128-bit/256-bit); Sage script for MSIS/RLWE bounds validation. (Partial: Toy params; secure presets defined but not fully validated in code.) |
| MUST | §8 | Constant-time ops; no secret-dependent branches; negl soundness error. (Partial: Some ops constant-time, but not audited globally.) |
| SHOULD | §8 | Re-entrant transcripts with labels; ZK features. (Partial: Poseidon2 transcripts, but not fully re-entrant.) |
| NICE | §1.3 | Pay-per-bit metrics; sparse support for lookups. |
---
