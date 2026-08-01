# SumCheck

## Purpose

- **What it is**: Interactive sum-check protocol implementation providing prover and verifier for single and batched sum-check instances, plus polynomial evaluation and interpolation utilities.
- **Key invariant**: For each round `i`, the prover's univariate polynomial `p_i` satisfies `p_i(0) + p_i(1) = running_sum`, and `deg(p_i) <= degree_bound`. After all rounds, `running_sum = oracle_eval(r)` where `r` is the vector of transcript-derived challenges.
- **Protocol role**: The sum-check is the core interactive proof primitive used inside Pi_CCS (FE and NC sumchecks) and batched sumcheck (Route A shared-challenge alignment). Every folding step bottoms out in a sum-check invocation.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `T = sum_{x in {0,1}^ell} g(x)` | Def 6, line 352 | `initial_sum` parameter | Claimed sum over Boolean hypercube |
| `p_i(X_i)` univariate | Def 6, line 353 | `rounds[i]` (coefficient vector) | Low-degree polynomial sent each round |
| `p_i(0) + p_i(1) = T_i` | Def 6, line 354 | Invariant check in prover/verifier | Running sum consistency |
| `r_i <- K` challenge | Def 6, line 355 | `challenges[i]` | Transcript-derived Fiat-Shamir challenge |
| `deg(p_i) <= d` | Def 6 | `degree_bound` | Degree bound per round |
| Lagrange interpolation | Standard | `interpolate_from_evals(xs, ys)` | Reconstructs polynomial from evaluations |
| Horner evaluation | Standard | `poly_eval_k(coeffs, x)` | Evaluates polynomial at extension-field point |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 6 (Sum-check protocol), Section 4, lines 352-355: prover sends `p_i`, verifier checks `p_i(0)+p_i(1)=T_{i-1}`, samples `r_i`.
- Section 7.3 (Pi_CCS), lines 481-548: sum-check invoked for the Q polynomial over row/Ajtai dimensions.
- Section 7.4 (Pi_RLC), lines 549-583: sum-check invoked for RLC reduction.
- Soundness error: `<= ell * d / |K|` (Schwartz-Zippel union bound over ell rounds, degree d).

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `SumCheck.spec.md` | `SuperNeo/SumCheck.lean` | Defines `SumCheckInstance`, `SumCheckTranscript`, `sumcheckAcceptedCore`, `sumcheckVerifierAccepted` |
| `ProofSystem/SumCheck/General.spec.md` | `SuperNeo/ProofSystem/SumCheck/General.lean` | `TheoremPackage` with `soundness`/`completeness`, Lund-Schwartz-Zippel bound |
| `ProofSystem/SumCheck/SingleRound.spec.md` | `SuperNeo/ProofSystem/SumCheck/SingleRound.lean` | Extraction and rejection theorems for acceptance predicates |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Trait | `RoundOracle` | trait | Core | Provides `evals_at`, `num_rounds`, `degree_bound`, `fold` for sum-check oracles |
| Prover | `run_sumcheck_prover(tr, oracle, initial_sum)` | fn | Core | Returns `(rounds, challenges)` with invariant `p_i(0)+p_i(1)=T_i` enforced |
| Verifier | `verify_sumcheck_rounds(tr, degree_bound, initial_sum, rounds)` | fn | Core | Returns `(challenges, final_sum, is_valid)` |
| Batched Prover | `run_batched_sumcheck_prover(tr, claims)` | fn | Core | Shared-challenge batched sum-check; returns `(shared_challenges, per_claim_results)` |
| Batched Verifier | `verify_batched_sumcheck_rounds(tr, per_claim_rounds, ...)` | fn | Core | Verifies batched sum-check with shared challenges |
| Batched Types | `BatchedClaim` | struct | Core | Oracle + claimed sum + label for one claim in batched sum-check |
| Batched Types | `BatchedClaimResult` | struct | Core | Per-claim result: round polynomials + final value |
| Evaluation | `poly_eval_k(coeffs, x)` | fn | Core | Horner's method polynomial evaluation over K |
| Evaluation | `poly_eval_k_base(coeffs, x)` | fn | Core | Optimized evaluation at base-field point `x in F_q` |
| Interpolation | `interpolate_from_evals(xs, ys)` | fn | Core | Lagrange interpolation returning coefficient vector |
| Errors | `SumcheckError::Invariant` | enum variant | Core | Round invariant `p(0)+p(1) != T` violation |
| Errors | `SumcheckError::BatchedInvariant` | enum variant | Core | Per-claim invariant violation in batched mode |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Round consistency: `p_i(0) + p_i(1) = running_sum` for each round | Unit test (prover enforces, verifier checks) | `sumcheckRoundConsistent` |
| Degree bound: `coeffs.len() <= degree_bound + 1` | Unit test | `sumcheckRoundPolyShape` |
| Prover-verifier agreement: same transcript produces same challenges | Unit test (roundtrip) | `sumcheckAccepted_challenges_eq` |
| `poly_eval_k` correctness: Horner agrees with naive evaluation | Unit test | (none) |
| `poly_eval_k_base` consistency: agrees with `poly_eval_k` for base-field inputs | Unit test | (none) |
| `interpolate_from_evals` round-trip: `poly_eval_k(interp(xs,ys), xs[i]) == ys[i]` | Unit test | (none) |
| Batched sum-check: shared challenges identical across all claims | Unit test | (none) |
| Batched sum-check: per-claim invariants hold independently | Unit test | (none) |
| Verifier rejects tampered round polynomial | Unit test | `sumcheckAccepted` rejection |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Transcript (Fiat-Shamir) produces unpredictable challenges | neo-transcript / Poseidon2 | Standard random-oracle assumption |
| Extension field `K = F_{q^2}` has sufficient size for soundness | Paper Appendix B.2 | `|K| = q^2 > 2^{128}`, soundness error `ell*d/|K|` is negligible |
| `InterpolationPlan` cache is thread-safe | `OnceLock<RwLock<BTreeMap>>` | Standard Rust synchronization |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `Fq`, `K`, `from_complex`, `KExtensions::scale_base`
- `neo-transcript`: `Transcript` trait for Fiat-Shamir challenge derivation

Downstream consumers:
- `neo-reductions::engines::optimized_engine`: `OptimizedOracle` and `NcOracle` implement `RoundOracle`
- `neo-reductions::engines::paper_exact_engine`: Paper-exact oracle implements `RoundOracle`
- `neo-fold`: batched sum-check for Route A shared-challenge alignment

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none yet) | (none) | Sum-check protocol is verified structurally; no golden-vector oracle tests currently exist |

## Quality Expectations

- No `todo!()` or `unimplemented!()` in the contract surface
- `SumcheckError` provides round index and expected/actual values for debugging
- `poly_eval_k_base` uses `KExtensions::scale_base` for performance (avoids full extension-field mul)
- `InterpolationPlan` caching amortizes Lagrange basis computation across rounds

## Acceptance Criteria

- `cargo test -p neo-reductions --release` succeeds
- Spec-derived tests in `spec-tests/sumcheck.rs` pass
- Prover-verifier roundtrip succeeds for degree bounds 1-5
- Tampered round polynomials are rejected by verifier
- `cargo clippy -p neo-reductions --all-targets --release -- -D warnings` clean

## Out of Scope

- Specific oracle implementations (OptimizedOracle, NcOracle) -- see Engines.spec.md
- Protocol-level Q polynomial definition -- see PiCCS.spec.md
- Transcript implementation details -- see neo-transcript crate
