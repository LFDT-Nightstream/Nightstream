# Foundations: remaining security links

This review uses the checked working tree on 2026-09-08. All 11 proof records closed before link work began. All seven links now have checked reduction/review evidence or the explicit owner-approved cryptographic assumption. The complete boundary, library and axiom gates passed.

## Closed links

| Record | Evidence | Scope |
|---|---|---|
| `F.setup.rust_vectors` | Current Lean emission and Rust setup/streaming checks passed. | RFC block, selected coefficients, seeds, complete descriptor, and four streaming cases. |
| `F.multilinear.root_bound` | `GoldilocksRoots.agreement_count_le`, `uniform_fullField_agreement_probability_le`, and `badChallenge_count_le`; axiom audit passed. | Actual `K`, actual fixed-width coefficient lists, and the existing bad-challenge event. Uniform sampling after the polynomial is fixed remains an explicit premise. |
| `F.setup.reduction_bias` | `ReductionBias.frequency_error_le`, `wide_frequency_error_le`, and selected `idealReductionErrorBudget_lt`; axiom audit passed. | Modular reduction under independent uniform 256-bit inputs. It does not assert that public-seed ChaCha20 supplies those inputs. |
| `F.profile.msis_assessment` | `security_analysis.py` and `FOUNDATION_SECURITY_PARAMETERS.json`. | Current parameters evaluated under named heuristic lattice-cost models. No hardness proof or security-level approval. |
| `F.commitment.binding` | `Binding.bindingCollision_to_shortKernel` and selected `productionBindingCollision_to_shortKernel`. | Same-key nonzero integer witness below `2B`, widened to the selected `8TB` instance. |
| `F.commitment.relaxed_binding` | `Binding.relaxedBindingCollision_to_shortKernel` and selected `productionRelaxedBindingCollision_to_shortKernel`. | Actual `C-C` challenges, strict `2B` openings, and a same-key integer witness below `8TB`. |

The root-count module connects the existing `FixedPhase.BadChallenge` event to the count bound. It retains the independently derived expected-polynomial representation. The extension-field sampling space has exactly `q²` elements. The proof does not establish Fiat–Shamir challenge uniformity or a complete protocol probability bound.

## Exact modular-reduction calculation

Let `N=2^256`, `q=18446744069414584321`, `a=N div q`, and `r=N mod q=4294967295`. A uniform integer from `[0,N)` contains `a` complete residue blocks and one tail of length `r`.

For any event on residues, its count is `a*c+t`, where `c` is its count in a complete block and `t` its count in the tail. The proved facts `0≤c≤q`, `0≤t≤r`, and `N=a*q+r` give an event-probability difference at most `r/N`. This is the event-frequency theorem in Lean, with no enumeration of the sample space.

The exact single-coordinate total variation distance is `r*(q-r)/(q*N)`: each of the first `r` residues has `a+1` preimages, and every other residue has `a` preimages. The simpler Lean bound `r/N` is sufficient here.

The selected key has `22*4708530*54=5593733640` scalar coefficients. For independent ideal inputs, replace one coordinate at a time. With all other coordinates fixed, every event becomes an event on that coordinate. The single-coordinate bound applies; averaging preserves it, and the triangle inequality sums the errors. The complete-key error is therefore at most `5593733640*r/N`, approximately `2^-191.6188`, and strictly below `2^-191`. Lean checks the scalar event bound, exact coefficient count, remainder, and numerical sum. The product-distribution argument in this paragraph is a mathematical argument, not a Lean theorem for product measures.

This error describes ideal modular reduction. It is not an estimate of the distance between the actual public-seed key and a uniform matrix.

## Current MSIS parameters and model results

The code reads the current Lean setup descriptor and checks its dimensions against the selected setup authority. It reads `q`, `d`, `b`, `k_rho`, and `T` from the working-tree Lean definitions.

| Quantity | Value |
|---|---:|
| `q` | 18446744069414584321 |
| Ring degree `d` | 54 |
| Row count `kappa` | 22 |
| Ring columns | 4708530 |
| Scalar rows `n=kappa*d` | 1188 |
| Scalar columns `m` | 254260620 |
| `b`, `k_rho`, `B` | 2, 16, 65536 |
| Expansion bound `T` | 216 |
| Relaxed-binding MSIS norm `8TB` | 113246208 |
| Euclidean relaxation `8TB*sqrt(m)` | Approximately 1.80577e12 |

The norm conversion follows SuperNeo Section 8 and Appendix B.6. The calculation uses the Euclidean branch of [lattice-estimator's SIS model](https://github.com/malb/lattice-estimator/blob/53da5982597709ba0fdf94ea37a84d822310fd84/estimator/sis_lattice.py) and the named [reduction-cost formulas](https://github.com/malb/lattice-estimator/blob/53da5982597709ba0fdf94ea37a84d822310fd84/estimator/reduction.py). The model selects lattice dimension 3734 and BKZ block size 431. The adjacent Chen values are recorded to show the numerical bracket.

| Model and cost unit | Current profile: log2 cost | Paper B.6 reference: log2 cost |
|---|---:|---:|
| MATZOV classical gates | 153.20 | 129.08 |
| MATZOV quantum depth × width | 146.70 | 124.98 |
| ADPS16 classical Core-SVP | 125.85 | 100.74 |
| ADPS16 quantum Core-SVP | 114.22 | 91.43 |

These are different cost models and units. They are not interchangeable security guarantees. In particular, this review does not certify 128-bit quantum Core-SVP security. The paper reference uses `k_rho=14`, `kappa=18`, and `19884107` ring columns only for comparison; it is not a Nightstream production profile.

The result comes from a CPython evaluation of the pinned formulas. Sage is not installed locally, and the public Sage computation endpoint returned errors; no full Sage estimator run is claimed. The script checks the upstream Chen inversion examples and reproduces the paper's approximate 129-bit classical result. Generic lattice attacks and the Euclidean relaxation are heuristic analysis. This calculation does not prove hardness for structured Module-SIS or for the particular public-seed matrix.

## Checked binding reductions

Ordinary binding subtracts the two openings. Additivity proves that this difference commits to zero under the same key. Distinctness makes the residue vector nonzero. The centered integer lift retains that nonzero value, the modular kernel equation and a strict `2B` norm bound. The `ZMod` modulus is fixed explicitly, so no intermediate natural-number conversion changes the lift.

Relaxed binding forms `Delta1*z2 - Delta2*z1`. Each delta is an actual difference of two production challenges. The existing quotient-ring support proof now covers any stated coordinate bound; its original fresh-coefficient theorems remain as special cases. With both openings below `2B`, one difference-challenge action is at most `2T*(2B-1)`. The final difference is at most `4T*(2B-1)`, which is strictly below `8TB`. Commitment homomorphism and commutativity give the kernel equation. The collision event already supplies the required cross-product inequality.

Both reductions are instantiated in `Export/Stage1/SetupBinding.lean` at `productionAjtaiKey`, including the current public seed, all 4708530 ring columns, 22 rows and the bound 113246208. No matrix is resampled or materialized. The ordinary reduction provides the stronger `2B` bound and widens it to the same selected MSIS instance.

These are reductions of successful attacks, not assertions that no short kernel vector exists. They do not assume global injectivity of a finite commitment map. For any attack distribution, each successful collision yields a successful MSIS witness; any computational hardness bound for that exact key therefore bounds collision success after accounting for the reduction's work.

## Approved setup assumption

**Public-seed setup.** The standard secret-seed PRG argument does not apply to this published seed. An observer can compute a coefficient from the seed and compare it with the supplied matrix. The real matrix always matches; an independent uniform field coefficient matches with probability `1/q`. Thus the distributions, when the seed is supplied, have an efficient distinguisher with advantage `1-1/q`. This does not show that the matrix has a short kernel vector or that the commitment is broken. It does show that ordinary ChaCha20 pseudorandomness cannot justify this matrix replacement.

Keeping the current setup needs an explicit MSIS hardness assumption for the distribution produced by this public-seed construction. The selected seed is recorded in `Poseidon2HashChainV1SetupAuthority.lean` as an owner-approved operating-system CSPRNG output. The compiled verifier freezes those bytes. Its approved premise is hardness for that specific matrix; the generation history does not establish a per-seed guarantee from a sampled-setup average. This differs from the paper's uniform-matrix premise and must not be described as a proof of secret-seed PRG security. The owner approved the exact current fixed-matrix assumption on 2026-09-08. The governing record is `PUBLIC_SEED_MSIS_ASSUMPTION.md`; it separates the current frozen seed from any future sampled-seed mode.

The earlier binding attempt stopped under the three-round rule. The continued attempt fixed the modulus in the integer lift, and both binding reductions now pass their focused checks and the full axiom gate. The full user goal remains active. Runtime algorithms, profile values, and the F′ architecture remain unchanged.

Final checkpoint: the complete boundary gate passed. The library build completed 3655 jobs in 56 seconds; the axiom/test build completed 3691 jobs in 19 seconds. Log: `/tmp/nightstream-foundation-links-checkpoint.log`. Focused root and bias proof/audit logs are `/tmp/nightstream-goldilocks-roots-audit.log` and `/tmp/nightstream-reduction-bias-audit.log`. The setup-distribution dependent rebuild took 295 seconds; its final audit took 2 seconds. No tactic limit was increased.


## Owner approval and governing scope

The owner approved the local assumption on 2026-09-08 with six conditions. The final, controlling statement is `PUBLIC_SEED_MSIS_ASSUMPTION.md`. It identifies the compiled setup as a point mass at the frozen seed and thus assumes hardness for that specific matrix. It does not infer a per-seed guarantee from a sampled-seed average or from the paper's uniform-matrix assumption.

All matrix-generation parameters and the selected security profile are tied to verifier-owned package/context authority. The new `packageIdentity_identifies_selected_setup_or_collision` theorem connects the exact setup to the expected identity or a named Poseidon2 collision. The final Stage 1 claim must say “secure assuming public-seed MSIS hardness for the selected setup” and retain its other outstanding premises. The approval introduces no Lean axiom and no numerical security guarantee.

Binding checkpoint: the complete boundary gate passed. The library build completed 3658 jobs in 306 seconds; the axiom/test build completed 3694 jobs in 18 seconds. Log: `/tmp/nightstream-binding-final.log`. The ordinary reduction checked in 2 seconds; the relaxed reduction and general norm bound each checked in 4 seconds. The selected-key wrapper checked in 3 seconds. No runtime algorithm, profile, tactic budget or root commit changed.
