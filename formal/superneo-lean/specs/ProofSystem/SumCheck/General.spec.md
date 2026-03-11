# General — Sum-Check Theorem Package and Facade

## Purpose

- **What it is**: Proof-system SumCheck facade with `SoundnessErrorBoundary`, `TheoremPackage` (soundness/completeness parameters, negligible soundness error), theorem forwarding from `SingleRound`, and top-level `soundness`/`completeness` surfaces.
- **What was added**: ArkLib-style probabilistic soundness surfaces (`SoundnessFailureEvent`, `SoundnessFailureAdvantage`) plus a round-by-round union-bound boundary (`RoundByRoundSoundnessBoundary`) and closure theorem. Also added an explicit verifier-coin game model (`CoinProbModel`, `OnlineProverStrategy`, `SoundnessGame`) for non-scaffold SumCheck soundness semantics.
- **Key property**: \(\text{TheoremPackage}(s, c)\) carries \(\varepsilon\) with \(\text{IsNegligible}(\varepsilon)\); \(\text{Accepted}(\text{inst}, \text{tr}) \to \text{ClaimTrue}(\text{inst})\) (soundness); \(\text{ClaimTrue}(\text{inst}) \to \exists \text{tr}, \text{Accepted}(\text{inst}, \text{tr})\) (completeness).
- **Protocol role**: Provides the typed boundary for sum-check soundness error and the theorem package used by folding reductions (Π_CCS, Π_RLC) in Sections 7.3–7.4. The public interface is intentionally narrower than the implementation file: core game/theorem-package surfaces plus the faithful prefix-dependent endpoint are the supported contract.

## Target Formulas

- \(\text{SoundnessErrorBoundary} \equiv \{\varepsilon : \text{ErrorFn}, (\forall n, 0 \le \varepsilon(n)), \text{IsNegligible}(\varepsilon)\}\).
- \(\text{SoundnessFailureEvent}(\text{inst},\text{tr}) := \text{Accepted}(\text{inst},\text{tr}) \land \neg \text{ClaimTrue}(\text{inst})\).
- \(\text{SoundnessFailureAdvantage}(\mathsf{Pr},\text{inst},\text{tr}) := \mathsf{Pr}[\text{SoundnessFailureEvent}(\text{inst},\text{tr})]\).
- \(\text{SoundnessFailureAdvantageBound}(\text{inst},\text{tr},\varepsilon) := \forall \mathsf{Pr},n,\ \Pr[\text{SoundnessFailureEvent}(\text{inst},\text{tr})] \le \varepsilon(n)\).
- \(\text{CoinProbModel.Pr} : (\text{coins} \to \text{Prop}) \to \mathbb{Q}\), i.e. probability over verifier-coin events.
- `OnlineProverStrategy`: non-anticipatory round prover (`roundPoly i` depends only on `coins[0..i)`).
- `SoundnessGame`: fixed table, false-claim condition, and online prover strategy.
- \(\text{adv}(\mathsf{Pr}, g) := \Pr[\text{failureEvent}_g]\), where `failureEvent_g coins := acceptsOn_g coins`.
- `lundBoundHolds`: cross-multiplied Lund shape `adv * |K| <= ℓ·d`.
- Full-field root-count bridge shape:
  `count(Eᵢ) ≤ dᵢ * |F|^(ℓ-1) -> count(Eᵢ) * |K| ≤ dᵢ * |F|^ℓ` under `|K| = |F|`.
- Mathlib polynomial bridge shape:
  `poly : Array F -> sumcheckPolynomialZMod poly : Polynomial (ZMod q)`,
  then (under `Fact (Nat.Prime q)`) `rootCount(poly) ≤ natDegree(sumcheckPolynomialZMod poly)`.
- \(\text{roundFailureUnion}(E,n) := \bigvee_{i < n} E(i)\), \(\text{roundErrorSum}(\varepsilon,n) := \sum_{i < n}\varepsilon_i\).
- `RoundByRoundSoundnessBoundary` proves:
  \[
  \Pr[\text{SoundnessFailureEvent}] \le \sum_{i < \text{rounds}} \varepsilon_i
  \]
  from event coverage + per-round probability bounds.
- \(\text{TheoremPackage}(s, c) \to \text{IsNegligible}(\text{eps})\).
- \(\text{TheoremPackage}(s, c) \wedge \text{Accepted}(\text{inst}, \text{tr}) \to \text{ClaimTrue}(\text{inst})\) (soundness).
- \(\text{TheoremPackage}(s, c) \wedge \text{ClaimTrue}(\text{inst}) \to \exists \text{tr}, \text{Accepted}(\text{inst}, \text{tr})\) (completeness).
- \(\text{soundness}(h) \wedge \text{Accepted} \to \text{ClaimTrue}\); \(\text{completeness}(h) \wedge \text{ClaimTrue} \to \exists \text{tr}, \text{Accepted}\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355.
- Section 7.3 (Π_CCS), lines 481–548: sum-check soundness error \(\le \ell d / |\mathbb{K}|\).

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Soundness error boundary | `SoundnessErrorBoundary` | Definitional |
| Soundness-failure event | `SoundnessFailureEvent`, `SoundnessFailureAdvantage` | Definitional |
| Soundness-failure bound | `SoundnessFailureAdvantageBound` | Definitional |
| Round-by-round union-bound package | `RoundByRoundSoundnessBoundary`, `RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError` | Theorem-Target |
| Theorem package | `TheoremPackage` | Definitional |
| eps projection | `TheoremPackage.eps` | Definitional |
| Negligible | `TheoremPackage.negligible` | Theorem-Target |
| Soundness | `TheoremPackage.soundness`, `soundness` | Boundary |
| Completeness | `TheoremPackage.completeness`, `completeness` | Boundary |
| Extraction | `accepted_rounds_eq`, etc. | Theorem-Target (forwarded) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Boundary | `SoundnessErrorBoundary` | `epsSoundness : ErrorFn`, `negligibleEpsSoundness` | Definitional |
| Paper bound | `lundSchwartzZippelSoundnessBound` | forwards paper-style bound surface `(ℓ·d, |K|)` from core SumCheck | Definitional |
| Game model | `CoinProbModel` | Probability over verifier-coin predicates | Definitional |
| Game model | `OnlineProverStrategy` | Non-anticipatory prover strategy surface (round-local, prefix-only dependency) | Definitional |
| Game model | `SoundnessGame` | Fixed-table false-claim game with online prover strategy | Definitional |
| Game model | `SoundnessGame.transcript`, `SoundnessGame.acceptsOn`, `SoundnessGame.failureEvent`, `SoundnessGame.advantage` | Game transcript generation + acceptance/failure/advantage surfaces | Definitional |
| Game model | `SoundnessGame.advantage_nonneg`, `SoundnessGame.advantage_le_one` | Basic advantage bounds from probability axioms | Theorem-Target |
| Game model | `SoundnessGame.lundBoundHolds`, `LundSoundnessAssumption` | Paper-facing non-scaffold Lund bound surface | Boundary |
| Game model | `roundFailureUnionCoins`, `pr_roundFailureUnionCoins_le_roundErrorSum` | finite-union bound for verifier-coin events | Theorem-Target |
| Game model | `LundRoundBoundary`, `SoundnessGame.lundBoundHolds_of_roundBoundary` | round-by-round boundary implies game-level Lund bound | Theorem-Target |
| Game model | `LundRoundBoundaryAssumption`, `lundSoundnessAssumption_of_roundBoundary` | theorem-native closure bridge from per-game round boundaries to global `LundSoundnessAssumption` | Theorem-Target |
| Game model | `pr_roundFailureUnionCoins_mul_le_const` | cross-multiplied union-bound helper `Pr(⋃E_i)*d ≤ n*k` | Theorem-Target |
| Game model | `LundRoundBoundaryScaled`, `SoundnessGame.lundBoundHolds_of_scaledRoundBoundary` | per-round cross-multiplied SZ bounds imply game-level Lund bound | Theorem-Target |
| Game model | `LundRoundScaledBoundaryAssumption`, `lundSoundnessAssumption_of_scaledRoundBoundary` | theorem-native closure bridge from scaled per-round boundaries to global `LundSoundnessAssumption` | Theorem-Target |
| Game model | `LundRoundKernel`, `LundRoundBoundaryScaled.of_kernel` | lower-level round-event/root-budget kernel lifts into scaled boundary | Theorem-Target |
| Game model | `LundRoundKernelAssumption`, `lundRoundScaledBoundaryAssumption_of_kernel`, `lundSoundnessAssumption_of_kernel` | global closure path from kernel lemmas to `LundSoundnessAssumption` | Theorem-Target |
| Game model | `SchwartzZippelRoundEventLemmas`, `LundRoundKernel.of_schwartzZippelRoundEventLemmas` | lower-level SZ/round-event theorem package for one game and kernel lift | Theorem-Target |
| Game model | `SchwartzZippelRoundEventAssumption`, `lundRoundKernelAssumption_of_schwartzZippelRoundEvent`, `lundRoundScaledBoundaryAssumption_of_schwartzZippelRoundEvent`, `lundSoundnessAssumption_of_schwartzZippelRoundEvent` | constructive all-games closure from lower-level SZ/round-event lemmas to Lund soundness | Theorem-Target |
| Full-field bridge | `fullFieldCoinPr_mul_den_nat`, `fullFieldCoinPr_mul_nat_le_of_countScaled`, `fullFieldCoinEventCount_scaled_of_pr_mul_nat_le` | bidirectional count/probability transfer in canonical full-field coin model | Theorem-Target |
| Full-field bridge | `fullFieldChallengeDomain_length`, `fullFieldCoinSpace_length` | concrete finite-domain cardinalities (`|F|`, `|F^ℓ| = |F|^ℓ`) for canonical coin model | Theorem-Target |
| Full-field bridge | `Fq`, `fToZMod`, `sumcheckPolynomialZMod` | concrete bridge from `Array F` coefficients to Mathlib polynomials over `ZMod q` | Definitional |
| Full-field bridge | `rootVanishingPoly`, `rootVanishingPoly_eval_eq_zero_of_mem`, `rootVanishingPoly_natDegree_eq_card`, `rootVanishingPoly_ne_zero`, `zmodToF`, `fToZMod_zmodToF`, `zmodPolyToCoeffArray`, `sumcheckPolynomialZMod_zmodPolyToCoeffArray` | executable finite-root polynomial construction and exact array/polynomial round-trip for degree-bounded polynomials | Theorem-Target |
| Full-field bridge | `fullFieldPolyRootCount`, `fullFieldPolyRootCount_le_card_roots`, `fullFieldPolyRootCount_le_natDegree_of_nonzero`, `fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero` | root-count closure from Mathlib roots/cardinality lemmas (under primality instance + nonzero witness) | Theorem-Target |
| Full-field bridge | `FullFieldRoundPolynomialRootMathlibLemmas`, `FullFieldRoundPolynomialRootLemmas.of_mathlib`, `FullFieldRoundPolynomialRootMathlibAssumption`, `FullFieldRoundPolynomialRootMathlibAssumptionAligned`, `no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch`, `not_fullFieldRoundPolynomialRootMathlibAssumption`, `fullFieldRoundPolynomialRootAssumption_of_mathlib`, `fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRootMathlib`, `fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundPolynomialRootMathlib`, `fullFieldRoundPolynomialRootMathlibAssumption_of_aligned` | constructive conversion from Mathlib-root-count package into existing polynomial-root package, with explicit aligned/all-games conversion and mismatch blocker | Theorem-Target |
| Full-field bridge | `FullFieldRoundPolynomialRootMathlibWitness`, `FullFieldRoundPolynomialRootMathlibWitnessAssumption`, `fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness`, `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_mathlib` | constructive all-games instantiation of Mathlib-root packages from full-field SZ round-event lemmas + algebraic polynomial witness lemmas; plus direct witness-layer derivation from existing Mathlib-root packages | Theorem-Target |
| Full-field bridge | `FullFieldRoundPolynomialRootSetWitness`, `FullFieldRoundPolynomialRootSetWitnessAssumption`, `FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness`, `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness`, `fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib`, `fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib`, `fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness`, `fullFieldRoundMathlibAssumption_of_rootSetWitness` | witness-layer elimination path: finite root-set coverage + degree budget -> internally constructed polynomial witness package, with direct derivations into Mathlib-root and combined full-field round packages | Theorem-Target |
| Full-field bridge | `fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelFullFieldWitness` | convenience closure using `SchwartzZippelRoundEventAssumptionFullField` alias | Theorem-Target |
| Full-field bridge | `FullFieldRoundMathlibLemmas`, `FullFieldRoundMathlibAssumption`, `FullFieldRoundMathlibAssumptionAligned`, `no_fullFieldRoundMathlibLemmas_of_domain_mismatch`, `fullFieldDomainAligned_of_fullFieldRoundMathlib`, `fullFieldRoundMathlibAssumptionAligned_of_fullFieldRoundMathlib`, `fullFieldRoundMathlibAssumption_of_aligned`, `fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned`, `fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundMathlibAligned`, `FullFieldRoundMathlibLemmas.to_schwartzZippel`, `FullFieldRoundMathlibLemmas.to_witness`, `fullFieldRoundMathlibAssumption_of_mathlib`, `fullFieldDomainAlignedAssumption_of_fullFieldRoundMathlib`, `fullFieldRoundMathlibAssumption_of_domainAlignedAssumption` | combined theorem-native package carrying both SZ event bounds and polynomial witness/root lemmas for the same round events; with explicit domain-alignment surfaces, aligned downstream closure, and mismatch blocker. Useful as a lower package, but overstrong as a universal positive-round endpoint for arbitrary executable online provers because it requires prefix-independent round witnesses | Theorem-Target |
| Full-field bridge | `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_fullFieldRoundMathlib`, `schwartzZippelRoundEventAssumptionFullField_of_fullFieldRoundMathlib`, `fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib`, `schwartzZippelRoundEventAssumptionFullField_of_witness` | constructive closure from the combined package into witness, SZ, and Mathlib-root assumption surfaces | Theorem-Target |
| Full-field bridge | `FullFieldRoundPolynomialRootLemmas`, `FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas` | explicit polynomial-root witness layer and constructive lift into paper root-count lemmas | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventCardinalityLemmas.of_schwartzZippel` | constructive conversion from full-field SZ round-event lemmas to count-scaled cardinality lemmas | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventRootCountLemmas.of_cardinality`, `FullFieldDomainAlignedAssumption` | constructive conversion from cardinality lemmas to paper-style root-count lemmas under `|K|=|F|` alignment | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventRootCountLemmas`, `FullFieldRoundEventCardinalityLemmas.of_rootCount` | constructive conversion from paper-style root-count bounds into count-scaled cardinality lemmas | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventRootCountAssumption`, `FullFieldRoundEventRootCountAssumptionAligned`, `no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch`, `not_fullFieldRoundEventRootCountAssumption`, `fullFieldDomainAlignedAssumption_of_fullFieldRoundEventRootCount`, `fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundEventRootCount`, `fullFieldRoundEventRootCountAssumption_of_aligned`, `fullFieldRoundEventCardinalityAssumption_of_rootCount` | global closure from root-count lemma package to full-field cardinality assumption, with aligned/all-games conversion and mismatch blocker | Theorem-Target |
| Full-field bridge | `FullFieldRoundPolynomialRootAssumption`, `FullFieldRoundPolynomialRootAssumptionAligned`, `no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch`, `not_fullFieldRoundPolynomialRootAssumption`, `fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRoot`, `fullFieldRoundPolynomialRootAssumptionAligned_of_fullFieldRoundPolynomialRoot`, `fullFieldRoundPolynomialRootAssumption_of_aligned`, `fullFieldRoundEventRootCountAssumption_of_polynomialRoot` | global closure from polynomial-root lemma packages to paper root-count assumptions, with aligned/all-games conversion and mismatch blocker | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventCardinalityLemmas`, `SchwartzZippelRoundEventLemmas.of_fullFieldCardinality` | constructive lift from full-field round-event cardinality lemmas into SZ theorem surface | Theorem-Target |
| Full-field bridge | `FullFieldRoundEventCardinalityAssumption`, `SchwartzZippelRoundEventAssumptionFullField`, `fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField`, `schwartzZippelRoundEventAssumptionFullField_of_cardinality`, `schwartzZippelRoundEventAssumptionFullField_of_rootCount` | global closure between full-field cardinality and full-field SZ round-event assumptions | Theorem-Target |
| Full-field bridge | `fullFieldRoundEventRootCountAssumption_of_cardinality`, `fullFieldRoundEventRootCountAssumption_of_schwartzZippelFullField`, `fullFieldRoundEventRootCountAssumption_of_fullFieldRoundMathlib`, `fullFieldRoundEventRootCountAssumption_of_rootSetWitness` | global closure from full-field cardinality/SZ/combined/root-set assumptions to root-count assumption | Theorem-Target |
| Full-field endpoint | `LundSoundnessAssumptionFullField`, `LundSoundnessAssumptionFullFieldAligned`, `lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent`, `lundSoundnessAssumptionFullField_of_rootCount`, `lundSoundnessAssumptionFullField_of_mathlib`, `lundSoundnessAssumptionFullFieldAligned_of_mathlibAligned`, `lundSoundnessAssumptionFullField_of_mathlibAligned` | full-field Lund soundness endpoint for canonical coin model; the supported executable endpoint is the prefix-dependent aligned positive-round route documented below | Theorem-Target |
| Prefix endpoint | `SoundnessGame.prefixGapRootSet`, `SoundnessGame.prefixGapEvent`, `SoundnessGame.prefixGapSchwartzZippelLemmas`, `SoundnessGame.lundBoundHolds_of_prefixGapSchwartzZippel`, `lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix`, `lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix` | faithful prefix-dependent aligned positive-round full-field Lund endpoint derived directly from executable online-prover algebra; the positive-degree case remains exposed as the main algebraic subtheorem | Theorem-Target |
| RBR boundary | `RoundByRoundSoundnessBoundary` | Per-round failure events + per-round bounds + coverage | Boundary |
| | `pr_roundFailureUnion_le_roundErrorSum` | finite-union probability upper bound | Theorem-Target |
| | `RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError` | global failure advantage bounded by round-error sum | Theorem-Target |
| | `RoundByRoundSoundnessBoundary.soundnessFailureAdvantageBound` | bound-by-error-function closure for fixed `prob` from total-round bound | Theorem-Target |
| Package | `TheoremPackage` | Carries `soundnessError : SoundnessErrorBoundary` | Definitional |
| | `TheoremPackage.eps` | Projects soundness error function | Definitional |
| | `TheoremPackage.nonneg` | pointwise nonnegativity of soundness error function | Theorem-Target |
| | `TheoremPackage.negligible` | \(\text{IsNegligible}(\text{eps})\) | Theorem-Target |
| | `soundnessFailureAdvantageBound_of_soundness` | soundness + nonnegative error implies theorem-facing soundness-failure bound | Theorem-Target |
| | `TheoremPackage.soundness` | \(\text{Accepted} \to \text{ClaimTrue}\) | Boundary |
| | `TheoremPackage.completeness` | \(\text{ClaimTrue} \to \exists \text{tr}, \text{Accepted}\) | Boundary |
| | `TheoremPackage.soundnessFailureAdvantage_eq_zero` | exact closure of failure-event probability under soundness assumption | Theorem-Target |
| | `TheoremPackage.soundnessFailureAdvantageBound` | package-level theorem-facing soundness-failure bound against `TheoremPackage.eps` | Theorem-Target |
| Constructive package | `theoremPackage_constructive` | canonical constructor from constructive core SumCheck closures + explicit soundness-error boundary | Theorem-Target |
| Constructive package | `soundnessErrorBoundary_zero` | canonical zero-error boundary (`epsSoundness := 0`) | Theorem-Target |
| Constructive package | `theoremPackage_constructive_zeroError` | canonical constructive theorem package with zero soundness error | Theorem-Target |
| Top-level | `soundness`, `completeness` | Assumption-instantiated surfaces | Boundary |

## Proof Obligations and Closure Plan

- `TheoremPackage.negligible`: closed.
- `pr_roundFailureUnion_le_roundErrorSum`: closed.
- `RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError`: closed.
- `RoundByRoundSoundnessBoundary.soundnessFailureAdvantageBound`: closed.
- `soundnessFailureAdvantage_eq_zero_of_soundness` / `TheoremPackage.soundnessFailureAdvantage_eq_zero`: closed.
- `soundnessFailureAdvantageBound_of_soundness` / `TheoremPackage.soundnessFailureAdvantageBound`: closed.
- `lundSchwartzZippelSoundnessBound`: closed (definitional forward from core SumCheck).
- `CoinProbModel`/`OnlineProverStrategy`/`SoundnessGame`/`LundSoundnessAssumption`: closed as explicit non-scaffolded game/boundary surfaces.
- `roundFailureUnionCoins` + `pr_roundFailureUnionCoins_le_roundErrorSum`: closed.
- `LundRoundBoundary` + `SoundnessGame.lundBoundHolds_of_roundBoundary`: closed.
- `LundRoundBoundaryAssumption` + `lundSoundnessAssumption_of_roundBoundary`: closed as theorem-native reduction from per-game round boundaries.
- `pr_roundFailureUnionCoins_mul_le_const`: closed (cross-multiplied finite-union bound).
- `LundRoundBoundaryScaled` + `SoundnessGame.lundBoundHolds_of_scaledRoundBoundary`: closed.
- `LundRoundScaledBoundaryAssumption` + `lundSoundnessAssumption_of_scaledRoundBoundary`: closed as theorem-native reduction from cross-multiplied per-round boundaries.
- `LundRoundKernel` + `LundRoundBoundaryScaled.of_kernel`: closed.
- `LundRoundKernelAssumption` + `lundRoundScaledBoundaryAssumption_of_kernel` + `lundSoundnessAssumption_of_kernel`: closed.
- `SchwartzZippelRoundEventLemmas` + `LundRoundKernel.of_schwartzZippelRoundEventLemmas`: closed.
- `SchwartzZippelRoundEventAssumption` + `lundRoundKernelAssumption_of_schwartzZippelRoundEvent` + `lundRoundScaledBoundaryAssumption_of_schwartzZippelRoundEvent` + `lundSoundnessAssumption_of_schwartzZippelRoundEvent`: closed (all-games constructive instantiation from lower-level SZ/round-event lemmas).
- `fullFieldCoinPr_mul_den_nat` + `fullFieldCoinPr_mul_nat_le_of_countScaled` + `fullFieldCoinEventCount_scaled_of_pr_mul_nat_le`: closed (bidirectional cardinality/probability transfer for canonical full-field model).
- `fullFieldChallengeDomain_length` + `fullFieldCoinSpace_length`: closed (`|F|` and `|F^ℓ|` cardinality lemmas for canonical full-field model).
- `sumcheckPolynomialZMod` + `fullFieldPolyRootCount_le_natDegree_of_nonzero` + `fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero`: closed as Mathlib bridge root-count lemmas (nonzero bridged polynomial witnesses remain explicit; `Fact (Nat.Prime Goldilocks.q)` is discharged in-repo by `Goldilocks.q_prime`).
- `FullFieldRoundPolynomialRootMathlibLemmas` + `FullFieldRoundPolynomialRootLemmas.of_mathlib` + `fullFieldRoundPolynomialRootAssumption_of_mathlib`: closed as theorem-native conversion from Mathlib-root-count packages into existing polynomial-root packages.
- `FullFieldRoundPolynomialRootMathlibWitness` + `FullFieldRoundPolynomialRootMathlibWitnessAssumption` + `fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness`: closed as theorem-native all-games constructor for Mathlib-root packages from full-field SZ round-event lemmas plus algebraic polynomial witness lemmas.
- `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_mathlib`: closed as direct witness-layer derivation from `FullFieldRoundPolynomialRootMathlibAssumption` (no separate witness input required when Mathlib-root package is already available).
- `FullFieldRoundEventRootCountAssumptionAligned` + `no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch` + `not_fullFieldRoundEventRootCountAssumption` + `fullFieldDomainAlignedAssumption_of_fullFieldRoundEventRootCount` + `fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundEventRootCount` + `fullFieldRoundEventRootCountAssumption_of_aligned`: closed (explicit aligned/all-games conversion plus mismatch impossibility at the root-count layer).
- `FullFieldRoundPolynomialRootAssumptionAligned` + `no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch` + `not_fullFieldRoundPolynomialRootAssumption` + `fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRoot` + `fullFieldRoundPolynomialRootAssumptionAligned_of_fullFieldRoundPolynomialRoot` + `fullFieldRoundPolynomialRootAssumption_of_aligned`: closed (explicit aligned/all-games conversion plus mismatch impossibility at the polynomial-root layer).
- `FullFieldRoundPolynomialRootMathlibAssumptionAligned` + `no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch` + `not_fullFieldRoundPolynomialRootMathlibAssumption` + `fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRootMathlib` + `fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundPolynomialRootMathlib` + `fullFieldRoundPolynomialRootMathlibAssumption_of_aligned`: closed (explicit aligned/all-games conversion plus mismatch impossibility at the Mathlib-root layer).
- `rootVanishingPoly` + `zmodPolyToCoeffArray` + `sumcheckPolynomialZMod_zmodPolyToCoeffArray`: closed as executable polynomial-construction bridge from finite root sets to array witnesses.
- `FullFieldRoundPolynomialRootSetWitness` + `FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness` + `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness`: closed as constructive witness-layer elimination path from root-set coverage/budget lemmas.
- `fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib` + `fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib`: closed as direct derivation of the root-set witness assumption from stronger existing lower-layer packages.
- `fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness` + `fullFieldRoundMathlibAssumption_of_rootSetWitness`: closed as direct constructive closure from root-set witnesses into Mathlib-root and combined full-field round packages.
- `fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelFullFieldWitness`: closed (alias-level constructor from `SchwartzZippelRoundEventAssumptionFullField` + witness assumption).
- `FullFieldRoundMathlibLemmas` + `FullFieldRoundMathlibAssumption` + projections (`to_schwartzZippel`, `to_witness`) + `fullFieldRoundMathlibAssumption_of_mathlib`: closed (single combined package for event and polynomial layers, with direct construction from Mathlib-root assumptions).
- `FullFieldRoundMathlibAssumptionAligned` + `no_fullFieldRoundMathlibLemmas_of_domain_mismatch` + `fullFieldDomainAligned_of_fullFieldRoundMathlib` + `fullFieldRoundMathlibAssumptionAligned_of_fullFieldRoundMathlib` + `fullFieldRoundMathlibAssumption_of_aligned` + `fullFieldDomainAlignedAssumption_of_fullFieldRoundMathlib` + `fullFieldRoundMathlibAssumption_of_domainAlignedAssumption`: closed (domain-alignment prerequisites are explicit, with mismatch impossibility theorem and conversion between aligned/all-games surfaces).
- `SoundnessGame.prefixGapRootSet` + `SoundnessGame.prefixGapEvent` + `SoundnessGame.prefixGapSchwartzZippelLemmas` + `SoundnessGame.lundBoundHolds_of_prefixGapSchwartzZippel` + `lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix`: closed. This is the main algebraic endpoint for aligned positive-round executable online-prover games.
- `lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix`: closed. The aligned positive-round endpoint now covers both the positive-degree branch and the degree-zero branch; when `maxDegree = 0`, `sumcheckDegreeCompatible` rules out accepting failure events, so the Lund bound holds trivially.
- `fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned` + `fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundMathlibAligned`: closed constructive aligned-chain projection from the combined package into Mathlib-root and root-count aligned surfaces.
- `fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_fullFieldRoundMathlib` + `schwartzZippelRoundEventAssumptionFullField_of_fullFieldRoundMathlib` + `fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib` + `schwartzZippelRoundEventAssumptionFullField_of_witness`: closed constructive closure chain from the combined package.
- `fullFieldRoundEventRootCountAssumption_of_mathlib`: closed (direct chain from Mathlib-root packages to paper root-count assumptions).
- `fullFieldPolyRootCount` + `FullFieldRoundPolynomialRootLemmas` + `FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas`: closed (constructive theorem path from explicit polynomial-root witnesses into paper root-count lemmas).
- `FullFieldRoundEventCardinalityLemmas.of_schwartzZippel`: closed (full-field SZ round-event lemmas -> count-scaled cardinality lemmas).
- `FullFieldRoundEventRootCountLemmas.of_cardinality` + `FullFieldDomainAlignedAssumption`: closed (count-scaled cardinality -> paper root-count under `|K|=|F|` alignment).
- `FullFieldRoundEventRootCountLemmas` + `FullFieldRoundEventCardinalityLemmas.of_rootCount`: closed (root-count form to cardinality-scaled form).
- `FullFieldRoundEventRootCountAssumption` + `fullFieldRoundEventCardinalityAssumption_of_rootCount`: closed (global root-count to cardinality closure).
- `fullFieldRoundEventCardinalityAssumption_of_mathlib`: closed (direct chain from Mathlib-root packages to full-field cardinality closure).
- `FullFieldRoundPolynomialRootAssumption` + `fullFieldRoundEventRootCountAssumption_of_polynomialRoot`: closed (global polynomial-root package -> global paper root-count package).
- `fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField`: closed (global full-field SZ -> global cardinality closure).
- `fullFieldRoundEventRootCountAssumption_of_cardinality` + `fullFieldRoundEventRootCountAssumption_of_schwartzZippelFullField`: closed (global cardinality/SZ -> global root-count closure, with domain alignment).
- `fullFieldRoundEventRootCountAssumption_of_fullFieldRoundMathlib` + `fullFieldRoundEventRootCountAssumption_of_rootSetWitness`: closed (direct global root-count closure from combined package and root-set witness package).
- `SchwartzZippelRoundEventAssumptionFullField_of_rootCount`: closed (root-count to full-field SZ closure).
- `schwartzZippelRoundEventAssumptionFullField_of_mathlib`: closed (direct chain from Mathlib-root packages to full-field SZ closure).
- `FullFieldRoundEventCardinalityLemmas` + `SchwartzZippelRoundEventLemmas.of_fullFieldCardinality`: closed.
- `FullFieldRoundEventCardinalityAssumption` + `SchwartzZippelRoundEventAssumptionFullField` + `schwartzZippelRoundEventAssumptionFullField_of_cardinality`: closed.
- `LundSoundnessAssumptionFullField` + `LundSoundnessAssumptionFullFieldAligned` + `lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent` + `lundSoundnessAssumptionFullField_of_rootCount` + `lundSoundnessAssumptionFullField_of_mathlib`: closed.
- `lundSoundnessAssumptionFullFieldAligned_of_mathlibAligned` + `lundSoundnessAssumptionFullField_of_mathlibAligned`: closed (aligned-game endpoint and aligned-to-global bridge for Mathlib-root closure).
- Constructive package constructors (`theoremPackage_constructive`, `soundnessErrorBoundary_zero`, `theoremPackage_constructive_zeroError`): closed.
- `TheoremPackage.soundness`, `TheoremPackage.completeness`, `soundness`, `completeness`: assumption-instantiated surfaces remain intentionally typed boundaries in this wrapper module.
- Core closure status: `SuperNeo.SumCheck` now provides constructive closures
  (`sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`) and a canonical bundle `sumcheckAssumptions_constructive`.

## Assumption Ledger

- `SoundnessAssumption` [Boundary-surface]: provided as a typed input here; constructively closed upstream in `SuperNeo.SumCheck`.
- `CompletenessAssumption` [Boundary-surface]: provided as a typed input here; constructively closed upstream in `SuperNeo.SumCheck`.
- `LundSoundnessAssumption` [Boundary-surface]: explicit probabilistic game-level soundness bound over verifier-coin events with non-anticipatory prover strategy.
- `LundRoundBoundaryAssumption` [Boundary-surface]: if per-game round-by-round failure boundaries are provided, `LundSoundnessAssumption` is constructively discharged by `lundSoundnessAssumption_of_roundBoundary`.
- `SchwartzZippelRoundEventAssumption` [Boundary-surface, theorem-native lower layer]: packages the per-game SZ/round-event lemmas required to derive `LundSoundnessAssumption` via a fully constructive chain.
- `Fact (Nat.Prime Goldilocks.q)` [Closed prerequisite]: discharged in-repo by `SuperNeo.Goldilocks.q_prime` (`SuperNeo/GoldilocksPrime.lean`) and used by the Mathlib root-count bridge over `ZMod q`.
- `FullFieldRoundPolynomialRootMathlibAssumption` [Boundary-surface, theorem-native lower layer]: packages per-game bridged-polynomial nonzero/shape/count witnesses; constructively discharges `FullFieldRoundPolynomialRootAssumption`.
- `FullFieldRoundPolynomialRootMathlibAssumptionAligned` [Boundary-surface, theorem-native lower layer]: aligned-game variant (`|K| = |F|`) of the Mathlib-root package.
- `FullFieldRoundPolynomialRootMathlibWitnessAssumption` [Boundary-surface, theorem-native lower layer]: existential all-games witness surface (`∃ hSz, witness`) used to instantiate `FullFieldRoundPolynomialRootMathlibAssumption` without requiring witness lemmas for arbitrary externally supplied `hSz`.
- `FullFieldRoundPolynomialRootSetWitnessAssumption` [Boundary-surface, lower than witness layer]: finite per-round root-set coverage + degree budgets; constructively discharges `FullFieldRoundPolynomialRootMathlibWitnessAssumption`.
- `FullFieldRoundMathlibAssumption` [Boundary-surface, theorem-native lower layer]: combined per-game package containing both SZ event bounds and polynomial witness/root lemmas for matching round events; constructively discharges `FullFieldRoundPolynomialRootMathlibWitnessAssumption` and `SchwartzZippelRoundEventAssumptionFullField`.
- `FullFieldRoundMathlibAssumptionAligned` [Boundary-surface, theorem-native lower layer]: aligned-game variant (`|K| = |F|`) of the combined package. It remains a valid lower package, but not the faithful universal positive-round endpoint for arbitrary executable online provers because it requires prefix-independent round witnesses.
- `FullFieldRoundPolynomialRootAssumption` [Boundary-surface, theorem-native lower layer]: packages per-game polynomial-root witnesses and root-count bounds; constructively discharges `FullFieldRoundEventRootCountAssumption`.
- `FullFieldRoundPolynomialRootAssumptionAligned` [Boundary-surface, theorem-native lower layer]: aligned-game variant (`|K| = |F|`) of the polynomial-root package.
- `FullFieldRoundEventRootCountAssumption` [Boundary-surface, theorem-native lower layer]: packages full-field paper-style root-count lemmas (`count ≤ d·|F|^(ℓ-1)`) and discharges `FullFieldRoundEventCardinalityAssumption` constructively.
- `FullFieldRoundEventRootCountAssumptionAligned` [Boundary-surface, theorem-native lower layer]: aligned-game variant (`|K| = |F|`) of the root-count package.
- `FullFieldRoundEventCardinalityAssumption` [Boundary-surface, theorem-native lower layer]: packages canonical full-field round-event cardinality bounds that discharge full-field SZ/Lund closures constructively.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProofSystem.SumCheck.SingleRound`, `SuperNeo.ProofSystem.Types`, `SuperNeo.ProofSystem.Security`, `SuperNeo.GoldilocksPrime`.
- **Consumers**:
  - `SuperNeo.ProofSystem.SumCheck`: imports General for barrel.
  - `SuperNeo.ProofSystem.Folding.PiCCS`, `PiRLC`: depend on sum-check acceptance/claim for Π_CCS and Π_RLC.

## Implementation Plan

Keep theorem package as typed boundary façade; import constructive assumptions from core SumCheck when a closed package is desired.

## Quality Expectations

Theorem package exposes explicit soundness-error surface; assumptions are minimal and documented.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- Assumption ledger documents boundary assumptions.

## Out of Scope

- Non-full-field or non-aligned variants of the executable prefix-dependent Lund endpoint.
- Further file-level cleanup/removal of older packaged lower routes that are no longer on the critical path.
