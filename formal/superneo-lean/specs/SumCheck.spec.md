# SumCheck

## Purpose

- **What it is**: The interactive sum-check protocol formalization, defining `SumCheckInstance` (claimed value, rounds, max degree), `SumCheckTranscript` (challenges, round polynomials), a stronger standalone scaffold predicate `sumcheckAcceptedCore`, and a paper-facing verifier predicate `sumcheckVerifierAccepted`.
- **Core witness surface**: `SumCheckDefinition6Statement` packages one Definition-6 theorem instance: a concrete statement together with an honest-transcript constructor for every verifier challenge vector of the right length, plus accepted/final-oracle proofs for those transcripts.
- **Scope note**: The standalone constructive realization is table/MLE-based. A more abstract reusable `SumCheck(T; Q)` library can be layered on top, but the module target here is the exact Definition-6 theorem surface and its constructive closure.
- **Key property**: `sumcheckAccepted inst tr` implies structural properties: `tr.roundPolys.size = inst.rounds`, `tr.challenges.size = tr.roundPolys.size`, and each round's `p(0) + p(1) = eval(p_{prev}, r_{prev})`.
- **Protocol role**: Sum-check is the interactive reduction backbone of SuperNeo. Section 7.3 (Π_CCS) and Section 7.4 (Π_RLC) both invoke sum-check to reduce multivariate polynomial claims to point-evaluation queries, which are then handled by MLE evaluation.

## Target Formulas

- `p_0(0) + p_0(1) = v` (initial round sum = claimed value)
- `p_{i+1}(0) + p_{i+1}(1) = p_i(r_i)` (round transition)
- `deg(p_i) ≤ maxDegree` (paper-facing degree bound)
- `|p_i| = maxDegree + 1` (normalized executable encoding)
- `|challenges| = |roundPolys| = rounds` (transcript shape)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 6 (Sum-check protocol), Section 4, lines 352-355.
- Section 7.3 (Π_CCS), lines 440-470: sum-check invocation in CCS folding.
- Section 7.4 (Π_RLC), lines 471-489: sum-check invocation in RLC.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/SumCheck.lean` | Definition 6 (sum-check) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Structures | `SumCheckInstance` | structure | Definitional | `rounds`, `maxDegree`, `domainSize`, `claimedValue` |
| Structures | `SumCheckTranscript` | structure | Definitional | `challenges`, `roundPolys` |
| Structures | `SumCheckClaim` | structure | Definitional | Verifier-claim object packaging transcript witness + verifier predicates |
| Structures | `SumCheckStatement` | structure | Definitional | Paper-facing statement object: parameter/degree constraints, hypercube table, size `2^rounds`, and sum-equals-claim |
| Structures | `SumCheckDefinition6Statement` | structure | Definitional | Definition-6 theorem object: a statement plus an honest transcript family and its accepted/final-oracle proofs for every valid verifier challenge vector |
| Evaluation | `sumcheckEvalPoly` | def | Definitional | Horner-form univariate eval |
| Evaluation | `sumcheckTableSum` | def | Definitional | Hypercube-table sum surface |
| Predicates | `sumcheckRoundConsistent` | def | Definitional | Transcript shape check |
| Predicates | `sumcheckRoundPolyShape` | def | Definitional | Polynomial coefficient count |
| Predicates | `sumcheckRoundPolyDegreeLe` | def | Definitional | Paper-facing coefficient-array degree bound |
| Predicates | `sumcheckRoundDegrees` | def | Definitional | Per-round paper-facing degree bound |
| Predicates | `sumcheckRoundShapes` | def | Definitional | All round polys shaped |
| Predicates | `sumcheckFoldConsistent` | def | Definitional | Round-to-round transition |
| Predicates | `sumcheckInitialRoundConsistent` | def | Definitional | `p_0(0) + p_0(1) = v` |
| Predicates | `sumcheckFinalClaimConsistent` | def | Definitional | Legacy scaffold helper (not part of acceptance endpoint) |
| Predicates | `sumcheckFinalOracleConsistentWithTable` | def | Definitional | Shape-safe final-oracle consistency indexed by a raw table (no sum-equals-claim field required) |
| Acceptance | `sumcheckAcceptedCore` | def | Definitional | Standalone scaffold acceptance (parameter, degree-compatibility, round shape/transition, initial, fold) |
| Acceptance | `sumcheckVerifierAccepted` | def | Definitional | Paper-facing verifier predicate (round consistency, degree bound, initial, fold) |
| Acceptance | `sumcheckAccepted` | def | Definitional | Executable standalone scaffold acceptance; equal to `sumcheckAcceptedCore` |
| Acceptance | `sumcheckAcceptedForTable` | def | Definitional | `sumcheckAcceptedCore ∧ sumcheckFinalOracleConsistentWithTable inst table tr` |
| Claim | `sumcheckParameterConsistent` | def | Definitional | `maxDegree ≤ domainSize` |
| Claim | `sumcheckDegreeCompatible` | def | Definitional | Internal completeness restriction of the table-based standalone model formalized here: `rounds = 0 ∨ 0 < maxDegree` |
| Claim | `sumcheckBaseClaimTrue` | def | Definitional | Legacy base claim: `parameterConsistent ∧ degreeCompatible` |
| Claim | `sumcheckPaperClaimTrue` | def | Definitional | Definition-6 theorem-witness surface: `Nonempty (SumCheckDefinition6Statement inst)` |
| Claim | `sumcheckClaimTrue` | def | Definitional | Alias to the Definition-6 theorem-witness surface |
| Soundness bound | `sumcheckLundSoundnessNumerator` | def | Definitional | Paper-style numerator `ℓ·d` with `ℓ = rounds`, `d = maxDegree` |
| Soundness bound | `sumcheckLundSoundnessDenominator` | def | Definitional | Paper-style denominator `|K|` modeled as `domainSize` |
| Soundness bound | `sumcheckLundSoundnessBound` | def | Definitional | Pair `(ℓ·d, |K|)` corresponding to paper bound `ℓ·d/|K|` |
| Extraction | `sumcheckAccepted_rounds_eq` | theorem | Theorem-Target | `|roundPolys| = rounds` |
| Extraction | `sumcheckAccepted_challenges_eq` | theorem | Theorem-Target | `|challenges| = |roundPolys|` |
| Extraction | `sumcheckAccepted_fold_step` | theorem | Theorem-Target | Fold-step identity |
| Extraction | `sumcheckAccepted_initial_round` | theorem | Theorem-Target | Initial round check |
| Extraction | `sumcheckAccepted_round_sum_step` | theorem | Theorem-Target | Round-sum transition |
| Rejection | `sumcheckAccepted_not_of_challenge_size_ne` | theorem | Theorem-Target | Reject bad challenge count |
| Rejection | `sumcheckAccepted_not_of_roundpoly_size_ne` | theorem | Theorem-Target | Reject bad poly count |
| Rejection | `sumcheckAccepted_not_of_bad_round_shape` | theorem | Theorem-Target | Reject bad poly shape |
| Rejection | `sumcheckAccepted_not_of_bad_initial_round` | theorem | Theorem-Target | Reject bad initial round |
| Closure | `sumcheckFinalOracleConsistent_iff_withTable` | theorem | Theorem-Target | Bridge between statement-indexed and table-indexed endpoint relations |
| Closure | `sumcheckVerifierAccepted_of_accepted` | theorem | Theorem-Target | Scaffold acceptance implies the paper-facing verifier predicate |
| Closure | `sumcheckAccepted_of_acceptedForTable` | theorem | Theorem-Target | Fixed-table acceptance + table-sum equality reconstructs statement-indexed acceptance |
| Closure | `sumcheckAccepted_parameter_consistent` | theorem | Theorem-Target | Accepted transcript enforces parameter consistency |
| Closure | `sumcheckAccepted_degree_compatible` | theorem | Theorem-Target | Accepted transcript enforces degree compatibility |
| Closure | `sumcheckSoundness_constructive` | theorem | Theorem-Target | Constructive structural closure for the standalone scaffold (`accepted -> claimTrue`) |
| Closure | `sumcheckStructuralSoundness_constructive` | abbrev | Theorem-Target | Preferred structural alias for the standalone constructive soundness theorem |
| Closure | `sumcheckCompleteness_from_baseClaim_constructive` | theorem | Theorem-Target | Honest-transcript existence from legacy base claim |
| Closure | `sumcheckHonestTranscript_accepted_of_baseClaim` | theorem | Theorem-Target | Direct accepted proof for canonical honest transcript from base claim |
| Closure | `sumcheckPaperClaimTrue_of_baseClaim_constructive` | theorem | Theorem-Target | Build paper statement witness from legacy base claim |
| Closure | `sumcheckPaperClaimTrue_of_accepted` | theorem | Theorem-Target | Build paper statement witness from accepted transcript |
| Closure | `sumcheckPaperClaimTrue_iff_baseClaim` | theorem | Theorem-Target | Explicitly exposes that the standalone claim surface collapses to scaffold well-formedness |
| Closure | `sumcheckClaimTrue_iff_baseClaim` | theorem | Theorem-Target | Same collapse for the public standalone alias |
| Closure | `sumcheckFinalOracleConsistent_of_baseClaim_constructive` | theorem | Theorem-Target | Constructive final-oracle consistency for canonical statement + honest transcript |
| Closure | `sumcheckStatementTranscriptConsistent_of_baseClaim_constructive` | theorem | Theorem-Target | Constructive existence of statement/transcript consistency witness from base claim |
| Closure | `sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript` | theorem | Theorem-Target | For a chosen `stmt`, accepted + final-oracle consistency at same `tr` implies consistency |
| Closure | `sumcheckCompleteness_constructive` | theorem | Theorem-Target | Canonical standalone theorem-witness completeness (`claimTrue -> ∃ accepted transcript`) |
| Closure | `sumcheckStructuralCompleteness_constructive` | abbrev | Theorem-Target | Preferred structural alias for the standalone constructive completeness theorem |
| Closure | `sumcheckAssumptions_constructive` | def | Theorem-Target | Canonical constructive `SumCheckAssumptions` package |
| Boundary | `SumcheckSoundnessAssumption` | def | Boundary | `accepted → claimTrue` |
| Boundary | `SumcheckCompletenessAssumption` | def | Boundary | `claimTrue → ∃ tr, accepted` |
| Boundary | `SumCheckAssumptions` | structure | Boundary | Bundle of both |
| Bridge | `sumcheckClaimTrue_of_soundness` | theorem | Theorem-Target | Conditional soundness |

## Proof Obligations and Closure Plan

The module target is to provide:
- extraction/rejection theorems from the executable scaffold acceptance,
- a paper-facing verifier predicate `sumcheckVerifierAccepted`,
- constructive closure from accepted transcripts to Definition-6 theorem witnesses (`sumcheckSoundness_constructive`),
- constructive closure from Definition-6 theorem witnesses to accepted transcripts (`sumcheckCompleteness_constructive`),
- same-transcript closure bridges via `sumcheckFinalOracleConsistent_of_baseClaim_constructive`,
  `sumcheckStatementTranscriptConsistent_of_baseClaim_constructive`, and
  `sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript`,
- fixed-table endpoint surfaces (`sumcheckFinalOracleConsistentWithTable`, `sumcheckAcceptedForTable`) for downstream probabilistic soundness games.

## Assumption Ledger

- `SumcheckSoundnessAssumption`: abstract standalone soundness package shape `sumcheckAccepted -> sumcheckClaimTrue`.
- `SumcheckCompletenessAssumption`: abstract standalone completeness package shape `sumcheckClaimTrue -> ∃ tr, accepted`.
- `SumCheckAssumptions`: bundle of those two abstract theorem shapes.
- `sumcheckAssumptions_constructive`: canonical constructive inhabitant of that bundle for the standalone Definition-6 surface.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/MLE.lean`: MLE evaluators for sum-check claims.
- `SuperNeo/EqPoly.lean`: `eqPoly` for sum-check polynomial construction.

Downstream consumers:
- `SuperNeo/PiCCS.lean`: uses `SumCheckInstance`, `sumcheckAccepted` for Π_CCS.
- `SuperNeo/PiRLC.lean`: uses sum-check for Π_RLC.
- `SuperNeo/ProofSystem.SumCheck`: uses `SumCheckAssumptions` for proof-system composition.

## Implementation Plan

Keep the executable scaffold, the Definition-6 theorem witness surface, and the paper-facing verifier predicate aligned so that downstream protocol modules can consume either the standalone constructive package or the proof-system game package without changing the mathematical claim.

## Quality Expectations

Acceptance predicate must be a conjunction of independently testable components. Extraction theorems must provide one-step access to each component without pattern-matching on the conjunction.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Constructive closures (`sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`) are provided for the Definition-6 theorem-witness surface.

## Out of Scope

- A maximally generic reusable `SumCheck(T; Q)` library abstraction beyond the table/MLE realization used here.
- Full probabilistic soundness proof (Schwartz-Zippel) for a non-scaffolded interactive model.
- Interactive oracle proof (IOP) formalization.
- Concrete sum-check examples (live in `Checks.lean`).
