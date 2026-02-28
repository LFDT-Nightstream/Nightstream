# SuperNeo Lean Cross-Check (Standalone)

This folder is independent from `crates/` and the Rust workspace.
It implements SuperNeo math/protocol components in Lean and cross-checks them
against vectors generated from Rust (`neo-math`), while also exporting Lean
reference vectors consumed by Rust tests.

## Bidirectional Golden-Vector Strategy

- Rust -> Lean: large generated fixtures (trace/layout/protocol-style).
- Lean -> Rust: primitive conformance (Goldilocks, ring/eq/mle exporters).
- Keep both: system-level parity in Lean, primitive-level conformance in Rust.

## Final Goal (Paper-Faithful End-to-End Lean)

Target: a paper-faithful Lean formalization of SuperNeo end-to-end.

Completion criteria:
- protocol relations/transcripts/acceptance are modeled in Lean (not only shells),
- core claims are proved as quantified `Prop` theorems,
- final composition theorem `Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS` is proved with explicit assumptions,
- security boundaries are either proved here or explicit external assumptions,
- `lake exe check` remains a regression gate (never a substitute for universal proofs).

Reuse note:
- ArkLib reuse is allowed for Sumcheck/protocol work, with attribution in-file
  (repo/path/commit/license + modification note).

## How SuperNeo Is Defined In Lean (One Screen)

Primary review entrypoint:
- `import SuperNeo.ProofSystem`

Core protocol objects (paper-facing names):
- `SuperNeo.ProofSystem.ConstraintSystem.CCS`
- `SuperNeo.ProofSystem.ConstraintSystem.CE`
- `SuperNeo.ProofSystem.ConstraintSystem.CERelaxed`
- `SuperNeo.ProofSystem.Sumcheck.{Instance, Transcript, Accepted, ClaimTrue}`
- `SuperNeo.ProofSystem.Folding.PiCCS.*`
- `SuperNeo.ProofSystem.Folding.PiRLC.*`
- `SuperNeo.ProofSystem.Folding.PiDEC.*`
- `SuperNeo.ProofSystem.Protocol.FinalTheoremShape`

Canonical final endpoint:
- `SuperNeo.ProofSystem.Protocol.finalTheoremShape_of_assumptions`

Lean sketch:

```lean
import SuperNeo.ProofSystem

open SuperNeo.ProofSystem

-- Constraint-system layer
#check ConstraintSystem.CCS
#check ConstraintSystem.CE
#check ConstraintSystem.CERelaxed

-- SumCheck layer
#check Sumcheck.Accepted
#check Sumcheck.soundness
#check Sumcheck.completeness

-- Folding/reduction layers
#check Folding.PiCCS.soundness_relations
#check Folding.PiRLC.weak_relaxed
#check Folding.PiDEC.final_of_assumption

-- Final theorem shape
#check Protocol.FinalTheoremAssumptions
#check Protocol.FinalTheoremShape
#check Protocol.finalTheoremShape_of_assumptions
```

## Canonical Final Theorem Shape

The paper-facing endpoint is now fixed in code under:
- `SuperNeo.ProofSystem.Protocol.FinalTheoremAssumptions`
- `SuperNeo.ProofSystem.Protocol.FinalCompletenessStatement`
- `SuperNeo.ProofSystem.Protocol.FinalKnowledgeSoundnessStatement`
- `SuperNeo.ProofSystem.Protocol.FinalTheoremShape`

Shape targets:
- Completeness: honest premises imply `PSCEValid` for the composed output.
- Knowledge soundness / RoK boundary: explicit negligible total error budget.

Lean sketch:

```lean
def FinalCompletenessStatement (hA : FinalTheoremAssumptions) : Prop := ...
def FinalKnowledgeSoundnessStatement
  (hA : FinalTheoremAssumptions) (prob : Security.ProbModel) : Prop := ...
structure FinalTheoremShape
  (hA : FinalTheoremAssumptions) (prob : Security.ProbModel) : Prop where
  completeness : FinalCompletenessStatement hA
  knowledgeSoundness : FinalKnowledgeSoundnessStatement hA prob
```

Current status:
- Completeness statement shape is concrete and wired to existing reduction theorem wrappers.
- Knowledge-soundness is an explicit boundary shape (probability/extractor details still to be fully instantiated).

## Security Formalization Model

Security interfaces live in:
- `SuperNeo/ProofSystem/Security.lean`

Provided surfaces:
- `ProbModel` (minimal distribution/probability interface),
- `ErrorFn`, `IsNegligible`,
- `ErrorModel` with separate terms (`ε_sumcheck`, `ε_schwartzZippel`, `ε_binding`, `ε_relaxedBinding`, `ε_total`) and negligible proofs.

This is the canonical place for probability/error accounting; executable checks do not substitute these theorem-level boundaries.

## SumCheck Status (Explicit)

Current implementation route:
- `SuperNeo.Sumcheck` provides protocol objects plus acceptance/result relations.
- `SuperNeo.ProofSystem.Sumcheck` provides paper-facing wrappers and theorem surfaces.
- Full soundness/completeness is currently represented as explicit assumption boundaries:
  - `SuperNeo.SumcheckSoundnessAssumption`
  - `SuperNeo.SumcheckCompletenessAssumption`

Interpretation:
- SumCheck is not claimed as fully proved end-to-end yet.
- Final theorem statements remain paper-faithful by carrying these assumptions explicitly.

## Assumption Registry (Trusted Boundaries)

Canonical registry location:
- `SuperNeo.ProofSystem.Protocol.FinalTheoremAssumptions`

| Boundary | Current mode | Lean surface |
|---|---|---|
| Reduction composition (`Π_CCS`, `Π_RLC`, `Π_DEC`) | Assumed boundary (threaded as structured assumptions) | `FinalTheoremAssumptions.reduction` |
| SumCheck soundness | Assumed boundary (explicit) | `FinalTheoremAssumptions.sumcheckSoundnessBoundary` |
| SumCheck completeness | Assumed boundary (explicit) | `FinalTheoremAssumptions.sumcheckCompletenessBoundary` |
| Schwartz-Zippel | Assumed boundary (explicit) | `FinalTheoremAssumptions.schwartzZippelBoundary` |
| Ajtai binding | Assumed boundary (explicit) | `FinalTheoremAssumptions.ajtaiBindingBoundary` |
| Ajtai relaxed binding | Assumed boundary (explicit) | `FinalTheoremAssumptions.ajtaiRelaxedBindingBoundary` |
| Low-norm invertibility | Assumed boundary (explicit typed assumption) | `FinalTheoremAssumptions.lowNormInvertibilityBoundary` |
| Error accounting (`ε_*`, `ε_total`) | Structured theorem surface (negligibility obligations explicit) | `SuperNeo.ProofSystem.Security.ErrorModel` |

## No-Checks-In-Proofs Policy (Enforced)

Policy:
- The proof-facing layer must not import regression modules:
  - `SuperNeo.Checks`
  - `SuperNeo.Generated.*`
  - `SuperNeo.Regression`
- Regression checks stay isolated to runtime/parity entrypoints.

Enforcement:
- `lake exe check` now includes `proof_import_wall=<bool>`.
- `all_checks=true` requires `proof_import_wall=true`.
- The guard scans:
  - `SuperNeo/ProofSystem/**`
  - layer entrypoints (`CoreMath`, `PaperMath`, `ProtocolBase`, `ProtocolTrack`, `Composition`)

## Build & Check

```bash
cd formal/superneo-lean
lake build
lake exe check
```

Expected tail:

```text
proof_import_wall=true
all_checks=true
```

### Vector regeneration

Rust -> Lean:

```bash
cargo run --manifest-path formal/superneo-lean/rust-vectors/Cargo.toml
```

Lean -> Rust:

```bash
cd formal/superneo-lean
lake exe goldilocks-golden > SuperNeo/Generated/GoldilocksGolden.csv
lake exe ring-golden > SuperNeo/Generated/RingGolden.csv
lake exe eq-mle-golden > SuperNeo/Generated/EqMleGolden.csv
lake exe p9-p11-p12-golden > SuperNeo/Generated/P9P11P12Golden.csv
lake exe p13-p14-golden > SuperNeo/Generated/P13P14Golden.csv
```

Rust conformance:

```bash
cargo test -p neo-math --release goldilocks_matches_lean_golden_vectors
cargo test -p neo-math --release ring_mul_matches_lean_golden_vectors
cargo test -p neo-memory --release eq_mle_matches_lean_golden_vectors
cargo test -p neo-math --release p9_p11_p12_matches_lean_golden_vectors
cargo test -p neo-math --release p13_p14_matches_lean_golden_vectors
```

## Layout by Layer

Layer entrypoints:
- `SuperNeo/CoreMath.lean`
- `SuperNeo/PaperMath.lean`
- `SuperNeo/ProtocolBase.lean`
- `SuperNeo/ProtocolTrack.lean`
- `SuperNeo/Composition.lean`
- `SuperNeo/Regression.lean`

Algebra primitives (no protocol knowledge):
- `Field`, `Ring`, `Norm`, `Decomp`, `EqPoly`, `MLE`, `CoeffMaps`, `Embedding`, `Interp`

Paper definitions and identities:
- `BarLift`, `MatrixTransform`, `EvalLink`, `EvalHom`, `ModuleHom`, `Thm3Core`, `InvertibilityAxioms`, `SamplingSet`, `PolyLemmas`

Protocol relations and reductions:
- `ProtocolRelations`, `Sumcheck`, `PiCCS`, `PiRLC`, `PiDEC`, `InteractiveReductions`

Composition chain:
- `P20 -> P21 -> ProtocolReduction -> ProtocolTheorem`

Paper-facing API (stable surface):
- `ProofSystem.lean` facade
- `ProofSystem/Security.lean`
- `ProofSystem/Types.lean`
- `ProofSystem/ConstraintSystem{,.lean,/CCS.lean}`
- `ProofSystem/Sumcheck.lean`
- `ProofSystem/Folding{,.lean,/PiCCS.lean,/PiRLC.lean,/PiDEC.lean}`
- `ProofSystem/Protocol.lean`

Regression/parity:
- `Checks`, `Generated/Vectors`, golden exporters + CSVs

## Repository Alignment

The following advertised entrypoints/facades are present in-tree (not planned-only):
- `SuperNeo/CoreMath.lean`
- `SuperNeo/PaperMath.lean`
- `SuperNeo/ProtocolBase.lean`
- `SuperNeo/ProtocolTrack.lean`
- `SuperNeo/Composition.lean`
- `SuperNeo/ProofSystem.lean` plus `ProofSystem/{ConstraintSystem,Sumcheck,Security,Folding,Protocol}.lean`

### Paper-facing imports

Preferred layered imports:
- `import SuperNeo.ProofSystem.ConstraintSystem`
- `import SuperNeo.ProofSystem.Sumcheck`
- `import SuperNeo.ProofSystem.Folding`
- `import SuperNeo.ProofSystem.Protocol`

Or single facade:
- `import SuperNeo.ProofSystem`

## Top Blockers For Paper-Level Soundness

- `P6`: constructive split/recompose proof from definitions (not wrapper-level only).
- `P10/P11/P14`: universal theorem chain closure for inner-product transform, lift, and eval-hom.
- `M21`: full SumCheck theorem package with explicit soundness-error statement shape.
- `M23/M24/M25`: strong/weak/knowledge reductions instantiated against paper-faithful protocol objects.
- `M26/M27`: final composition + final SuperNeo theorem in paper-level quantifier form.

## Check Output Breakdown (`lake exe check`)

| Output flag | What `true` means | Evidence | Remaining gap |
|---|---|---|---|
| `proof_import_wall` | proof-facing modules do not import regression/check modules | enforced by `Main.lean` import-wall scan | Keep module boundaries clean as code evolves |
| `superneo_cases` | `ct(bar(a)*b)=dot(a,b)` on generated cases | Rust vectors + Lean recomputation | Universal theorem |
| `ring_mul_cases` | `mulRq` matches generated coefficients | Rust vectors | Universal quotient-ring proof |
| `norm_cases` | `normInfCoeffs` matches generated norms | Rust vectors | General norm theorem package |
| `split_cases` | split/recompose/bounds pass on generated cases | Rust vectors + invariants | Constructive universal decomposition |
| `eq_cases` | `eqPoly` matches generated/Boolean behavior | Rust vectors + invariants | Full quantified selector integration |
| `mle_cases` | inner-product MLE equals folding MLE | Rust vectors + identity check | Universal MLE identity theorem |
| `embedding_vec_cases` | vector embed/unembed parity + round-trip | Rust vectors + invariants | Full bijection/linearity theorem |
| `embedding_matrix_cases` | matrix embed/unembed parity + round-trip | Rust vectors + invariants | Full matrix bijection/linearity theorem |
| `bar_lift_vec_cases` | vector bar-lift parity + linearity checks | Rust vectors + invariants | Universal Definition 8 theorem |
| `bar_lift_matrix_cases` | matrix bar-lift parity checks | Rust vectors | Universal matrix lift theorem |
| `matrix_transform_cases` | `Mz = ct(bar(M)z)` parity checks pass | Rust vectors + invariants | Universal Theorem 4 derivation |
| `eval_link_cases` | eval-link computations/identities match | Rust vectors + invariants | Universal Remark 2 theorem |
| `eval_hom_cases` | eval-hom linear-combination checks pass | Rust vectors + invariants | Universal Theorem 5 proof |
| `module_hom_cases` | representative module-hom sanity passes | Fixed witnesses | Abstract theorem-native module-hom layer |
| `invertibility_cases` | concrete invertibility preconditions hold | Constant checks | Prove/justify assumption boundary |
| `sampling_cases` | sampling/expansion checks pass | Rust vectors + bound checks | Universal Theorem 9 proof |
| `eq_lift_cases` | eq-lift table checks pass | Rust vectors + checks | Universal Appendix C lemmas |
| `poly_lemma_cases` | SZ interface sanity passes | Fixed witnesses | General SZ theorem package |
| `coeff_map_cases` | coeff-map round-trips/sanity pass | Mixed vectors + checks | Complete inverse/linearity theorem suite |
| `parameter_cases` | parameter and shape sanity passes | Constant/invariant checks | Full downstream inequality closure |
| `interp_cases` | interpolation/eval parity checks pass | Rust vectors | General interpolation correctness/uniqueness |
| `all_checks` | conjunction of all checks is `true` | Aggregate gate | No new math content |

## Proof Dependency Map (`P1..P21` + `M21..M27`)

```text
                    CORE MATH
===============================================================
    ALGEBRA                      NORMS                   POLYNOMIALS
------------------           -----------------       ------------------

Field/Ring (P1) --> Coeff maps (P3)    Field/Ring (P1) --> Norm (P5)       Field/Ring (P1) --> Eq poly (P7)
Field/Ring (P1) --> Ring arith (P4)                        |                                    |
                    Coeff maps (P3) --> Embedding (P9)     v                                    v
                                             |    Norm (P5) --> split_b (P6)           Eq poly (P7) --> MLE (P8)
                                             v                                                          |
Thm 3: inner product (P10) <-- Params (P2)                                                             v
         |                          Params (P2) + Norm (P5) --> Thm 8: invertibility (P16)     MLE (P8) --> SZ lemmas (P18)
         v                                                              |                                    |
Def 8: bar-lift (P11)                                                   v                                    v
         |                                                   Thm 9: sampling (P17)              Interpolation (P19)
         v
Thm 4: Mz = ct(bar(M)z) (P12)
         |
         v                              Coeff maps (P3) --> Module hom (P15)
Remark 2: eval/ct link (P13)                                     |
         |                                                        |
         +-----------------------+ Module hom (P15) --> Thm 5: eval hom (P14)


===============================================================
Arithmetic obligations (P20)  <-- split_b, Thm 4, Thm 5, Thm 8, Thm 9, SZ, Interp
         |
         v
Protocol target (P21)  <-- Thm 3, Thm 4, Thm 5, Thm 8, Thm 9, SZ, Interp, P20
===============================================================


    PROTOCOL REDUCTIONS (paper-faithful end-to-end target)
----------------------------------------------------------------
CCS/CE relations (M22) <-- Eq poly, MLE, SZ, Interp, Thm 3, Thm 4, Thm 5, Thm 8, Thm 9, P20, P21

SumCheck (M21)
     |
     v
Π_CCS: strong IR (M23)
     |
     v
Π_RLC: weak IR (M24)
     |
     v
Π_DEC: reduction of knowledge (M25)
     |
     v
Strong/Weak composition — paper Thm 6 (M26)
     |
     v
Final SuperNeo theorem (M27)
```

## Status Summary

| ID | Paper concept | Status | Remaining gap |
|---|---|---|---|
| P1 | Field/ring dimensions | Done | — |
| P2 | Concrete parameters | Done | — |
| P3 | Coefficient maps and `ct` bridge | Done | — |
| P4 | Quotient-ring arithmetic | Done | — |
| P5 | Centered norm theorem layer | Done | Tighten downstream usage cleanup |
| P6 | `split_b` decomposition | In progress | Extend constructive closure beyond native base-2 challenge path (`k ≥ 8`) |
| P7 | `eq` selector behavior | Done | Full integration across protocol proofs |
| P8 | MLE identity | In progress | Fully quantified theorem closure |
| P9 | Embedding bijection/linearity | In progress | Complete theorem-native linearity suite |
| P10 | Theorem 3 inner-product transform | In progress | Universal derivation from foundations |
| P11 | Definition 8 bar-lift | In progress | Universal theorem closure |
| P12 | Theorem 4 matrix transform | In progress | Universal theorem closure |
| P13 | Remark 2 eval/`ct` linkage | In progress | Strong theorem-native integration |
| P14 | Theorem 5 eval homomorphism | In progress | Full theorem-native proof |
| P15 | Module-hom abstraction | In progress | Rich abstract theorem layer |
| P16 | Theorem 8 invertibility | In progress | Prove or keep explicit trusted boundary |
| P17 | Theorem 9 sampling expansion | In progress | Universal theorem |
| P18 | SZ and eq-lift lemmas | In progress | Full quantified lemma set |
| P19 | Interpolation correctness | In progress | Uniqueness/correctness theorem package |
| P20 | Arithmetic obligations bundle | In progress | Remove remaining compatibility wrappers |
| P21 | Protocol target composition | In progress | Paper-exact theorem phrasing and closure |
| M21 | SumCheck formalization | In progress | Paper-exact rounds + theorem-level soundness bound statement (error term explicit) + completeness |
| M22 | CCS/CE relations | In progress | Complete paper-definition alignment |
| M23 | `Π_CCS` strong IR theorem | In progress | Full paper instantiation |
| M24 | `Π_RLC` weak IR theorem | In progress | Full paper instantiation |
| M25 | `Π_DEC` RoK theorem | In progress | Extractor-grade final theorem |
| M26 | Strong/weak composition (Thm 6) | In progress | Generic theorem-level composition closure |
| M27 | Final SuperNeo theorem | In progress | End-to-end paper-faithful statement/proof |

## Full-Proof Completion Contract

- SuperNeo is fully proven in Lean only when required `P*` and `M*` items are `Done`.
- `all_checks=true` is necessary for regression confidence, not sufficient for full proof completion.
- Remaining assumption boundaries must be proved in-repo or explicitly documented in final theorem statements.
- Paper-faithful end-to-end requires:
  - protocol objects for SumCheck / `Π_CCS` / `Π_RLC` / `Π_DEC` (not only wrappers),
  - theorem-native composition over those objects,
  - explicit assumption accounting in final theorem statements.

## Paper Sources

- `docs/superneo-paper/02_2_Technical_overview.md`
- `docs/superneo-paper/04_4_Preliminaries.md`
- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/06_6_Strong_and_weak_interactive_reductions.md`
- `docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
- `docs/superneo-paper/11_B_Concrete_parameters.md`
- `docs/superneo-paper/12_C_Additional_Background.md`
- `docs/superneo-paper/13_D_Deferred_theorems_and_proofs.md`
