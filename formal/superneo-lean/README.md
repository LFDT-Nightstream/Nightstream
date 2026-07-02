# SuperNeo Lean Cross-Check (Standalone)

This folder is intentionally outside `crates/` and independent from the Rust workspace.

It provides the theorem-facing Lean implementation of core SuperNeo/Neo math surfaces.
Lean is the mathematical source of truth.

Operationally:
- Lean theorem/definition surfaces are authoritative.
- The maintenance boundary is Lean-only.
- The older Rust-vector/artifact conformance lanes were removed; they remain
  available in git history if a cross-language gate is ever revived.

## What is checked

- SuperNeo inner-product transform identity:
  - `ct(cf_inv(bar(a)) * cf_inv(b)) == dot(a, b)`
- Ring multiplication in `R_q = F_q[X]/(X^54 + X^27 + 1)`
- Centered `l_inf` norm on field/ring elements
- Balanced `split_b` decomposition and recomposition
- `eq(x,y)` polynomial behavior (Boolean and non-Boolean inputs)
- MLE identity via inner-product form vs. folding form
- Definition 7 coefficient embedding round-trips (element/vector/matrix)
- Definition 8 bar-lift transform checks (vector and matrix forms)
- Theorem 4 computational identity checks: `Mz = ct(bar(M)z)`
- Remark 2 evaluation/`ct` linkage checks
- Theorem 5 linear-combination evaluation homomorphism checks
- Module-homomorphism sanity checks used by evaluation homomorphism
- Theorem 8 assumption boundary + concrete precondition checks
- Definition 17/Theorem 9 strong-sampling + expansion-factor checks
- Appendix C Lemma 6-style eq-lifting table checks (+ SZ bound interface sanity)
- Neo-style polynomial interpolation/evaluation helpers (base field)

## Layout

The directory structure mirrors the paper's four main sections. Each section
directory is paired with a same-named top-level barrel module that re-exports
it, and every implementation module `X.lean` sits next to its machine-checked
boundary `XInterface.lean` (human spec in `specs/X.spec.md`).

All modules under `SuperNeo/` — including every `*Interface.lean` — are part
of the default `lake build` (recursive lib glob), so a drifted interface fails
the build instead of rotting silently.

| Directory | Barrel | Paper section | Contents |
|---|---|---|---|
| `SuperNeo/Primitives/` | `SuperNeo/Primitives.lean` | Section 4 (Preliminaries) | Goldilocks field/extension, ring `R_q`, coefficient maps, norms, balanced decomposition, `eq`/MLE/sum-check cores, interpolation, Appendix B.2 parameters |
| `SuperNeo/EmbeddingTheory/` | `SuperNeo/EmbeddingTheory.lean` | Section 5 (Defs 7-8, Thms 3-5) | Definition 7 embeddings, Theorem 3 core, Definition 8 bar-lift, Theorem 4 matrix transform, Remark 2 eval linkage, Definition 15 module homs, Theorem 5 eval homomorphism |
| `SuperNeo/SecurityModel/` | `SuperNeo/SecurityModel.lean` | Section 6 (Defs 9-10, 16-18, Thms 2, 6, 8-9) | Weak/strong interactive reductions, Theorem 8 invertibility (axioms + constructive Goldilocks), Definition 17/Theorem 9 sampling sets |
| `SuperNeo/FoldingProtocol/` | `SuperNeo/FoldingProtocol.lean` | Section 7 (Defs 11-14, Lemmas 3-4, Thm 7) | Section 7.1 relations/data/context owners, Π_CCS, Π_RLC, Π_DEC, arithmetic bundle/obligations, protocol target(s), final protocol theorem, parent-authority Fiat-Shamir reroute lemma |
| `SuperNeo/ProofSystem/` | `SuperNeo/ProofSystem.lean` | Proof-system facade | Types, probability/error model, lattice assumptions (MSIS/Ajtai), CCS constraint system, sum-check facade, protocol entrypoint |
| `SuperNeo/Golden/` | — (separate `goldilocks-golden` exe) | — | Golden-value executable, excluded from the theorem import wall |

To locate a paper claim: pick the section directory, open the module named
after the construction (for example `FoldingProtocol/PiDEC.lean` for Section
7.5), and read its `...Interface.lean` neighbor for the curated theorem
surface with paper line anchors.

## Run Lean checks

```bash
cd formal/superneo-lean
lake build
lake exe check
```

`lake exe check` currently verifies only the theorem import wall for the
maintained Lean package. The older Rust-vector and Rust-artifact conformance
commands remain in the repo only as archived integration machinery and are not
part of the maintained default build.

## Run SumCheck tests

The SumCheck test suite lives under `tests/` and is elaboration-driven:
`#guard` checks and `example` proofs fail at compile time.

```bash
cd formal/superneo-lean
lake build SumCheckTests
```

`lake build SumCheckTests` currently includes:
- standalone/core SumCheck smoke tests,
- proof-style SumCheck examples,
- prefix-soundness smoke tests for the executable proof-system path.

This complements `lake exe check`; it does not replace the global regression gate.

Expected output ends with:

```text
all_checks=true
```

## Check Output Breakdown (`lake exe check`)

`lake exe check` now reports:

- `proof_import_wall=true` when theorem-facing barrel modules do not import the
  archived generated/vector/regression surfaces
- `all_checks=true` when that import-wall gate passes

## Paper-Faithful Proof-Complete Goal (Project Policy)

The target for this project is **paper-faithful proof-complete closure** of the protocol and its subparts.
This means:

1. Final protocol goal:
   - `S7.6` (Protocol Theorem) is proved from theorem-native dependencies (not only executable checks/skeleton wrappers).
2. Subpart goal:
   - each required milestone on the critical path is closed with quantified theorem surfaces.
3. Boundary policy:
   - `Done (Boundary)` is an intermediate milestone (interface/assumption boundary closed),
     **not** the final project state.
4. Check policy:
   - `lake exe check` must remain green, but checks are regression evidence, not substitutes for universal proofs.
5. Paper-faithfulness policy:
   - completion requires proving the same mathematical construction stated in the paper,
     not only an equivalent-by-definition interface surface.
   - if an executable construction exists (e.g. folding evaluator), the theorem-facing claim must include
     a proved bridge from that executable construction to the paper formula.
6. Trusted assumptions:
   - any remaining trusted assumption must be explicit, minimized, and documented with closure intent.

## Status Labels

| Label | Meaning | Final for project? |
|---|---|---|
| `Accepted (SuperNeo path)` | The exact concrete dependency chain used by the active SuperNeo theorem is closed and paper-faithful, but a broader reusable/generalized version of the same result may still be open. | Yes, for current repo scope |
| `Done (Boundary)` | The local theorem/module is proved from an explicit assumption bundle or boundary surface, but the upstream provider of that bundle is still open. Downstream modules can consume it; repo-wide closure has not reached the source of the assumptions yet. | No |
| `Done (Witness-Level)` | The algebraic/witness-level content of the paper claim is proved: relation-level implications, constructive witness manipulations, and advantage bounds for the failure events of carried witnesses. The paper's probabilistic interactive-reduction statement itself (Definition 5's `(G, K, P, V)` protocol object, PPT adversaries and malicious provers, `⟨P*, V⟩` executions, EPT extractors, and the success-probability inequalities of Definitions 9-10) is not formalized. See `Faithfulness Boundary`. | Final for the algebraic layer; open against the closure standard |
| `In progress` | Some theorem surfaces exist, but proof obligations remain open. | No |
| `Good shell` | Composition skeleton exists; full derivation is not complete. | No |
| `Done (Proof-Complete)` | The paper-faithful theorem is closed at the module itself: no local placeholder bundle remains except the intended theorem-level assumptions (for example, the paper's cryptographic hardness assumption). | Yes |

At the current project state, every tracked milestone row is `Done (Proof-Complete)`
except the reduction-flavored rows (S6.1, S7.2-S7.4, S7.6 / M20-M22, M34-M36, M38),
which are `Done (Witness-Level)`; see the `Faithfulness Boundary` section for the
precise gap. The remaining label definitions are retained because they were used
during closure and still matter for interpreting older discussions.

### How To Read These Labels

- `Accepted (SuperNeo path)` means "good enough for the actual SuperNeo theorem route, not yet maximally generalized."
  Example: this label is reserved for cases where the active theorem route is closed while a broader reusable generalization is intentionally deferred.

- `Done (Boundary)` means "this module works if you hand it the right upstream theorem bundle, but that upstream bundle is not fully eliminated yet."
  Example: this is the status to use when a local theorem is closed only after assuming a still-open upstream provider bundle, rather than from the intended paper theorem inputs directly.

- `Done (Proof-Complete)` means "the module's own paper claim is proved directly, with only the intended theorem-level assumptions left explicit."
  Example: `S6.3` is proof-complete because Theorem 8's invertibility claim is proved constructively at the paper's concrete Goldilocks floor.

- `Done (Witness-Level)` means "the algebraic content is fully proved, but the paper states this claim about adversaries and extractors, and that probabilistic layer is not formalized."
  Example: `S7.2` proves that one CE witness yields the strong Π_CCS statement, but Lemma 3's quantification over malicious provers with an extractor is not modeled.

## Current Practical Reading

- One-sentence status: the tracked SuperNeo formalization is proof-complete at the algebraic/witness level; the paper's probabilistic interactive-reduction layer is not formalized.
- The active native Goldilocks / `paperCarrier`-difference route is closed end-to-end through `S7.6` at that witness level, with the theorem-level MSIS hardness assumption as the only explicit security input.
- Definition-, parameter-, and arithmetic-flavored rows (Sections 4-5, S6.2-S6.5, S7.1, S7.5) are `Done (Proof-Complete)`; the reduction-flavored rows (S6.1, S7.2-S7.4, S7.6) are `Done (Witness-Level)`.
- The largest open formalization gap is therefore the adversary/extractor framework itself; see `Faithfulness Boundary`.

## Faithfulness Boundary (Witness-Level vs Paper-Probabilistic)

The reduction-flavored milestones (M20-M22, M34-M36, M38; rows S6.1,
S7.2-S7.4, S7.6) are closed at the witness level, not at the paper's
probabilistic level.

Formalized and machine-checked:

- the relation-level reductions (CE implies claim truth, relaxed CE, and
  invertibility-witness existence) and their composition,
- constructive SumCheck soundness/completeness for the table-based
  realization,
- zero- or negligible-advantage bounds for the failure events of carried
  witnesses (SumCheck transcript failure, the concrete Schwartz-Zippel
  interpolation check, MSIS/Ajtai breaks) under an abstract probability
  model, and
- the commitment-level binding-collision-to-MSIS extractors of Theorem 2.

Not formalized:

- Definition 5's interactive-protocol object `(G, K, P, V)`,
- PPT adversaries and malicious provers, and `⟨P*, V⟩` protocol executions
  as random variables,
- EPT extractors and the success-probability inequalities that
  Definitions 9-10 and Theorem 6 quantify over.

A consequence worth keeping in mind: the SumCheck acceptance predicate is
table-authoritative, so its constructive soundness theorem does not model a
cheating prover against a randomized verifier; the probabilistic content on
that side lives in the Lund/prefix advantage-bound layer instead.

Closing this gap means formalizing the adversary/extractor framework and
restating Lemmas 3-4, Theorem 7, and Theorems 1/6 against it. Until then,
`Done (Witness-Level)` rows are final for the algebraic layer only, and the
closure standard's "quantified theorem level" bar remains open for them.

## Opening-Convergence Follow-On Frontier

The tracked SuperNeo milestone set is closed, but there is one historical
follow-on theorem frontier from the retired `opening-convergence-lean` package
(removed from the tree; recoverable from git history), which reached its own
local closure:

- the existing SumCheck formalization is still base-field (`SuperNeo.F`) only
- Nightstream opening convergence Phase 1 is over the extension field
  `SuperNeo.KExt`

The first two prerequisite layers are now in place:

1. `SuperNeo/Primitives/ExtensionField.lean` provides the quadratic extension carrier,
2. `SuperNeo/Primitives/ExtensionMLE.lean` provides the extension-field MLE/equality +
   linearity layer,
3. `SuperNeo/Primitives/ExtensionSumCheck.lean` provides the extension-field
   Definition-6 protocol surface and verifier-side acceptance scaffold.

So the next paper-faithful closure target is the **soundness/completeness
closure of extension-field SumCheck**:

1. accepted-transcript to claim-truth closure over `SuperNeo.KExt`,
2. the terminal-value / honest-table theorem needed to replace the carried
   `sumcheckTerminalCorrect` hypothesis from the retired
   `opening-convergence-lean` package,
3. the proof-system-level generalization or specialization needed to connect
   that closure into `ProofSystem/SumCheck/General.lean`

This is not a regression in the tracked SuperNeo milestone table. It is a new
consumer-driven generalization frontier opened by the now-closed
opening-convergence package.

## Reader Guide

If you only need the conclusion, read in this order:

1. This README:
   - `Current Practical Reading`
   - `Milestone Table`
   - `Status Summary`
2. The capstone theorem docs:
   - `specs/ProtocolTheorem.spec.md`
   - `SuperNeo/FoldingProtocol/ProtocolTheoremInterface.lean`
   - `SuperNeo/FoldingProtocol/ProtocolTheorem.lean`
3. The main prerequisite bridges:
   - `specs/SumCheck.spec.md`
   - `specs/ProtocolRelations.spec.md`
   - `specs/InteractiveReductions.spec.md`
   - `specs/ProofSystem/LatticeReductions.spec.md`

Operationally:
- `lake build` + `lake exe check` show the repo is green.
- The milestone table tells you which paper claim each module closes.
- The specs say what each module means mathematically.
- The interface files show the typed theorem-facing surface exported to downstream modules.

## Proof Dependency Map (Section-Aligned)

Milestones are aligned to paper sections. Each ID `S<section>.<item>` maps to a specific
paper Definition, Theorem, or Lemma.

```text
    SECTION 4: PRIMITIVES                SECTION 5: EMBEDDING THEORY
    ========================             ============================

    S4.1 Field/Ring/Dims                 S5.1 Embedding (Def 7)
      |                                    |
      +---> S4.2 Norm/Decomp              v
      |       |                          S5.2 Thm 3 core <--- S4.6
      +---> S4.3 EqPoly/MLE               |
      |       |                            v
      |       +---> S4.5 PolyLemmas      S5.3 BarLift (Def 8)
      |                Interp              |
      |                                    v
      +---> S4.4 SumCheck               S5.4 MatrixTransform (Thm 4)
      |                                    |
      +---> S4.6 Parameters               +---> S5.5 EvalLink + ModuleHom
                                           |       |
                                           |       v
                                           +---> S5.6 EvalHom (Thm 5)


    SECTION 6: SECURITY MODEL           SECTION 7: FOLDING PROTOCOL
    ==========================           ============================

    S6.1 InteractiveReductions           S7.1 CCS Relations (Defs 11-14)
         (Defs 9-10, Thm 6)               |
                                           v
    S6.2 Lattice/MSIS/Ajtai             S7.2 Π_CCS  (Lemma 3) <--- S4.4
         (Defs 4, 16, 18, Thm 2)          |
      |                                    v
      +---> S6.3 Invertibility           S7.3 Π_RLC  (Lemma 4)
      |      (Thm 8) <--- S4.2            |
      |       |                            v
      +---> S6.4 Sampling               S7.4 Π_DEC  (Thm 7)
             (Def 17, Thm 9)              |
                                           v
    S6.5 Error/Negligible Model          S7.5 Arithmetic Obligations
                                              <--- S4.2, S4.5, S5.4, S5.6,
                                                   S6.3, S6.4
                                           |
                                           v
                                         S7.6 Protocol Theorem (Thm 1)
                                              <--- S5.2, S6.1, S6.2, S6.5,
                                                   S7.2, S7.3, S7.4, S7.5
```

## Milestone Table

| ID | Paper item | Lean modules | Core claim target | Depends on | Enables | Status |
|---|---|---|---|---|---|---|
| `S4.1` | Defs 1-2 (Field/Ring/Dims) | `Field.lean`, `Dimensions.lean`, `Ring.lean`, `CoeffMaps.lean` | Base field/ring algebra, coefficient maps, `ct` bridge. | - | S4.2, S4.3, S4.4, S5.1 | Done (Proof-Complete): the base field/ring algebra and coefficient-map theorem surfaces are proved constructively, including round-trip, `ct` compatibility, and ring-shape preservation. |
| `S4.2` | Def 3 + decomposition | `Norm.lean`, `Decomp.lean` | Centered `l_∞` norm bounds, `split_b` recomposition. | S4.1 | S6.3, S6.4, S7.5 | Done (Proof-Complete): centered norm bounds and balanced/base-2 decomposition round-trip, field-lift, and bool↔prop closure are all proved constructively. |
| `S4.3` | `eq` polynomial + MLE | `EqPoly.lean`, `MLE.lean` | `eq` is Boolean-cube selector; MLE identity `ṽ(r) = ⟨v, r̂⟩`. | S4.1 | S4.5, S5.5 | Done (Proof-Complete): `eq` is closed on the Boolean cube, and `MLE.lean` proves identity, folding equivalence, chi/dot equivalence, and linearity packages. |
| `S4.4` | Def 6 (sum-check) | `SumCheck.lean` | Sum-check soundness/completeness boundary. | S4.1 | S7.2 | Done (Proof-Complete): `SumCheck.lean` now exposes a Definition-6 theorem witness object `SumCheckDefinition6Statement`, proves constructive soundness/completeness directly against that surface, and constructs honest transcripts for arbitrary verifier challenge vectors of the right length; the underlying realization remains table/MLE-based, but no local theorem gap remains. |
| `S4.5` | Lemmas 5-6, interpolation | `PolyLemmas.lean`, `Interp.lean` | Schwartz-Zippel, eq-lifting, interpolation correctness. | S4.3 | S7.5 | Done (Proof-Complete): `Interp.lean` gives constructive interpolation correctness/uniqueness, and `PolyLemmas.lean` gives quantified Boolean-cube eq-lift closure plus theorem-native Schwartz-Zippel bound bridges. |
| `S4.6` | App B.2 parameters | `Parameters.lean`, `Goldilocks.lean` | Concrete constants and bound checks. | - | S5.2, S6.3 | Done (Proof-Complete): Appendix B.2 constants, positivity facts, and the core Goldilocks bound checks are proved constructively in-module. |
| `S5.1` | Def 7 (embedding) | `Embedding.lean` | Element/vector/matrix embedding bijection + linearity. | S4.1 | S5.2 | Done (Proof-Complete): element/vector/matrix embedding bijection and linearity are proved constructively, and `p9EmbeddingAssumption_holds` closes the combined package. |
| `S5.2` | Thm 3 (inner-product transform) | `Thm3Core.lean` | `ct(cf⁻¹(bar(a)) · cf⁻¹(b)) = ⟨a, b⟩`. | S4.1, S4.6, S5.1 | S5.3, S7.6 | Done (Proof-Complete for the native paper instance via `thm3CoreAssumption_native`; generic closure is provided as finite basis criterion/checker `thm3BasisKernelCheck`). |
| `S5.3` | Def 8 (bar-lift) | `BarLift.lean` | Blockwise lifting is correct and linear. | S5.1, S5.2 | S5.4 | Done (Proof-Complete) for module-level theorem closure (`barLiftVector_add_constructive`, `barLiftVector_scale_constructive`, `barLiftLinearityAssumption_closed`). |
| `S5.4` | Thm 4 (matrix-vector transform) | `MatrixTransform.lean` | `Mz = ct(bar(M)z)` for all valid M, z. | S5.2 | S5.5, S5.6, S7.5 | Done (Proof-Complete): Theorem 4 is proved constructively from Theorem 3 by block decomposition, and the module now exposes theorem-native entrypoints from `thm3CoreAssumption`, the finite basis-kernel witness, and the finite basis-kernel checker. |
| `S5.5` | Remark 2 + Def 15 | `EvalLink.lean`, `ModuleHom.lean` | Eval/`ct` linkage; module-hom linearity. | S4.1, S5.4 | S5.6 | Done (Proof-Complete): eval-link and module-hom quantified theorem/check bridges are proved in-module; remaining generic gaps are upstream, not in these local shells. |
| `S5.6` | Thm 5 (eval homomorphism) | `EvalHom.lean` | Linear-combination preservation under evaluation. | S5.4, S5.5 | S7.5 | Done (Proof-Complete): theorem-native closure is proved constructively from MLE linearity, and all eval-hom boundary constructors are derived in-module. |
| `S6.1` | Defs 5, 9-10, Thm 6 | `InteractiveReductions.lean` | Weak/strong reductions compose correctly. | - | S7.6 | Done (Witness-Level): strong/weak composition theorems are proved from `InteractiveReductionAssumptions` (one `ProtocolTargetAssumptions` bundle plus one accepted SumCheck transition witness) by composing the Π_CCS/Π_RLC/Π_DEC theorems; the Definition-5/9/10 adversary-extractor model that Theorem 6 quantifies over is not formalized. |
| `S6.2` | Defs 4, 16, 18, Thm 2 | `ProofSystem/Lattice.lean`, `ProofSystem/LatticeReductions.lean`, `ProofSystem/LatticePaper.lean` | Ajtai commitment properties, MSIS hardness, binding reductions. | - | S6.3, S7.6 | Done (Proof-Complete): Defs 4/16/18 and Theorem 2 are proved constructively; the generic carrier route leaves only the paper's intended `samplingCarrier` + strong-sampling inputs explicit, and the active Goldilocks `paperCarrier` route reconstructs the full Ajtai reduction package directly from theorem-level MSIS hardness. |
| `S6.3` | Thm 8 (invertibility) | `InvertibilityAxioms.lean`, `InvertibilityGoldilocks.lean` | Low-norm invertibility preconditions and interface. | S4.2, S4.6, S6.2 | S6.4, S7.5 | Done (Proof-Complete): the theorem surface is shape-aware (`hasRingDegreeShape a → 0 < ‖a‖∞ < B → invertibleRq a`), `InvertibilityGoldilocks.lean` proves the concrete Goldilocks theorem at the paper floor `goldilocksPaperBInv = 383`, the narrower threshold-`5` route is a corollary, and the active `paperCarrier`-difference route is derived from that constructive theorem in-repo. |
| `S6.4` | Def 17 + Thm 9 (sampling) | `SamplingSet.lean` | Strong-sampling + expansion-factor interface. | S4.2, S6.3 | S7.5 | Done (Proof-Complete) for module-level contract surfaces (`samplingDiffSet`, `strongSamplingExpansionProp`, and associated theorem wrappers). |
| `S6.5` | Error/negligible model | `ProofSystem/{Types,Security,Negligible}.lean` | `ProbModel`, `ErrorModel`, `IsNegligible`. | - | S7.6 | Done (Proof-Complete): the canonical `ErrorModel` now derives `epsTotal` and its negligibility internally from the five component boundaries, and the final theorem consumes that model directly on the active protocol path. |
| `S7.1` | Defs 11-14 (CCS) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean`, `ProtocolSection71Context.lean` | Norm-bounded CCS structure and evaluation relations. | - | S7.2, S7.3 | Done (Proof-Complete): `ProofSystem/ConstraintSystem/CCS.lean` formalizes the paper-facing Section 7.1 structure / CCS / CE / global-parameter objects with explicit statement and witness predicates; `ProtocolRelations.lean` owns the compact relation predicates and the single theorem-native Definition-14 owner `ProtocolSection71TheoremInstance` with two-way relation bridges; `ProtocolSection71Context.lean` packages that owner with its target context as the single-object owner consumed externally. |
| `S7.2` | Sec 7.3, Lemma 3 (Π_CCS) | `PiCCS.lean` | Π_CCS is a strong interactive reduction. | S4.4, S7.1 | S7.4 | Done (Witness-Level): the relation-level content of Lemma 3 is proved from compact `ceRelation` (`piCCSStrong_of_ce`) and from `ProtocolTargetAssumptions` plus a SumCheck transition witness (`piCCSStrong_of_assumptions`); the probabilistic strong-reduction statement (adversary, `⟨P*, V⟩`, extractor) is not formalized. |
| `S7.3` | Sec 7.4, Lemma 4 (Π_RLC) | `PiRLC.lean` | Π_RLC is a weak interactive reduction. | S7.2 | S7.4 | Done (Witness-Level): the relation-level content of Lemma 4 is proved from compact `ceRelation` (`piRLCWeak_of_ce`) and from `ProtocolTargetAssumptions` plus a transition witness (`piRLCWeak_of_assumptions`); the probabilistic weak-reduction statement is not formalized. |
| `S7.4` | Sec 7.5, Thm 7 (Π_DEC) | `PiDEC.lean` | Π_DEC is a reduction of knowledge. | S7.3 | S7.6 | Done (Witness-Level): the relation-level content of Theorem 7 is proved from the weak `Π_RLC` statement (`piDEC_of_weak`), from compact `ceRelation` (`piDEC_of_ce`), and from `ProtocolTargetAssumptions` plus a transition witness (`piDEC_of_assumptions`); the reduction-of-knowledge statement itself is not formalized. |
| `S7.5` | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean`, `ProtocolTarget.lean` | Side-conditions compose cleanly for protocol reduction. | S4.2, S4.5, S5.4, S5.6, S6.3, S6.4 | S7.6 | Done (Proof-Complete): theorem-native arithmetic bundles and protocol-target derivations are proved; `ProtocolTargetAssumptions` is the single protocol-side owner, with `ofPaperCarrierDiff` internalizing the proved Goldilocks invertibility bridge on the active route. |
| `S7.6` | Thm 1 (protocol theorem) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | End-to-end completeness + knowledge-soundness. | S5.2, S6.1, S6.2, S6.5, S7.2, S7.3, S7.4, S7.5 | Final claim | Done (Witness-Level): theorem shape and canonical final-assumption assembly are proved; knowledge-soundness is stated as witness-level composition plus advantage bounds for the carried failure events rather than as a quantification over PPT adversaries with an extractor. On the active `paperCarrier` path the final package derives Ajtai reduction data directly from the theorem-level MSIS hardness assumption, the narrowed Goldilocks Appendix B.2 route fixes the concrete paper lattice constants while leaving only message length explicit, the active `paperCarrier`-difference route consumes the proved Goldilocks invertibility theorem directly rather than an external invertibility boundary, and the active native-bar Goldilocks route derives the witness-level SumCheck and local Schwartz-Zippel packages internally from the accepted transition witness plus arithmetic obligations while reconstructing the internal MSIS boundary from the theorem-level hardness assumption. |

### Tracked Status and Exit Criteria

Completion policy reminder: every row below targets `Done (Proof-Complete)` as the terminal state.
Rows marked `Done (Boundary)` are intentionally intermediate; none remain in the tracked table below.

| ID | Status now | Missing now | Exit criteria |
|---|---|---|---|
| `S4.1` | Done (Proof-Complete). | None at the module level. | Base field/ring algebra and coefficient-map theorem surfaces remain available directly to S5.2/S5.5. |
| `S4.2` | Done (Proof-Complete). | None at the module level. | Norm/decomposition obligations remain discharged directly from theorem lemmas in downstream consumers. |
| `S4.3` | Done (Proof-Complete). | None at the module level. | MLE identity, folding bridge, chi/dot equivalence, and linearity remain theorem-native for S4.5/S5.5. |
| `S4.4` | Done (Proof-Complete). | Optional extension only: factor the constructive Definition-6 realization into a maximally generic reusable `SumCheck(T; Q)` library if broader reuse is desired. | Sum-check now provides a theorem-native Definition-6 witness surface plus constructive soundness/completeness directly in `SumCheck.lean`, and downstream consumers continue to consume it without extra boundary assumptions. |
| `S4.5` | Done (Proof-Complete). | None at the module level. | Full polynomial lemma set consumed by S7.5. |
| `S4.6` | Done (Proof-Complete). | None at the module level. | Appendix B.2 constants and core positivity/bound theorems remain available directly to S5.2/S6.3. |
| `S5.1` | Done (Proof-Complete). | None at the module level. | Definition-7 embedding package remains constructively closed and consumed directly by downstream theorem constructors. |
| `S5.2` | Done (Proof-Complete for native paper instance). | Optional extension only: prove basis criterion for additional concrete bar constructions. | Native Theorem-3 remains constructive while preserving downstream theorem interfaces. |
| `S5.3` | Done (Proof-Complete) for module-level closure. | None for the module-level theorem closure; optional extension is additional non-native bar design validation. | Keep bar-lift linearity theorem-native and reused directly by S5.4/S5.5 constructors. |
| `S5.4` | Done (Proof-Complete). | None at the module level; optional extension only is additional concrete bar classification beyond the theorem-native Theorem-3 surfaces already exported by `Thm3Core.lean`. | Full Theorem-4 proof remains available directly from `thm3CoreAssumption`, the finite basis-kernel witness, and the finite basis-kernel checker. |
| `S5.5` | Done (Proof-Complete). | None at the module level. | Remark-2 linkage and Definition-15 module-hom linearity remain constructively closed and feed Theorem 5 directly. |
| `S5.6` | Done (Proof-Complete). | None at the module level. | Theorem-5 remains proved constructively and feeds S7.5 without additional local boundaries. |
| `S6.1` | Done (Witness-Level). | The Definition-5/9/10 adversary-extractor formalization. | Theorem-6 composition remains available from `InteractiveReductionAssumptions` (protocol-target bundle plus SumCheck witness). |
| `S6.2` | Done (Proof-Complete). | None at the module level; optional extension only is additional carrier-parametric library packaging beyond the paper theorem inputs already exposed. | Theorem-2 binding reductions remain available directly from theorem-level MSIS hardness together with the paper carrier/strong-sampling inputs, and the active Goldilocks route reconstructs the Ajtai package internally from the theorem-level hardness assumption. |
| `S6.3` | Done (Proof-Complete). | Optional extension only: abstract the constructive Goldilocks proof beyond the paper's concrete floor `goldilocksPaperBInv = 383` if a wider bound-parametric library theorem is desired. | Theorem 8 remains proved constructively at the Appendix B.2 floor, with the narrower threshold-`5` and `paperCarrier`-difference routes derived as corollaries. |
| `S6.4` | Done (Proof-Complete) for module-level theorem surfaces. | Downstream protocol threading (S7.5) still needs full theorem-only closure. | Universal sampling expansion theorem wired into S7.5. |
| `S6.5` | Done (Proof-Complete). | None at the module level. | Error model derives total-error decomposition/negligibility internally and is consumed directly by S7.6. |
| `S7.1` | Done (Proof-Complete). | None at the module level; the remaining work is concrete protocol setup instantiation of one explicit `ProtocolSection71TheoremInstance`, which belongs upstream of Section 7.1 itself. | Definitions 11-14 remain formalized as proof-system objects plus the single protocol-side owner `ProtocolSection71TheoremInstance` / `ProtocolSection71Context`, bridging into compact `ccsRelation` / `ceRelation`. |
| `S7.2` | Done (Witness-Level). | The probabilistic strong-reduction layer of Lemma 3. | Π_CCS remains available from compact `ceRelation` and from `ProtocolTargetAssumptions` plus a SumCheck witness. |
| `S7.3` | Done (Witness-Level). | The probabilistic weak-reduction layer of Lemma 4. | Π_RLC remains available from compact `ceRelation` and from `ProtocolTargetAssumptions` plus a transition witness. |
| `S7.4` | Done (Witness-Level). | The reduction-of-knowledge layer of Theorem 7. | Π_DEC remains available from the weak `Π_RLC` statement, compact `ceRelation`, and `ProtocolTargetAssumptions` plus a transition witness. |
| `S7.5` | Done (Proof-Complete). | None at the module level. | `protocolTargetProp` remains derivable from `ProtocolTargetAssumptions`, whose `ofPaperCarrierDiff` constructor packages the active-route inputs without opaque local assumption bundles. |
| `S7.6` | Done (Witness-Level). | The adversary/extractor knowledge-soundness layer of Theorem 1. | End-to-end protocol theorem is consumed from the paper-faithful theorem-level assumptions only, with the witness-level SumCheck and local Schwartz-Zippel boundaries derived canonically in-module on the active native Goldilocks path and the internal MSIS boundary reconstructed from the theorem-level hardness assumption. |

## Math Breakdown (Current Status)

Source references:
- `docs/superneo-paper/04_4_Preliminaries.md`
- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/11_B_Concrete_parameters.md`
- `docs/superneo-paper/12_C_Additional_Background.md`
- `docs/superneo-paper/13_D_Deferred_theorems_and_proofs.md`

### Section 4: Preliminaries

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M1 | Definition 1 (field/ring/dimension) | `Field.lean` + `Dimensions.lean` | S4.1 | Done (Proof-Complete) |
| M2 | Definition 2 (`cf`, `cf⁻¹`, `ct`) | `CoeffMaps.lean` + `Ring.lean` | S4.1 | Done (Proof-Complete) |
| M3 | Ring arithmetic in `R_q` | `Ring.lean` | S4.1 | Done (Proof-Complete) |
| M4 | Definition 3 (centered `l_∞` norm) | `Norm.lean` | S4.2 | Done (Proof-Complete) |
| M5 | `split_b` decomposition | `Decomp.lean` | S4.2 | Done (Proof-Complete) |
| M6 | `eq` polynomial on Boolean hypercube | `EqPoly.lean` | S4.3 | Done (Proof-Complete) |
| M7 | MLE identity | `MLE.lean` | S4.3 | Done (Proof-Complete) |
| M8 | Definition 6 (sum-check protocol) | `SumCheck.lean` | S4.4 | Done (Proof-Complete) |
| M9 | Lemma 5 (Schwartz-Zippel) | `PolyLemmas.lean` | S4.5 | Done (Proof-Complete) |
| M10 | Lemma 6 (eq-lifting) | `PolyLemmas.lean` | S4.5 | Done (Proof-Complete) |
| M11 | Polynomial interpolation/evaluation | `Interp.lean` | S4.5 | Done (Proof-Complete) |
| M12 | Appendix B.2 concrete parameters | `Parameters.lean` + `Goldilocks.lean` | S4.6 | Done (Proof-Complete) |

### Section 5: Embedding Products with Evaluation Homomorphism

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M13 | Definition 7 (coefficient embedding) | `Embedding.lean` | S5.1 | Done (Proof-Complete) |
| M14 | Theorem 3 (inner-product transform) | `Thm3Core.lean` | S5.2 | Done (Proof-Complete for native paper instance) |
| M15 | Definition 8 (lifting transform) | `BarLift.lean` | S5.3 | Done (Proof-Complete) |
| M16 | Theorem 4 (matrix-vector product transform) | `MatrixTransform.lean` | S5.4 | Done (Proof-Complete) |
| M17 | Remark 2 (evaluation/ct linkage) | `EvalLink.lean` | S5.5 | Done (Proof-Complete) |
| M18 | Definition 15 (module homomorphisms) | `ModuleHom.lean` | S5.5 | Done (Proof-Complete) |
| M19 | Theorem 5 (evaluation homomorphism) | `EvalHom.lean` | S5.6 | Done (Proof-Complete) |

### Section 6: Security Model

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M20 | Definition 5 (interactive reductions) | `InteractiveReductions.lean` | S6.1 | Done (Witness-Level) |
| M21 | Definitions 9-10 (weak/strong reductions) | `InteractiveReductions.lean` | S6.1 | Done (Witness-Level) |
| M22 | Theorem 6 (strong-weak composition) | `InteractiveReductions.lean` | S6.1 | Done (Witness-Level) |
| M23 | Definition 4 (ring commitment scheme) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M24 | Definition 16 (MSIS) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M25 | Definition 18 (Ajtai commitment) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M26 | Theorem 2 (Ajtai properties) | `ProofSystem/LatticeReductions.lean`, `ProofSystem/LatticePaper.lean` | S6.2 | Done (Proof-Complete) |
| M27 | Theorem 8 (low-norm invertibility) | `InvertibilityAxioms.lean`, `InvertibilityGoldilocks.lean` | S6.3 | Done (Proof-Complete) |
| M28 | Definition 17 (strong sampling sets) | `SamplingSet.lean` | S6.4 | Done (Proof-Complete) |
| M29 | Theorem 9 (expansion factors) | `SamplingSet.lean` | S6.4 | Done (Proof-Complete) |

### Section 7: Folding Protocol

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M30 | Definition 11 (structure) | `ProofSystem/ConstraintSystem/CCS.lean` | S7.1 | Done (Proof-Complete) |
| M31 | Definition 12 (norm-bounded CCS) | `ProofSystem/ConstraintSystem/CCS.lean` | S7.1 | Done (Proof-Complete) |
| M32 | Definition 13 (CCS evaluation relation) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean` | S7.1 | Done (Proof-Complete) |
| M33 | Definition 14 (global parameters) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean` | S7.1 | Done (Proof-Complete) |
| M34 | Lemma 3 (Π_CCS is strong) | `PiCCS.lean` | S7.2 | Done (Witness-Level) |
| M35 | Lemma 4 (Π_RLC is weak) | `PiRLC.lean` | S7.3 | Done (Witness-Level) |
| M36 | Theorem 7 (Π_DEC reduction of knowledge) | `PiDEC.lean` | S7.4 | Done (Witness-Level) |
| M37 | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean` | S7.5 | Done (Proof-Complete) |
| M38 | Theorem 1 (full composition) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | S7.6 | Done (Witness-Level) |

### Infrastructure

| ID | Item | Lean target | Status |
|---|---|---|---|
| M39 | Theorem import wall (`lake exe check`) | `Main.lean` | Done (all checks pass) |

### Status Summary

| State | Count |
|---|---|
| Accepted (SuperNeo path) | 0 |
| Done (Boundary) | 0 |
| Done (Proof-Complete) | 31 (M1-M19, M23-M33, M37) |
| Done (Witness-Level) | 7 (M20, M21, M22, M34, M35, M36, M38) |
| In progress | 0 |
| Checks green | 1 (M39) |
| Good shell | 0 |
| Not started | 0 |
