# SuperNeo Lean Cross-Check (Standalone)

This folder is intentionally outside `crates/` and independent from the Rust workspace.

It provides the theorem-facing Lean implementation of core SuperNeo/Neo math surfaces.
Lean is the mathematical source of truth.

Operationally:
- Lean theorem/definition surfaces are authoritative.
- Golden vectors from this formalization are used to validate Rust implementations.
- Rust-generated vectors (`neo-math`) are used as executable regression evidence.
- Target direction: run corresponding Lean and Rust computations in parallel and
  require value equality at each compared surface.

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

The module structure mirrors the paper's four main sections.

### Barrel files (top-level re-exports)

| Barrel | Paper section | Re-exports |
|---|---|---|
| `SuperNeo/Primitives.lean` | Section 4 (Preliminaries) | Goldilocks, Field, Dimensions, Ring, CoeffMaps, Norm, Decomp, EqPoly, MLE, SumCheck, PolyLemmas, Interp, Parameters |
| `SuperNeo/EmbeddingTheory.lean` | Section 5 (Embedding + Eval Hom) | Embedding, Thm3Core, BarLift, MatrixTransform, EvalLink, ModuleHom, EvalHom |
| `SuperNeo/SecurityModel.lean` | Section 6 + Appendix C security | InteractiveReductions, ProofSystem/{Types,Security,Negligible,Lattice,LatticePaper,LatticeReductions}, InvertibilityAxioms, InvertibilityGoldilocks, SamplingSet |
| `SuperNeo/FoldingProtocol.lean` | Section 7 (Folding scheme) | ProofSystem/{ConstraintSystem,SumCheck,Folding}, ProtocolRelations, ProtocolSection71Data, ProtocolSection71Context, PiCCS, PiRLC, PiDEC, ArithmeticBundle, ArithmeticObligations, ProtocolTarget, ProtocolTargetData, ProtocolMathTarget, ProtocolTheorem, ProofSystem/Protocol |

### Section 4: Primitives (Definitions 1-6, polynomial tools)

- `SuperNeo/Goldilocks.lean`: Goldilocks field constants
- `SuperNeo/Field.lean`: Goldilocks modular arithmetic implementation
- `SuperNeo/Dimensions.lean`: concrete `η`, `d`, and shape helpers
- `SuperNeo/Ring.lean`: ring multiplication/reduction, `ct`, bar-block mat-vec
- `SuperNeo/CoeffMaps.lean`: `cf` / `cf⁻¹` map definitions and round-trips
- `SuperNeo/Norm.lean`: centered-representative norms
- `SuperNeo/Decomp.lean`: balanced base-`b` decomposition (`split_b`) helpers
- `SuperNeo/EqPoly.lean`: `eq` polynomial helpers
- `SuperNeo/MLE.lean`: multilinear-extension identities (`r_hat`, folding)
- `SuperNeo/SumCheck.lean`: standalone Definition-6 sum-check scaffold plus paper-facing verifier view and constructive theorem witness surface
- `SuperNeo/PolyLemmas.lean`: reusable polynomial helpers for Lemma 5/6
- `SuperNeo/Interp.lean`: polynomial eval + interpolation
- `SuperNeo/Parameters.lean`: Appendix B.2 concrete parameter constants

### Section 5: Embedding Theory (Definitions 7-8, Theorems 3-5)

- `SuperNeo/Embedding.lean`: Definition 7 element/vector/matrix embeddings
- `SuperNeo/Thm3Core.lean`: Theorem 3 core proposition + dimensional preconditions
- `SuperNeo/BarLift.lean`: Definition 8 blockwise lifting transform
- `SuperNeo/MatrixTransform.lean`: Theorem 4 computational transform identity
- `SuperNeo/EvalLink.lean`: Remark 2 coefficientwise evaluation linkage
- `SuperNeo/ModuleHom.lean`: Definition 15 module-hom interfaces + linearity
- `SuperNeo/EvalHom.lean`: Theorem 5 computational evaluation homomorphism

### Section 6: Security Model (Definitions 9-10, 16-18, Theorems 2, 6, 8-9)

- `SuperNeo/InteractiveReductions.lean`: Definitions 9-10 weak/strong reductions, Theorem 6 composition
- `SuperNeo/ProofSystem/Types.lean`: proof-system facade types (Context, Claim, Witness)
- `SuperNeo/ProofSystem/Security.lean`: probability/error model
- `SuperNeo/ProofSystem/Negligible.lean`: `ErrorFn`, `IsNegligible`
- `SuperNeo/ProofSystem/Lattice.lean`: Definition 16 (MSIS), Definition 18 (Ajtai), Theorem 2
- `SuperNeo/ProofSystem/LatticePaper.lean`: Goldilocks Appendix B.2 lattice parameter family + active paper-carrier specialization
- `SuperNeo/ProofSystem/LatticeReductions.lean`: MSIS-to-Ajtai security reductions
- `SuperNeo/InvertibilityAxioms.lean`: Theorem 8 assumption boundary and concrete precondition checks
- `SuperNeo/InvertibilityGoldilocks.lean`: constructive Goldilocks Theorem 8 proof at the paper floor `b_inv = 383`
- `SuperNeo/SamplingSet.lean`: Definition 17/Theorem 9 sampling-set and expansion checks

### Section 7: Folding Protocol (Definitions 11-14, Lemmas 3-4, Theorem 7)

- `SuperNeo/ProofSystem/ConstraintSystem/CCS.lean`: Definitions 11-13 (CCS structure)
- `SuperNeo/ProofSystem/SumCheck/`: proof-system-level sum-check facade
- `SuperNeo/ProtocolRelations.lean`: Section 7.1 relation predicates
- `SuperNeo/ProtocolSection71Data.lean`: explicit protocol-side Definition-14 data owner
- `SuperNeo/ProtocolSection71Context.lean`: single-object compact theorem-native Section 7.1 owner
- `SuperNeo/ProtocolTargetData.lean`: explicit protocol-side Section 7.5 target-data owner
- `SuperNeo/PiCCS.lean`: Section 7.3 Π_CCS, Lemma 3
- `SuperNeo/PiRLC.lean`: Section 7.4 Π_RLC, Lemma 4
- `SuperNeo/PiDEC.lean`: Section 7.5 Π_DEC, Theorem 7
- `SuperNeo/ArithmeticBundle.lean`: bundled arithmetic prerequisites
- `SuperNeo/ArithmeticObligations.lean`: arithmetic side-conditions for protocol reduction
- `SuperNeo/ProtocolTarget.lean`: protocol-target bridge (Thm 3 + obligations)
- `SuperNeo/ProtocolMathTarget.lean`: protocol math-target from arithmetic bundle
- `SuperNeo/ProtocolTheorem.lean`: final theorem shape (completeness + knowledge-soundness)
- `SuperNeo/ProofSystem/Protocol.lean`: proof-system entrypoint (final theorem wiring)
- `SuperNeo/ProofSystem/Folding/`: proof-system folding wrappers

### Infrastructure

- `SuperNeo/ProtocolReduction.lean`: medium-term theorem skeletons
- `SuperNeo/Checks.lean`: cross-check predicates against generated vectors
- `SuperNeo/Regression.lean`: regression harness
- `SuperNeo/Generated/Vectors.lean`: Rust-generated constants (bar matrix + cases)
- `rust-vectors/`: standalone Rust generator crate

## Regenerate vectors from Rust

```bash
cargo run --manifest-path formal/superneo-lean/rust-vectors/Cargo.toml
```

## Run Lean checks

```bash
cd formal/superneo-lean
lake build
lake exe check
```

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

`lake exe check` reports 21 atomic checks plus one aggregate gate.
`true` means the corresponding executable predicate passed on all configured inputs.
This is stronger than unit smoke tests, but weaker than full universal theorem proofs.

| Output flag | `true` means (exactly what passed) | Evidence type | Remaining gap to a full SuperNeo proof |
|---|---|---|---|
| `superneo_cases` | For every generated `(a,b)`: `ct(mulRqPhi(superneoBarBlock(bar,a), b))`, `dot(a,b)`, and expected values all agree. | Rust-generated vectors + Lean recomputation | Native Theorem-3 closure is now proved constructively; optional extension only is broader non-native `bar` generalization. |
| `ring_mul_cases` | `mulRqPhi a b` matches expected coefficient vectors for all generated multiplication cases. | Rust-generated vectors | Prove quotient-ring multiplication semantics universally. |
| `norm_cases` | `normInfCoeffs` equals expected norms on all generated norm cases. | Rust-generated vectors | Prove general norm properties/bounds used in later theorems. |
| `split_cases` | `splitBalancedVec` digits match expected, recomposition equals expected and original input, and per-digit bounds hold. | Rust-generated vectors + invariant check | The decomposition round-trip/bound theorems are now proved constructively in `Decomp.lean`; this executable gate remains regression evidence. |
| `eq_cases` | `eqPoly x y` matches expected; Boolean points also satisfy indicator behavior check. | Rust-generated vectors + Boolean invariant | Prove full hypercube-indicator theorem. |
| `mle_cases` | Inner-product MLE and folding MLE both match expected and match each other (`mleIdentity`). | Rust-generated vectors + identity check | The MLE identity, folding equivalence, chi/dot equivalence, and linearity packages are now proved in `MLE.lean`; this executable gate remains regression evidence. |
| `embedding_vec_cases` | `embedVec` matches expected blocks and `unembedVec (embedVec v) = v`. | Rust-generated vectors + round-trip invariant | The quantified vector embedding theorems are now proved in `Embedding.lean`; this executable gate remains regression evidence. |
| `embedding_matrix_cases` | `embedMatrix` matches expected blocks and `unembedMatrix (embedMatrix M) = M`. | Rust-generated vectors + round-trip invariant | The quantified matrix embedding theorems are now proved in `Embedding.lean`; this executable gate remains regression evidence. |
| `bar_lift_vec_cases` | Bar-lift outputs for `v`, `w`, `v+w`, `s*v` match expected and satisfy add/scale linearity checks. | Rust-generated vectors + linearity invariant | The vector-level Definition-8 closure is now proved constructively; this executable gate remains regression evidence. |
| `bar_lift_matrix_cases` | `barLiftMatrix` matches expected lifted matrices on all generated cases. | Rust-generated vectors | The matrix-level Definition-8 closure is now proved constructively; this executable gate remains regression evidence. |
| `matrix_transform_cases` | `matrixVecDirect M z`, `matrixVecCtBar bar M z`, and expected vectors all agree; identity predicate also holds. | Rust-generated vectors + identity check | Theorem-4 closure is now proved constructively in `MatrixTransform.lean`, including direct theorem-native entrypoints from the finite basis-kernel Theorem-3 witness/checker; this executable gate remains regression evidence. |
| `eval_link_cases` | Evaluation-link computations (`evalRingVec`, `ct`, expected outputs) agree and `evalLinkIdentity`/`evalLinkForMatrix` checks pass. | Rust-generated vectors + identity checks | The eval-link theorem layer is now proved; any remaining broader gaps are upstream open preliminaries such as coefficient-map and MLE generalization, not `EvalLink.lean` itself. |
| `eval_hom_cases` | Evaluation homomorphism outputs (`Y1`, `Y2`, linear combo, direct combo) all match expected and each other; `evalHom2` holds. | Rust-generated vectors + homomorphism invariant | Theorem-5 closure is now proved constructively in `EvalHom.lean`; this executable gate remains regression evidence. |
| `module_hom_cases` | `moduleHomSanity` passes add/scale preservation for representative concrete homomorphisms. | Fixed sanity witnesses (not generated) | The abstract module-hom theorem/check bridges are now proved in `ModuleHom.lean`; this executable gate remains regression evidence. |
| `invertibility_cases` | Concrete Goldilocks arithmetic side-conditions for the low-norm invertibility interface are satisfied. | Deterministic constant checks | The active Goldilocks `paperCarrier`-difference theorem path is closed, and the shape-aware Goldilocks low-norm theorem at threshold `5` is proved; remaining optional work is the broader standalone low-norm theorem beyond that narrow route. |
| `sampling_cases` | Strong-sampling predicate, max norm, bound, empirical expansion, and `empirical <= bound` all match expected/hold. | Rust-generated vectors + bound check | Theorem-9 closure is now proved in `SamplingSet.lean`; this executable gate remains regression evidence. |
| `eq_lift_cases` | `eqLiftFromTable` matches expected sums; Boolean-point behavior matches expected values when applicable. | Rust-generated vectors + Boolean-point check | The quantified eq-lifting theorems are now proved in `PolyLemmas.lean`; this executable gate remains regression evidence. |
| `poly_lemma_cases` | `polyLemmaSanity` passes (`eqLiftAllBoolean` on a sample table + SZ interface condition). | Fixed sanity witnesses | Theorem-native Schwartz-Zippel/eq-lift closure is now proved in `PolyLemmas.lean`; this executable gate remains regression evidence. |
| `coeff_map_cases` | Coefficient-map round-trip checks pass on superneo/ring-generated data; additional sanity predicates pass. | Mixed: generated data + sanity predicates | Coefficient-map round-trip, `ct` compatibility, and shape-preservation theorem surfaces are now proved in `CoeffMaps.lean`; this executable gate remains regression evidence. |
| `parameter_cases` | Shape sanity, concrete parameter sanity, and norm sanity predicates all hold. | Deterministic constant/invariant checks | Appendix B.2 constants and core positivity/bound theorems are now proved in `Parameters.lean`; this executable gate remains regression evidence. |
| `interp_cases` | Lightweight interpolation regression slot reports green; the actual interpolation correctness/uniqueness claim now lives in the quantified proofs in `Interp.lean`. | Proof layer in `Interp.lean` + lightweight executable gate | Keep theorem closure and, if desired later, make the Goldilocks/ZMod interpolation path executable for vector replay. |
| `all_checks` | Logical conjunction of every check above is `true`. | Aggregate gate | No new math content; only reports that all current executable checks passed. |

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
| `In progress` | Some theorem surfaces exist, but proof obligations remain open. | No |
| `Good shell` | Composition skeleton exists; full derivation is not complete. | No |
| `Done (Proof-Complete)` | The paper-faithful theorem is closed at the module itself: no local placeholder bundle remains except the intended theorem-level assumptions (for example, the paper's cryptographic hardness assumption). | Yes |

At the current project state, every tracked milestone row is already `Done (Proof-Complete)`.
These label definitions are retained because they were used during closure and still
matter for interpreting older discussions, intermediate branches, and future extensions.

### How To Read These Labels

- `Accepted (SuperNeo path)` means "good enough for the actual SuperNeo theorem route, not yet maximally generalized."
  Example: this label is reserved for cases where the active theorem route is closed while a broader reusable generalization is intentionally deferred.

- `Done (Boundary)` means "this module works if you hand it the right upstream theorem bundle, but that upstream bundle is not fully eliminated yet."
  Example: this is the status to use when a local theorem is closed only after assuming a still-open upstream provider bundle, rather than from the intended paper theorem inputs directly.

- `Done (Proof-Complete)` means "the module's own paper claim is proved directly, with only the intended theorem-level assumptions left explicit."
  Example: `S7.6` is proof-complete on the active route because the final theorem now derives its local SumCheck, Schwartz-Zippel, and internal MSIS packages in-module and leaves only the theorem-level MSIS hardness assumption explicit.

## Current Practical Reading

- One-sentence status: the tracked SuperNeo formalization is now paper-faithful and proof-complete.
- The active native Goldilocks / `paperCarrier`-difference route is now proof-complete end-to-end through `S7.6`.
- All tracked milestone rows are now `Done (Proof-Complete)`.
- Any remaining work is optional library generalization or cleanup outside the tracked paper-faithful milestone set.

## Reader Guide

If you only need the conclusion, read in this order:

1. This README:
   - `Current Practical Reading`
   - `Milestone Table`
   - `Status Summary`
2. The capstone theorem docs:
   - `specs/ProtocolTheorem.spec.md`
   - `SuperNeo/ProtocolTheoremInterface.lean`
   - `SuperNeo/ProtocolTheorem.lean`
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
| `S6.1` | Defs 5, 9-10, Thm 6 | `InteractiveReductions.lean` | Weak/strong reductions compose correctly. | - | S7.6 | Done (Proof-Complete): strong/weak composition theorems are proved directly from one explicit `ProtocolTargetData` owner plus a SumCheck witness, and also from the narrower Section 7.1 theorem-native owners (`ProtocolSection71Data`, `ProtocolSection71Context`, `ProtocolSection71TheoremInstance`, `ProtocolSection71Setup`, `ProtocolSection71Provider`, and realized `CE.Holds`); `InteractiveReductionAssumptions` remains only as a compatibility bundle. |
| `S6.2` | Defs 4, 16, 18, Thm 2 | `ProofSystem/Lattice.lean`, `ProofSystem/LatticeReductions.lean`, `ProofSystem/LatticePaper.lean` | Ajtai commitment properties, MSIS hardness, binding reductions. | - | S6.3, S7.6 | Done (Proof-Complete): Defs 4/16/18 and Theorem 2 are proved constructively; the generic carrier route leaves only the paper's intended `samplingCarrier` + strong-sampling inputs explicit, and the active Goldilocks `paperCarrier` route reconstructs the full Ajtai reduction package directly from theorem-level MSIS hardness. |
| `S6.3` | Thm 8 (invertibility) | `InvertibilityAxioms.lean`, `InvertibilityGoldilocks.lean` | Low-norm invertibility preconditions and interface. | S4.2, S4.6, S6.2 | S6.4, S7.5 | Done (Proof-Complete): the theorem surface is shape-aware (`hasRingDegreeShape a → 0 < ‖a‖∞ < B → invertibleRq a`), `InvertibilityGoldilocks.lean` proves the concrete Goldilocks theorem at the paper floor `goldilocksPaperBInv = 383`, the narrower threshold-`5` route is a corollary, and the active `paperCarrier`-difference route is derived from that constructive theorem in-repo. |
| `S6.4` | Def 17 + Thm 9 (sampling) | `SamplingSet.lean` | Strong-sampling + expansion-factor interface. | S4.2, S6.3 | S7.5 | Done (Proof-Complete) for module-level contract surfaces (`samplingDiffSet`, `strongSamplingExpansionProp`, and associated theorem wrappers). |
| `S6.5` | Error/negligible model | `ProofSystem/{Types,Security,Negligible}.lean` | `ProbModel`, `ErrorModel`, `IsNegligible`. | - | S7.6 | Done (Proof-Complete): the canonical `ErrorModel` now derives `epsTotal` and its negligibility internally from the five component boundaries, and the final theorem consumes that model directly on the active protocol path. |
| `S7.1` | Defs 11-14 (CCS) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean`, `ProtocolSection71Data.lean` | Norm-bounded CCS structure and evaluation relations. | - | S7.2, S7.3 | Done (Proof-Complete): `ProofSystem/ConstraintSystem/CCS.lean` formalizes the paper-facing Section 7.1 structure / CCS / CE / global-parameter objects with explicit statement and witness predicates, `ProtocolRelations.lean` gives the compact specialization layer (`ProtocolSection71Objects` / `ProtocolSection71Realization` / `ProtocolSection71Specialization` / `ProtocolSection71Setup` / `ProtocolSection71Provider` / `ProtocolSection71TheoremInstance`), and `ProtocolSection71Data.lean` now exposes the explicit protocol-side Definition-14 data owner that canonically builds the specialized theorem instance and the single-object compact context owner. |
| `S7.2` | Sec 7.3, Lemma 3 (Π_CCS) | `PiCCS.lean`, `ProofSystem/Folding/PiCCS.lean` | Π_CCS is a strong interactive reduction. | S4.4, S7.1 | S7.4 | Done (Proof-Complete): Lemma 3 is proved directly from compact `ceRelation`, from the Section 7.1 theorem-native owners (`ProtocolSection71Data`, `ProtocolSection71Context`, `ProtocolSection71TheoremInstance`, `ProtocolSection71Setup`, `ProtocolSection71Provider`, and realized `CE.Holds`), from `ProtocolTargetData` plus a SumCheck witness, and from the active paper-facing/native routes; `PiCCSAssumptions` remains only as a compatibility bundle. |
| `S7.3` | Sec 7.4, Lemma 4 (Π_RLC) | `PiRLC.lean`, `ProofSystem/Folding/PiRLC.lean` | Π_RLC is a weak interactive reduction. | S7.2 | S7.4 | Done (Proof-Complete): Lemma 4 is proved directly from compact `ceRelation`, from the Section 7.1 theorem-native owners, from `ProtocolTargetData` plus a transition witness, and from the active paper-facing/native routes; `PiRLCAssumptions` remains only as a compatibility bundle. |
| `S7.4` | Sec 7.5, Thm 7 (Π_DEC) | `PiDEC.lean`, `ProofSystem/Folding/PiDEC.lean` | Π_DEC is a reduction of knowledge. | S7.3 | S7.6 | Done (Proof-Complete): Theorem 7 is proved directly from the weak `Π_RLC` statement, from compact `ceRelation`, from the Section 7.1 theorem-native owners, from `ProtocolTargetData` plus a transition witness, and from the active paper-facing/native routes; `PiDECAssumptions` remains only as a compatibility bundle. |
| `S7.5` | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean`, `ProtocolTarget.lean`, `ProtocolTargetData.lean`, `ProtocolMathTarget.lean` | Side-conditions compose cleanly for protocol reduction. | S4.2, S4.5, S5.4, S5.6, S6.3, S6.4 | S7.6 | Done (Proof-Complete): theorem-native arithmetic bundles and protocol-target derivations are proved, and the explicit protocol-side owner `ProtocolTargetData` canonically derives `protocolTargetProp`; the legacy `ProtocolTargetAssumptions` surface remains only as a compatibility bundle. |
| `S7.6` | Thm 1 (protocol theorem) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | End-to-end completeness + knowledge-soundness. | S5.2, S6.1, S6.2, S6.5, S7.2, S7.3, S7.4, S7.5 | Final claim | Done (Proof-Complete): theorem shape and canonical final-assumption assembly are proved; on the active `paperCarrier` path the final package derives Ajtai reduction data directly from the theorem-level MSIS hardness assumption, the narrowed Goldilocks Appendix B.2 route fixes the concrete paper lattice constants while leaving only message length explicit, the active `paperCarrier`-difference route consumes the proved Goldilocks invertibility theorem directly rather than an external invertibility boundary, and the active native-bar Goldilocks route derives the witness-level SumCheck and local Schwartz-Zippel packages internally from the accepted transition witness plus arithmetic obligations while reconstructing the internal MSIS boundary from the theorem-level hardness assumption. |

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
| `S6.1` | Done (Proof-Complete). | None at the module level; optional extension only is additional concrete protocol-route instantiations. | Theorem-6 composition remains available directly from `ProtocolTargetData` plus a SumCheck witness and from the narrower Section 7.1 theorem-native owners; `InteractiveReductionAssumptions` is compatibility-only. |
| `S6.2` | Done (Proof-Complete). | None at the module level; optional extension only is additional carrier-parametric library packaging beyond the paper theorem inputs already exposed. | Theorem-2 binding reductions remain available directly from theorem-level MSIS hardness together with the paper carrier/strong-sampling inputs, and the active Goldilocks route reconstructs the Ajtai package internally from the theorem-level hardness assumption. |
| `S6.3` | Done (Proof-Complete). | Optional extension only: abstract the constructive Goldilocks proof beyond the paper's concrete floor `goldilocksPaperBInv = 383` if a wider bound-parametric library theorem is desired. | Theorem 8 remains proved constructively at the Appendix B.2 floor, with the narrower threshold-`5` and `paperCarrier`-difference routes derived as corollaries. |
| `S6.4` | Done (Proof-Complete) for module-level theorem surfaces. | Downstream protocol threading (S7.5) still needs full theorem-only closure. | Universal sampling expansion theorem wired into S7.5. |
| `S6.5` | Done (Proof-Complete). | None at the module level. | Error model derives total-error decomposition/negligibility internally and is consumed directly by S7.6. |
| `S7.1` | Done (Proof-Complete). | None at the module level; the remaining work is concrete protocol setup instantiation of one explicit `ProtocolSection71Data` package, which belongs upstream of Section 7.1 itself. | Definitions 11-14 remain formalized as proof-system objects and protocol-side theorem-native data owners, with canonical bridges into `ProtocolSection71TheoremInstance` / `ProtocolSection71Context` and then into compact `ccsRelation` / `ceRelation`. |
| `S7.2` | Done (Proof-Complete). | None at the module level; optional extension only is additional concrete route instantiation beyond the theorem-native surfaces already proved. | Π_CCS remains available directly from compact `ceRelation`, the Section 7.1 theorem-native owners, `ProtocolTargetData` plus a SumCheck witness, and the active route constructors, with `PiCCSAssumptions` retained only as compatibility. |
| `S7.3` | Done (Proof-Complete). | None at the module level; optional extension only is additional concrete route instantiation beyond the theorem-native surfaces already proved. | Π_RLC remains available directly from compact `ceRelation`, the Section 7.1 theorem-native owners, `ProtocolTargetData` plus a transition witness, and the active route constructors, with `PiRLCAssumptions` retained only as compatibility. |
| `S7.4` | Done (Proof-Complete). | None at the module level; optional extension only is additional concrete route instantiation beyond the theorem-native surfaces already proved. | Π_DEC remains available directly from the weak `Π_RLC` statement, compact `ceRelation`, the Section 7.1 theorem-native owners, `ProtocolTargetData` plus a transition witness, and the active route constructors, with `PiDECAssumptions` retained only as compatibility. |
| `S7.5` | Done (Proof-Complete). | None at the module level; optional upstream generalization now sits outside this row rather than inside the arithmetic/protocol-target assembly itself. | `protocolTargetProp` remains derivable directly from `ProtocolTargetData` or equivalent active-route inputs without opaque local assumption bundles. |
| `S7.6` | Done (Proof-Complete). | None on the active theorem route; optional extension only is broader genericization beyond the concrete Goldilocks/paper path. | End-to-end protocol theorem is consumed from the paper-faithful theorem-level assumptions only, with the witness-level SumCheck and local Schwartz-Zippel boundaries derived canonically in-module on the active native Goldilocks path and the internal MSIS boundary reconstructed from the theorem-level hardness assumption. |

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
| M20 | Definition 5 (interactive reductions) | `InteractiveReductions.lean` | S6.1 | Done (Proof-Complete) |
| M21 | Definitions 9-10 (weak/strong reductions) | `InteractiveReductions.lean` | S6.1 | Done (Proof-Complete) |
| M22 | Theorem 6 (strong-weak composition) | `InteractiveReductions.lean` | S6.1 | Done (Proof-Complete) |
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
| M32 | Definition 13 (CCS evaluation relation) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean`, `ProtocolSection71Data.lean` | S7.1 | Done (Proof-Complete) |
| M33 | Definition 14 (global parameters) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean`, `ProtocolSection71Data.lean` | S7.1 | Done (Proof-Complete) |
| M34 | Lemma 3 (Π_CCS is strong) | `PiCCS.lean` | S7.2 | Done (Proof-Complete) |
| M35 | Lemma 4 (Π_RLC is weak) | `PiRLC.lean` | S7.3 | Done (Proof-Complete) |
| M36 | Theorem 7 (Π_DEC reduction of knowledge) | `PiDEC.lean` | S7.4 | Done (Proof-Complete) |
| M37 | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean` | S7.5 | Done (Proof-Complete) |
| M38 | Theorem 1 (full composition) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | S7.6 | Done (Proof-Complete) |

### Infrastructure

| ID | Item | Lean target | Status |
|---|---|---|---|
| M39 | Executable cross-check harness | `Main.lean` + `Checks.lean` | Done (all checks pass) |

### Status Summary

| State | Count |
|---|---|
| Accepted (SuperNeo path) | 0 |
| Done (Boundary) | 0 |
| Done (Proof-Complete) | 38 (M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20, M21, M22, M23, M24, M25, M26, M27, M28, M29, M30, M31, M32, M33, M34, M35, M36, M37, M38) |
| In progress | 0 |
| Checks green | 1 (M39) |
| Good shell | 0 |
| Not started | 0 |
