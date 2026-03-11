# ProofSystem/LatticeReductions.spec.md

## Purpose

- **What it is**: MSIS-to-Ajtai binding reduction theorems that derive Ajtai commitment security from Module-SIS hardness.
- **Key property**: `ajtaiBoundaries_of_msis` bundles standard + relaxed binding consequences in probabilistic-bound form (negligible + advantage bounds) over the reduction carrier `C`.
- **Protocol role**: Provides the lattice-security reduction consumed by `ProtocolTheorem` to close the protocol's security proof under MSIS hardness.

## Target Formulas

- `‖subVec n w₁ w₂‖∞ < msisNormBound` = norm transfer from binding collision witnesses (standard: 2B, relaxed: 4TB per Theorem 2).
- `MSISHardnessAssumption → ∃ ε_MSIS, negl(ε_MSIS) ∧ Adv_MSIS ≤ ε_MSIS` = `msisAdvantageBound_of_hardness`.
- `MSISHardnessAssumption → MSISHardnessBoundary` = `MSISHardnessBoundary.ofHardness`.
- `AjtaiBindingAdvantageBound + IsNegligible → AjtaiBindingAssumption` = `no_ajtaiBindingCollision_of_advantageBound`.
- `AjtaiRelaxedBindingAdvantageBound + IsNegligible → AjtaiRelaxedBindingAssumption` = `no_ajtaiRelaxedBindingCollision_of_advantageBound`.
- `MSISHardnessAssumption → AjtaiBindingAssumption(params) ∧ AjtaiRelaxedBindingAssumption(params, C)` = `ajtaiBoundaries_of_msis`.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Theorem 2 (Ajtai Properties), lines 319-321: B-binding from MSIS_{m,2B}, (B,C)-relaxed binding from MSIS_{m,4TB} + strong sampling.
- Definition 4 (Ring commitment), lines 304-315: B-binding and (B,C)-relaxed binding collision definitions; Δ₁, Δ₂ ∈ (C-C) for relaxed.
- Definition 16 (MSIS), lines 743-744.
- Definition 17 (Strong sampling sets), lines 747-749: C ⊆ R_F with ‖a-b‖∞ < b_inv for distinct a,b ∈ C.
- Theorem 9 (Expansion factors), line 751: expansion factor of C ≤ 2·φ(η)·max_{ρ∈C} ‖ρ‖∞.
- Definition 18 (Ajtai commitment), lines 753-756.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Norm transfer | `bindingCollision_subWitness_norm_lt_msisNormBound` | Theorem-Target |
| Standard MSIS break | `msisBreakEvent_of_bindingCollision` | Theorem-Target |
| Relaxed MSIS break | `msisBreakEvent_of_relaxedBindingCollision` | Theorem-Target |
| Hardness unpacking | `msisAdvantageBound_of_hardness` | Theorem-Target |
| Hardness packaging | `MSISHardnessBoundary.ofHardness` | Theorem-Target |
| Standard binding | `no_ajtaiBindingCollision_of_advantageBound` | Theorem-Target |
| Ajtai from MSIS (standard) | `ajtaiBinding_of_msis` | Theorem-Target |
| Ajtai from MSIS (relaxed) | `ajtaiRelaxedBinding_of_msis` | Theorem-Target |
| Combined bundle | `ajtaiBoundaries_of_msis` | Theorem-Target |
| Reduction bundle | `MSISToAjtaiReductions` (structure) | Definitional |
| Bundle constructor | `MSISToAjtaiReductions.mk` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Norm | `bindingCollision_subWitness_norm_lt_msisNormBound` | `‖w₁ - w₂‖∞ < msisNormBound` | Theorem-Target |
| Extraction | `msisBreakEvent_of_bindingCollision` | Collision → MSIS break | Theorem-Target |
| Extraction | `msisBreakEvent_of_relaxedBindingCollision` | Relaxed collision → MSIS break | Theorem-Target |
| Security | `msisAdvantageBound_of_hardness` | Hardness package → explicit negligible + MSIS advantage bound package | Theorem-Target |
| Security | `MSISHardnessBoundary.ofHardness` | Hardness assumption → canonical explicit MSIS boundary package | Theorem-Target |
| Binding | `ajtaiBinding_of_msis` | MSIS → standard binding | Theorem-Target |
| Binding | `ajtaiRelaxedBinding_of_msis` | MSIS → relaxed binding | Theorem-Target |
| Bundle | `ajtaiBoundaries_of_msis` | MSIS → both bindings | Theorem-Target |
| Extractor algebra | `mulRq_sub_right`, `dotRq_subVec_linearity`, `subVec_ne_zero_of_ne`, `matVecMul_subVec` | Subtraction-linearity + collision witness non-triviality + matrix subtraction linearity | Theorem-Target |
| Reduction-law bundle | `LatticeReductionLaws` | Threads only `samplingCarrier : SamplingCarrier` and `strongSampling : strongSamplingExpansionProp C T`; extractor algebra is now derived theorem-natively from ring operations | Boundary |
| Constructor | `LatticeReductionLaws.ofCarrier` | Canonical constructor from explicit carrier + strong-sampling theorem | Theorem-Target |
| Constructor | `LatticeReductionLaws.ofPaperCarrier` | Canonical constructor specialized to `paperCarrier` | Theorem-Target |
| Constructor | `LatticeReductionLaws.ofPaperCarrierFromBounds` | Derives `paperCarrier` laws from subtraction/multiplication norm bundles | Theorem-Target |
| Derivation | `LatticeReductionLaws.paperStrongSampling_of_bounds` | Specializes `strongSamplingExpansionProp_of_paperCarrier` at `params.relaxedExpansion` | Theorem-Target |
| Constructor | `MSISToAjtaiReductions.ofLaws` | Canonical reduction-package constructor from an explicit `LatticeReductionLaws` instance | Theorem-Target |
| Constructor | `MSISToAjtaiReductions.ofPaperCarrier` | Specializes reduction package to `paperCarrier` from a strong-sampling theorem | Theorem-Target |
| Constructor | `MSISToAjtaiReductions.ofPaperCarrierFromBounds` | Specializes reduction package to `paperCarrier` from norm bundles (`hSub`, `hMul`) | Theorem-Target |
| Constructor | `MSISToAjtaiReductions.ofPaperCarrierFromThreeDLeAndMSISBoundary` | Specializes reduction package to the active `paperCarrier` path, deriving Ajtai error terms and bounds directly from an `MSISHardnessBoundary` plus `3*d ≤ params.relaxedExpansion` | Theorem-Target |
| Closed norm law | `normInfVec_subVec_le_derived` | \(\|\text{subVec}\,n\,v_1\,v_2\|_\infty \le \|v_1\|_\infty + \|v_2\|_\infty\) from `Field.centeredAbs_sub_le` + max-fold lemmas | Theorem-Target |

## Proof Obligations

- All theorem statements must hold without `sorry`.
- No module-level `axiom`s; carrier/algebra laws that are not definitional are threaded explicitly via `LatticeReductionLaws` and carried in `MSISToAjtaiReductions` only for the abstract carrier-parametric route.
- The specialized `paperCarrier` constructors reconstruct the internal MSIS boundary theorem-natively from the theorem-level MSIS hardness assumption, then derive Ajtai error/bound packaging and strong-sampling packaging from that boundary together with the proved sampling bounds. The further Goldilocks Appendix B.2 specialization lives in `specs/ProofSystem/LatticePaper.spec.md`.

## Assumption Ledger

Generic carrier-parametric boundary assumptions are explicit in `LatticeReductionLaws`:

- `strongSampling`: supply `strongSamplingExpansionProp` for the chosen sampling carrier.
  Closure target: instantiate via `LatticeReductionLaws.ofCarrier`, `.ofPaperCarrier`, or `.ofPaperCarrierFromBounds` with a concrete sampling set proof.

Relaxed-collision carrier membership is explicit in collision witnesses:
- `RelaxedBindingCollision params C` carries `inDiff1 : samplingDiffSet C delta1`
- `RelaxedBindingCollision params C` carries `inDiff2 : samplingDiffSet C delta2`

Derived internally (not a boundary field):
- `normInfVec_subVec_le_derived`: vector subtraction norm triangle, proved in-module.
- `smulVec_comm_derived`: commutation of nested ring-vector scalar actions, proved from ring multiplication.
- `matVecMul_smulVec_derived`: matrix/scalar compatibility, proved from `dotRq` linearity and ring multiplication.
- `normInfVec_smulVec_le_of_diff`: vector-level bound
  \(\|\delta \cdot v\|_\infty \le 4T\cdot B\) proved from `strongSampling` + `normInfVec` max aggregation.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.ProofSystem.Lattice` (uses structures, vector operations, and boundary package typeclass).
- **Consumers**:
  - `SuperNeo.ProtocolTheorem`: imports `LatticeReductions` for `ajtaiBoundaries_of_msis` in the final theorem.
  - `SuperNeo.ProofSystem.Protocol`: uses reduction bundle in the proof-system capstone.

## Design Notes

- The module separates the abstract carrier-parametric route from the specialized `paperCarrier` / Goldilocks route.
- `LatticeReductionLaws` records the explicit carrier-side laws needed for the abstract route.
- The specialized constructors expose the concrete paper-facing carrier route without reintroducing those abstract laws at the final-theorem surface.
- The concrete Goldilocks Appendix B.2 parameter family is documented separately in `specs/ProofSystem/LatticePaper.spec.md`.

## Quality Expectations

- All reduction theorems proved without sorry.
- Clear norm-transfer chain from collision to MSIS break.

## Acceptance Criteria

- `lake build` succeeds.
- All theorems type-check without sorry.
- Spec documents Theorem 2 with line ranges.

## Out of Scope

- Concrete MSIS advantage computation (abstracted via `MSISAdvantageBound`).
- Probabilistic game semantics for challenger/adversary.
