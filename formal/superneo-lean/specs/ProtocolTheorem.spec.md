# ProtocolTheorem Spec

## Purpose

- **What it is**: The capstone final-theorem module for the SuperNeo formalization.
- **Key property**: `FinalTheoremAssumptions ctx -> FinalTheoremShape ctx hA`.
- **Protocol role**: Combines the protocol reductions from Sections 5–7 with the lattice-security/error-accounting layer from Appendix B/C.

## Target Formulas

- `FinalTheoremShape ctx hA` packages:
  - `FinalCompletenessStatement ctx hA`
  - `FinalKnowledgeSoundnessStatement ctx hA`
- `FinalKnowledgeSoundnessStatement ctx hA` contains:
  - `strongCompositionStatement ctx`
  - `SchwartzZippelAdvantageBound ctx hA.errorModel.epsSchwartzZippel`
  - `SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound (sumcheckInstanceOfContext ctx) hA.reduction.sumcheckTransitionWitness.transcript hA.errorModel.epsSumcheck`
  - `SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS`
  - `SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding`
  - `SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding`
  - total-error decomposition
  - `SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal`
- `FinalTheoremAssumptions.sumcheckAdvantageBound` is the canonical witness-level SumCheck failure bound aligned to `epsSumcheck`.
- `FinalErrorPackage.ofComponentBoundaries` assembles a canonical shared `ErrorModel` from explicit SumCheck/SZ/MSIS/Ajtai component boundaries.
- `FinalErrorPackage.ofAlignedComponents` assembles the same package from explicit components plus one alignment witness.
- `FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe` specializes to the proved `paperCarrier` route and derives the MSIS-to-Ajtai reduction data from the MSIS boundary plus `3 * d <= params.relaxedExpansion`.
- `FinalErrorPackage.ofGoldilocksPaperCarrier` specializes further to the Goldilocks Appendix B.2 parameter family, leaving only `messageLength` explicit.
- `FinalTheoremAssumptions.ofBoundaryPackages` is the canonical constructor from a reduction bundle and one canonical final error package.
- `FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages`, `.ofAlignedPaperCarrierDiffBoundaryPackages`, and `.ofAlignedPaperCarrierLowNormBoundaryPackages` are the aligned constructors for the proved `paperCarrier`, paper-facing difference, and stronger low-norm routes.
- `FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages`, `.ofGoldilocksPaperCarrierDerivedSumcheck`, `.ofGoldilocksPaperCarrierDiffBoundaryPackages`, `.ofGoldilocksNativePaperCarrierDiffBoundaryPackages`, and `.ofGoldilocksPaperCarrierLowNormBoundaryPackages` are the Goldilocks Appendix B.2 constructors.
- On the native-bar Goldilocks route, the constructor derives the witness-level SumCheck boundary and the local Schwartz-Zippel boundary internally from the carried `SumCheckTransitionWitness` and arithmetic obligations.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.6: implied final protocol theorem
- Section 7: folding scheme and composition path
- Appendix B/C/D: parameter choices, MSIS, Ajtai binding, deferred proofs

## Module Mapping

- Implementation: `SuperNeo.ProtocolTheorem`
- Interface: `SuperNeo.ProtocolTheoremInterface`

## Contract Surface

| Group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Schwartz-Zippel | `schwartzZippelFailureEvent ctx`, `SchwartzZippelAdvantage ctx`, `SchwartzZippelAdvantageBound ctx`, `SchwartzZippelBoundary ctx` | `ctx : ProtocolTargetContext` | Context-local theorem-facing Schwartz-Zippel surfaces | Definitional | `FinalErrorPackage`, `FinalTheoremAssumptions` |
| Lattice | `LatticeParams`, `msisHardnessAssumption`, `ajtaiBindingAssumption`, `ajtaiRelaxedBindingAssumption params C` | None | Typed theorem-level MSIS/Ajtai security-assumption surfaces | Definitional | `FinalErrorPackage`, `FinalTheoremAssumptions` |
| Local replay surfaces | `SumcheckPrefixLundBoundary ctx`, `GoldilocksFullFieldLundBoundary ctx` | `ProtocolTargetContext` | Local faithful Lund replay/setup surfaces | Definitional | Internal protocol replay constructions |
| Canonical error package | `FinalErrorPackage ctx params` | `ctx : ProtocolTargetContext` | Bundles SumCheck/SZ/MSIS/MSIS-to-Ajtai error surfaces plus one shared `ErrorModel` | Definitional | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofComponentBoundaries` | Explicit SumCheck/SZ/MSIS/Ajtai component boundaries | Canonical final-error assembly deriving the shared `ErrorModel` internally | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofAlignedComponents` | Explicit component packages + one alignment witness | Canonical aligned final-error assembly | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe` | `3 * d <= params.relaxedExpansion`, `0 < params.relaxedExpansion`, aligned SumCheck/SZ/MSIS components | Canonical final-error assembly specialized to the proved `paperCarrier` path | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofGoldilocksPaperCarrier` | `messageLength` + SumCheck/SZ/MSIS component boundaries | Canonical final-error assembly specialized to the Goldilocks Appendix B.2 family | Theorem-Target | `FinalTheoremAssumptions` |
| Final assumptions | `FinalTheoremAssumptions ctx` | None | Bundles the reduction data, lattice parameters, and one canonical final error package | Definitional | Final theorem constructors |
| Constructor | `FinalTheoremAssumptions.ofBoundaryPackages` | Reduction + canonical final error package | Canonical final-theorem assembly | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages` | Reduction + aligned SumCheck/SZ/MSIS components + `3 * d <= params.relaxedExpansion` | Canonical final-theorem assembly on the proved `paperCarrier` path | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages` | `thm3` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta != 0` + SumCheck witness + aligned SumCheck/SZ/MSIS components + `3 * d <= params.relaxedExpansion` | Canonical final-theorem assembly on the paper-facing difference path | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages` | `thm3` + arithmetic + `lowNormInvertibilityAssumption B` with `5 <= B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta != 0` + SumCheck witness + aligned SumCheck/SZ/MSIS components + `3 * d <= params.relaxedExpansion` | Canonical final-theorem assembly on the stronger low-norm route | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages` | Reduction + `messageLength` + SumCheck/SZ/MSIS boundaries | Canonical Goldilocks final-theorem assembly on the proved `paperCarrier` path | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierDerivedSumcheck` | Reduction + `messageLength` + theorem-level MSIS hardness assumption | Canonical Goldilocks final-theorem assembly deriving the witness-level SumCheck and local Schwartz-Zippel boundaries internally and reconstructing the internal MSIS boundary theorem-natively | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages` | `thm3` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta != 0` + SumCheck witness + `messageLength` + SumCheck/SZ/MSIS boundaries | Canonical Goldilocks final-theorem assembly on the paper-facing difference path | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages` | `ctx.bar = nativeBarMatrix` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta != 0` + SumCheck witness + `messageLength` + theorem-level MSIS hardness assumption | Canonical active native-bar Goldilocks final-theorem assembly, deriving Theorem 3, witness-level SumCheck, local Schwartz-Zippel pieces, and the internal MSIS boundary package internally | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages` | `thm3` + arithmetic + `lowNormInvertibilityAssumption B` with `5 <= B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta != 0` + SumCheck witness + `messageLength` + SumCheck/SZ/MSIS boundaries | Canonical Goldilocks final-theorem assembly on the stronger low-norm route | Theorem-Target | `ProofSystem.Protocol` |
| Accessors | `FinalTheoremAssumptions.sumcheckErrorBoundary`, `sumcheckAdvantageBound`, `schwartzZippelBoundary`, `schwartzZippelAdvantageBound`, `msisHardnessBoundary`, `msisAdvantageBound`, `bindingAdvantageBound`, `relaxedBindingAdvantageBound`, `totalErrorDecompFromModel`, `totalErrorNegligible` | `FinalTheoremAssumptions` | Derived per-component theorem surfaces aligned to the shared `ErrorModel` | Theorem-Target | `ProofSystem.Protocol` |
| Final theorem | `FinalTheoremShape`, `finalTheoremShape_of_assumptions`, and the specialized `finalTheoremShape_of_*` wrappers | `FinalTheoremAssumptions` or the corresponding constructor inputs | Completeness plus knowledge-soundness | Theorem-Target | `ProofSystem.Protocol` |

## Design Notes

`finalTheoremShape_of_assumptions` is the canonical capstone theorem. It proves completeness from `weakComposition_of_assumptions` and knowledge-soundness from `strongComposition_of_assumptions`, the local Schwartz-Zippel bound, the witness-level SumCheck failure-advantage bound, the MSIS/Ajtai advantage bounds, and total-error accounting. The derived Goldilocks constructors reconstruct the internal MSIS boundary theorem-natively from the theorem-level MSIS hardness assumption and derive the shared `ErrorModel` and Ajtai reduction data from that reconstructed package. On the native-bar Goldilocks `paperCarrier`-difference route, the constructor also derives the witness-level SumCheck and local Schwartz-Zippel pieces internally, so the only explicit theorem-level security assumption on that route is the MSIS hardness assumption itself.

## Assumption Ledger

`FinalTheoremAssumptions` is the theorem-facing assumption surface. Its canonical constructors make the decomposition explicit: reduction data, lattice parameters, and the canonical final error package. On the native-bar Goldilocks `paperCarrier`-difference route, the explicit remaining security input is the theorem-level MSIS hardness assumption; the SumCheck, Schwartz-Zippel, and internal MSIS boundary pieces are derived internally from the carried transition witness, arithmetic obligations, and hardness theorem.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/InteractiveReductions.lean`
  - `SuperNeo/Interp.lean`
  - `SuperNeo/ProtocolRelations.lean`
  - `SuperNeo/ProofSystem/Lattice.lean`
  - `SuperNeo/ProofSystem/LatticeReductions.lean`
  - `SuperNeo/ProofSystem/SumCheck.lean`
  - `SuperNeo/ProofSystem/Security.lean`
- Downstream consumers:
  - `SuperNeo/ProofSystem/Protocol.lean`

## Construction Discipline

1. Define context-local Schwartz-Zippel surfaces and canonical final error assembly.
2. Bundle reduction data, lattice parameters, and aligned error packages in `FinalTheoremAssumptions`.
3. Derive all per-component theorem surfaces as accessors from that bundle.
4. State and prove `finalTheoremShape_of_assumptions` together with the specialized wrappers from those bundled components.

## Quality Expectations

- No `sorry` in any theorem.
- The final theorem surface stays minimal and theorem-native.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All advertised theorem surfaces are exported through the interface/re-export layers.

## Out of Scope

- Concrete protocol setup instantiation beyond the theorem constructors.
- Proof of the underlying cryptographic assumptions themselves.
