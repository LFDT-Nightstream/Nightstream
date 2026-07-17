import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270

/-!
Stable artifact facade for the exact selective-compiler 270-carrier fixture.

Owns: one handwritten import boundary over the generated layout values, the
selector prefix, one representative arm-gated row, and thirteen final-matrix
public-padding rows.

Does not own: decoding, semantic row classification, a full fixed-point F′
artifact, NIFS soundness, constraint necessity, or row removal.

Emits constraints: no.

| Child | Artifact ownership | Semantic owner |
|---|---|---|
| generated carrier layout | exact compiler ranges | `AlignedCompiler.ProductionCarrier` |
| generated selector prefix | three domain rows plus the total row | `SelectorComposition.ArtifactRefinement` |
| generated arm gate | one coefficient-exact retained row | `SelectorComposition.ArtifactRefinement` |
| generated public-padding rows | exact thirteen-port sparse rows | `SelectiveCcs.Padding.ArtifactRefinement` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270
  (logicalPublicInputLen publicInputLen publicPaddingColumns selectorColumns
    privateAlignmentPaddingColumns sharedPrivateStart sharedPrivateEnd
    branchStart branchEnd ringAlignmentPaddingStart ringAlignmentPaddingEnd
    rawSelectorRows rawOneHotRow rawGatedRow rawPaddingRows)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270
