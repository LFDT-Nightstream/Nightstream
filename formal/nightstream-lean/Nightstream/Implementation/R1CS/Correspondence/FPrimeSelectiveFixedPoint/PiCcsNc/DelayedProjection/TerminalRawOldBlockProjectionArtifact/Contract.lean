import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.RowSemantics

/-!
Artifact contract for the generated final-round-factorized production
raw-old-block projection.

This leaf assembles the independently proved profile shape, physical row
permutation, unique row ownership, and exact row semantics.  Runtime emitter
column placement and assignment satisfaction are subsequent refinement leaves.

Owns: the fixed `ArtifactContract` joining production dimensions, the complete
conceptual-to-physical row permutation, unique row ownership, and equality of
every generated row with its symbolic compiler row.

Does not own: runtime emitter-column renaming, concrete assignment values,
row satisfaction, terminal CE, semantic acceptance, costs, or row-removal
authority.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_artifact.profile` | generated dimensions equal the fixed compiler profile | checked / derived |
| `f_prime.pi_ccs_nc.delayed.projection_artifact.ownership` | every conceptual row maps bijectively to exactly one physical artifact row | derived |
| `f_prime.pi_ccs_nc.delayed.projection_artifact.equations` | each generated row equals the corresponding symbolic tensor, product, final-scale, or terminal equation | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

/-- Fixed conceptual-to-physical row identity for the generated production
program.  Runtime emitter placement is a subsequent column-renaming theorem;
it is not a premise of this contract. -/
def productionArtifactContract :
    ArtifactContract productionFactoredLayout artifactRow := by
  refine
    { profileRadix := rfl
      profileChildren := rfl
      profileActiveLanes := rfl
      profilePaddingLanes := rfl
      profileLogicalWidth := rfl
      profileFactorEnabled := productionFactorEnabled
      profileTensorVariables := productionTensorVariables
      profileFactoredVariable := productionFactoredVariable
      profileBlockCount := productionBlockCount
      profileTensorMultiplications :=
        productionFactoredTensorMultiplicationCount
      profileRows := productionRowCount_exact
      shape := productionShape
      physicalIndex := productionPhysicalIndex
      physicalIndex_injective := productionPhysicalIndex_injective
      physicalIndex_surjective := productionPhysicalIndex_surjective
      rowAt := productionRowAt }

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
