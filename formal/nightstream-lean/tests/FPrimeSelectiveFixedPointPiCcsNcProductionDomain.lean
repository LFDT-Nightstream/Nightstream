import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain

/-! Focused regression for the complete fixed-point `Pi_CCS` domain. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcProductionDomain

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain

#check rowCube_covers
#check semanticShape_carrierWidth
#check semanticShape_blockCount
#check flatDomain_covers
#check blockLaneDomain_covers
#check rowVariables_minimal
#check flatColumnVariables_minimal
#check eighteenBlockVariables_cover
#check blockVariables_minimal
#check laneVariables_minimal
#check artifact_width_accounting
#check artifact_fits_current_constructor_guard

example : semanticShape.rowVariables = 24 := semanticShape_rowVariables

example : semanticShape.logicalWidth = 11725506 :=
  semanticShape_logicalWidth_exact

example : FixedArtifact.unpaddedCoordinates = 11725454 /\
    FixedArtifact.relationColumns = 11725506 /\
    16000000 - FixedArtifact.relationColumns = 4274494 := by
  exact ⟨artifact_unpaddedCoordinates,
    artifact_relationColumns,
    artifact_fits_current_constructor_guard.2⟩

example : semanticShape.freshCount = 1 ∧
    semanticShape.runningCount = 14 ∧
    semanticShape.matrixCount = 13 ∧
    semanticShape.sourceCount = 15 := by
  decide

example :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables +
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables = 25 ∧
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.fe.columnVariables +
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.fe.laneVariables = 30 :=
  ⟨blockLaneRoundCount, legacyFlatRoundCount⟩

example : liveLaneCount = 54 ∧ virtualLaneCount = 10 ∧
    liveLaneCount + virtualLaneCount = 64 := by
  decide

example :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount
        semanticShape.carrierWidth = 217139 :=
  semanticShape_blockCount

example {variables : Nat} (covers : 217139 <= 2 ^ variables) :
    18 <= variables := by
  apply blockVariables_minimal
  simpa using covers

end Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcProductionDomain
