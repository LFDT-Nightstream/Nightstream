import Nightstream.Implementation.NebulaV2.RecursiveSizeClosure

/-! Focused checks for the finite V2 recursive-size certificate. -/

namespace NightstreamTests.NebulaV2RecursiveSizeClosure

open Nightstream.Implementation.NebulaV2.RecursiveSizeClosure

#check Payload
#check payloadCodec_canonical
#check requiredWords_exact
#check FiniteArtifactCapacity
#check finiteArtifactCapacity
#check MatchesCapacities
#check capacityHoldsForMatchingLayout
#check completeCompilerFit

example : RowFitOnly 1 1 :=
  rowFitOnly_does_not_imply_fullCapacity.1

example : ¬ FullCapacityFit 1 1 2 1 1 :=
  rowFitOnly_does_not_imply_fullCapacity.2

example :
    ¬ RequiredRowsPresent
      [Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreManifestFor.rejectingConstantRow]
      [Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreManifestFor.zeroRow] :=
  finiteCapacity_does_not_imply_requiredRowsPresent.2.2

end NightstreamTests.NebulaV2RecursiveSizeClosure
