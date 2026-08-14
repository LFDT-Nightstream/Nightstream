import Nightstream.Implementation.Nebula.FPrime.Recursive.SizeClosure

/-! Focused checks for the finite V2 recursive-size certificate. -/

namespace NightstreamTests.NebulaRecursiveSizeClosure

open Nightstream.Implementation.Nebula.RecursiveSizeClosure

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
      [Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.rejectingConstantRow]
      [Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.zeroRow] :=
  finiteCapacity_does_not_imply_requiredRowsPresent.2.2

end NightstreamTests.NebulaRecursiveSizeClosure
