import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Profile

/-! Focused theorem-shape checks for artifact-independent output profiles. -/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

#check Profile.fieldCount_ofSemanticShape
#check Profile.diagnosticThreeMatrix_fieldCount
#check Profile.steadyFixedPointThirteenMatrix_fieldCount
#check Profile.diagnosticThreeMatrix_ne_steadyFixedPointThirteenMatrix
#check Profile.diagnosticThreeMatrix_fieldCount_ne_steadyFixedPointThirteenMatrix

#guard Profile.fieldCount Profile.diagnosticThreeMatrix = 6683
#guard Profile.fieldCount Profile.steadyFixedPointThirteenMatrix = 23033
#guard Profile.diagnosticThreeMatrix != Profile.steadyFixedPointThirteenMatrix

example (shape : SemanticShape) :
    Profile.fieldCount (Profile.ofSemanticShape shape) =
      ActiveSemantics.fieldCount shape :=
  Profile.fieldCount_ofSemanticShape shape
