import NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry

/-!
Owns the concrete sparse-form inputs for the direct PiRLC product plan.
The parent supplies the PiCCS-owned value forms. Challenge, prior,
and final values use their checked block slots.

This module does not construct the final assignment or compose other phases.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCRetainedGeometry

abbrev Values (logicalWidth : Nat) :=
  Fin PiRLCProductSchedule.invocationCount → SparseForm logicalWidth

def productInputs {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth) :
    PiRLCProductPlan.Inputs program logicalWidth where
  oneColumn := oneColumn geometry
  challenge := fun invocation lane =>
    (challengeBlock program).form
      (challengeStart program) (challengeFits geometry)
      (PiRLCProductSourceBlocks.challengeIndex
        (PiRLCProductSchedule.descriptor invocation).source lane)
  value := fun invocation lane =>
    values ((PiRLCProductSchedule.descriptor invocation).withLane lane).invocation
  prior := fun invocation =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    if first : descriptor.source.val = 0 then
      .empty
    else
      (productOutputBlock program).form
        (productOutputStart program) (productOutputFits geometry) <|
          (descriptor.previousSource first).invocation
  output := fun invocation =>
    (productOutputBlock program).form
      (productOutputStart program) (productOutputFits geometry) invocation
  group := fun invocation group =>
    (productGroupBlock program).form
      (productGroupStart program) (productGroupFits geometry) <|
        Fin.encodeProd (invocation, group)

end NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs
