import NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry

/-!
Owns the concrete sparse-form inputs for the direct PiRLC product and First54
plans. The parent supplies the PiCCS-owned value forms. Challenge, prior,
and final values reuse existing block slots.

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
    (PiRLCFirst54RetainedBlocks.valueBlock program).form
      (valueStart program) (valueFits geometry) <|
        PiRLCFirst54DirectSchedule.valueIndex <|
          PiRLCProductSourceBlocks.challengeValueDescriptor
            (PiRLCProductSchedule.descriptor invocation).source lane
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

def first54Inputs {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCFirst54DirectPlan.Inputs program logicalWidth where
  oneColumn := oneColumn geometry
  reject := fun candidate =>
    (PiRLCFirst54RetainedBlocks.rejectBlock program).form
      (rejectStart program) (rejectFits geometry)
        (PiRLCFirst54DirectSchedule.candidateIndex candidate)
  symbol := fun candidate =>
    (PiRLCFirst54RetainedBlocks.symbolBlock program).form
      (symbolStart program) (symbolFits geometry)
        (PiRLCFirst54DirectSchedule.candidateIndex candidate)
  position := fun descriptor =>
    (PiRLCFirst54RetainedBlocks.positionBlock program).form
      (positionStart program) (positionFits geometry)
        (PiRLCFirst54DirectSchedule.positionIndex descriptor)
  value := fun descriptor =>
    (PiRLCFirst54RetainedBlocks.valueBlock program).form
      (valueStart program) (valueFits geometry)
        (PiRLCFirst54DirectSchedule.valueIndex descriptor)
  product := fun candidate =>
    (PiRLCFirst54RetainedBlocks.productBlock program).form
      (first54ProductStart program) (first54ProductFits geometry) candidate

end NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs
