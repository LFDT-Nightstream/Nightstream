import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPlan

/-!
Owns the value-preservation bridge for the direct PiRLC sampler Poseidon2
plan. It proves the exact cross-source chain and verifier-owned entry words.

This module does not own digest-lane decoding or First54 selection rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPreservation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) :
    Fin (PiRLCSamplerPoseidonPlan.sourceWidth program) → F :=
  PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment

structure Encoding {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) : Prop where
  sboxes : (PiRLCSamplerPoseidonPlan.retainedBlock program).EncodesAt
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    (PiRLCSamplerPoseidonPlan.retainedFits geometry) assignment
    (sourceAssignment program prefixAssignment)

theorem encodingOfRetained
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (parentFits : PiRLCRetainedGeometry.laterPoseidonStart program +
      (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount ≤
        logicalWidth)
    (retained : (PiRLCRetainedGeometry.laterPoseidonBlock program).EncodesAt
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits assignment
      prefixAssignment) :
    Encoding geometry assignment prefixAssignment := by
  exact ⟨PiRLCSamplerPoseidonPlan.retainedBlock_encodesAt geometry assignment
    prefixAssignment parentFits retained⟩

def outputValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  SparseLayer.evalState assignment
    ((PiRLCSamplerPoseidonPlan.interface geometry).output current)

/-- Every retained sampler final-state lane reconstructs the exact canonical
source value selected by the sampler Poseidon schedule. -/
theorem outputValue_sourceAssignment
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    outputValue geometry assignment current =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.externalF (fun lane =>
        sourceAssignment program prefixAssignment
          ((PiRLCSamplerPoseidonPlan.schedule program).block.source
            (PoseidonRetainedFamily.slot
              (PiRLCSamplerPoseidonPlan.schedule program) current
              (PoseidonRetainedSlots.finalRow lane)))) := by
  exact PoseidonRetainedFamily.outputState_eval
    (PiRLCSamplerPoseidonPlan.schedule program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    (PiRLCSamplerPoseidonPlan.retainedFits geometry) assignment
    (sourceAssignment program prefixAssignment) encoding.sboxes current

def piCcsFinalValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  SparseLayer.evalState assignment
    (PiRLCSamplerPoseidonPlan.piCcsFinalOutput geometry)

def previousValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  if first : current.val = 0 then
    piCcsFinalValue geometry assignment
  else
    outputValue geometry assignment
      ⟨current.val - 1, by
        have currentBound := current.isLt
        omega⟩

theorem previousOutput_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PiRLCSamplerPoseidonPlan.previousOutput geometry current) =
      previousValue geometry assignment current := by
  funext lane
  unfold PiRLCSamplerPoseidonPlan.previousOutput previousValue outputValue
    piCcsFinalValue
  split <;> rfl

def canonicalInput {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  let decoded := PiRLCSamplerPoseidonPlan.descriptor current
  let previous := previousValue geometry assignment current
  if decoded.2.val = 0 then
    fun lane => previous lane +
      PiRLCSamplerPoseidonPlan.entryWord decoded.1 lane
  else
    previous

theorem inputState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PiRLCSamplerPoseidonPlan.inputState geometry current) =
      canonicalInput geometry assignment current := by
  funext lane
  unfold PiRLCSamplerPoseidonPlan.inputState canonicalInput
  by_cases entry :
      (PiRLCSamplerPoseidonPlan.descriptor current).2.val = 0
  · simp only [entry, if_pos, SparseLayer.evalState, SparseForm.add_eval]
    rw [show
        (PiRLCSamplerPoseidonPlan.previousOutput geometry current lane).eval
            assignment = previousValue geometry assignment current lane by
      exact congrFun (previousOutput_eval geometry assignment current) lane]
    rw [SparseForm.singleton_eval, one, mul_one]
  · simp only [entry, SparseLayer.evalState]
    exact congrFun (previousOutput_eval geometry assignment current) lane

structure CanonicalSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  invocation : ∀ current,
    List.ofFn (outputValue geometry assignment current) =
      Spec.Poseidon2.permute
        (List.ofFn (canonicalInput geometry assignment current))

theorem rowsZero_implies_canonicalSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (rowsZero : (PiRLCSamplerPoseidonPlan.plan geometry).RowsZero assignment) :
    CanonicalSemantics geometry assignment := by
  have semantics := PiRLCSamplerPoseidonPlan.rowsZero_implies_semantics
    geometry assignment one rowsZero
  refine ⟨?_⟩
  intro current
  calc
    List.ofFn (outputValue geometry assignment current) =
        Spec.Poseidon2.permute
          (List.ofFn (SparseLayer.evalState assignment
            ((PiRLCSamplerPoseidonPlan.interface geometry).input current))) :=
      semantics.invocation current
    _ = Spec.Poseidon2.permute
          (List.ofFn (canonicalInput geometry assignment current)) := by
      change Spec.Poseidon2.permute
          (List.ofFn (SparseLayer.evalState assignment
            (PiRLCSamplerPoseidonPlan.inputState geometry current))) = _
      rw [inputState_eval geometry assignment one current]

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPreservation
