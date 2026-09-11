import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan
import NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectBridge
import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignmentNorm
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerBits

/-!
Owns the norm of the complete canonical assignment from actual PiRLC rows.
Only rejection flags and First54 positions are retained as bits. Their values
come from the checked sampler interface and the existing package source map.
Every other retained block uses the existing total field encoding. No Boolean
source value or agreement between different environments is assumed here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem samplerStart_eq
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    PiRLCFirst54DirectBridge.samplerStart source =
      SamplerChain.sourceOffset (Formal.samplerOffset PiRLCInputs.phaseOffset)
        source.val := by
  rfl

private theorem retainedReject_valid {application : Program}
    (raw : RawValues application) (interface : Sampler.Interface)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (bits : Sampler.RetainedBits
      (PiRLCFirst54DirectBridge.samplerStart candidate.source)
      (PiRLCFirst54DirectPlan.baseEnv application raw.base)) :
    LowNormSlot.Valid .bit
      (raw.retainedSource
        (PiRLCFirst54DirectPlan.retainedRejectColumn application candidate)) := by
  have expressionEq := congrArg
    (fun expression : Expr => expression.eval
      (PiRLCFirst54DirectPlan.baseEnv application raw.base))
    (PiRLCFirst54DirectBridge.acceptedExpr_eq interface candidate.source.val candidate)
  simp only [Sampler.selectorInterface, Expr.eval_sub, Expr.eval_var] at expressionEq
  have rejectEq := sub_right_inj.mp expressionEq
  change _ = 0 ∨ _ = 1
  unfold RawValues.retainedSource PiRLCFirst54DirectPlan.retainedRejectColumn
  rw [PiRLCRetainedPreservation.sourceAssignment_package]
  change PiRLCFirst54DirectPlan.baseEnv application raw.base candidate.rejectColumn = 0 ∨
    PiRLCFirst54DirectPlan.baseEnv application raw.base candidate.rejectColumn = 1
  rw [← rejectEq]
  exact bits.1 candidate.round

private theorem retainedPosition_valid {application : Program}
    (raw : RawValues application)
    (position : PiRLCFirst54DirectSchedule.Position)
    (bits : Sampler.RetainedBits
      (PiRLCFirst54DirectBridge.samplerStart position.candidate.source)
      (PiRLCFirst54DirectPlan.baseEnv application raw.base)) :
    LowNormSlot.Valid .bit
      (raw.retainedSource
        (PiRLCFirst54DirectPlan.retainedPositionColumn application position)) := by
  change _ = 0 ∨ _ = 1
  unfold RawValues.retainedSource PiRLCFirst54DirectPlan.retainedPositionColumn
  rw [PiRLCRetainedPreservation.sourceAssignment_package]
  change PiRLCFirst54DirectPlan.positionOutputValue application raw.base position = 0 ∨
    PiRLCFirst54DirectPlan.positionOutputValue application raw.base position = 1
  rw [PiRLCFirst54DirectBridge.positionOutputValue_eq_eval,
    ← PiRLCFirst54DirectBridge.selectorOffset_eq]
  exact bits.2 position.candidate.round position.slot

private theorem schedule_valid {application : Program}
    (raw : RawValues application) (interface : Sampler.Interface)
    (bits : ∀ source : Fin PiRLCFirst54DirectSchedule.sourceCount,
      Sampler.RetainedBits (PiRLCFirst54DirectBridge.samplerStart source)
        (PiRLCFirst54DirectPlan.baseEnv application raw.base)) :
    ∀ entry ∈ raw.schedule, ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot)) := by
  rw [← PerApplicationAssignmentPlan.expand_eq_schedule raw]
  intro entry member
  obtain ⟨kind, _, rfl⟩ := List.mem_map.mp member
  cases kind <;> intro slot
  case first54Reject =>
    exact retainedReject_valid raw interface
      (PiRLCFirst54DirectSchedule.candidate slot) (bits _)
  case first54Position =>
    exact retainedPosition_valid raw
      (PiRLCFirst54DirectSchedule.position slot) (bits _)
  all_goals trivial

variable {application : Program}
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (raw : RawValues application)
  (assumptions : Formal.Assumptions
    (PerApplicationFixedPoint.relation application fits)
    PiRLCInputs.interface PiRLCInputs.phaseOffset
    (PiRLCFirst54DirectPlan.baseEnv application raw.base))
  (rows : holdsFlat (PiRLCFirst54DirectPlan.baseEnv application raw.base)
    (Formal.opsAt (PerApplicationFixedPoint.relation application fits)
      PiRLCInputs.interface PiRLCInputs.phaseOffset))

include assumptions rows

/-- Actual canonical PiRLC rows and their phase input assumptions imply the
strict norm bound on every logical coordinate. The sampler rows establish
both retained bit families; the public hash and all field blocks use their
existing total low-norm encodings. No bit-validity premise is supplied. -/
theorem assignment_norm_of_piRlcRows
    (column : Fin (PerApplicationFixedPoint.logicalWidth application)) :
    centeredMagnitude (raw.assignment column) < 2 := by
  have samplerBits := Formal.retainedSamplerBits_of_rows
    (PerApplicationFixedPoint.relation application fits)
    PiRLCInputs.interface PiRLCInputs.phaseOffset
    (PiRLCFirst54DirectPlan.baseEnv application raw.base) assumptions rows
  have bits : ∀ source : Fin PiRLCFirst54DirectSchedule.sourceCount,
      Sampler.RetainedBits (PiRLCFirst54DirectBridge.samplerStart source)
        (PiRLCFirst54DirectPlan.baseEnv application raw.base) := by
    intro source
    rw [samplerStart_eq]
    exact samplerBits source
  apply CanonicalBlockAssignment.assignment_norm
    (encodedHashCells raw.outputDigest) raw.schedule
  · intro publicColumn
    exact encHash_norm (publicFits := PerApplicationFixedPoint.publicFits application)
      raw.outputDigest publicColumn
  · exact schedule_valid raw (SamplerChain.childInterface
      (Formal.samplerInterface (Formal.atOffset
        (PiRLCInputs.interface
          (publicFits := PerApplicationFixedPoint.publicFits application))
        PiRLCInputs.phaseOffset))
      (Formal.samplerOffset PiRLCInputs.phaseOffset) 0) bits

/-- The actual selected canonical assignment is pointwise bounded on the
complete Phi81 carrier. Its logical part is bounded by the PiRLC row theorem,
and the existing carrier extension fills every remaining coordinate with zero.
This is an encoding guarantee; the full semantic-step constructor must still
supply the canonical PiRLC rows and their input assumptions. -/
theorem completeAssignment_norm_of_piRlcRows
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (PerApplicationFixedPoint.logicalWidth application))) :
    centeredMagnitude (raw.completeAssignment column) < 2 := by
  unfold RawValues.completeAssignment
  split
  · exact assignment_norm_of_piRlcRows fits raw assumptions rows _
  · simp

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
