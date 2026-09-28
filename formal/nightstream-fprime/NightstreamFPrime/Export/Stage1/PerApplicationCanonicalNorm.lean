import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan
import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignmentNorm

/-! Every current retained block uses the total field encoding. The canonical
assignment and its zero extension therefore satisfy the low-norm bound for
all raw values, independently of row acceptance. -/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem schedule_valid {application : Program} (raw : RawValues application) :
    ∀ entry ∈ raw.schedule, ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot)) := by
  rw [← PerApplicationAssignmentPlan.expand_eq_schedule raw]
  intro entry member
  obtain ⟨kind, _, rfl⟩ := List.mem_map.mp member
  cases kind <;> intro slot <;> trivial

theorem assignment_norm {application : Program} (raw : RawValues application)
    (column : Fin (PerApplicationFixedPoint.logicalWidth application)) :
    centeredMagnitude (raw.assignment column) < 2 := by
  apply CanonicalBlockAssignment.assignment_norm
    (encodedHashCells raw.outputDigest) raw.schedule
  · intro publicColumn
    exact encHash_norm (publicFits := PerApplicationFixedPoint.publicFits application)
      raw.outputDigest publicColumn
  · exact schedule_valid raw

theorem completeAssignment_norm {application : Program} (raw : RawValues application)
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (PerApplicationFixedPoint.logicalWidth application))) :
    centeredMagnitude (raw.completeAssignment column) < 2 := by
  unfold RawValues.completeAssignment
  split
  · exact assignment_norm raw _
  · simp

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
