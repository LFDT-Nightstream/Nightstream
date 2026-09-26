import NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness

/-! Accepted compact rows force the direct sequential PiRLC outputs. This
statement holds for any assignment, independently of its witness program. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCSoundness

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCGeometry

theorem output_eq_direct {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment) (ring : RingIndex) :
    Phi81ProductPlan.evalState assignment (output interface ring) =
      PiRLCValues.output (PiRLCWitness.initial interface assignment)
        (PiRLCWitness.inputValues interface assignment) ring := by
  let initial := PiRLCWitness.initial interface assignment
  let values := PiRLCWitness.inputValues interface assignment
  have challengeEq (descriptor : Descriptor) :
      PiRLCPlan.challenge (planInterface interface) assignment descriptor.invocation =
        PiRLCValues.challenge initial descriptor.source.val := by
    change Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
      (List.ofFn initial) (PiRLCProductRingSchedule.descriptor descriptor.invocation).source.val = _
    rw [PiRLCProductRingSchedule.descriptor_invocation]
    rfl
  have recurrence := fun ring => soundness compiled interface assignment one rows ring
  have all : ∀ source : Nat, ∀ descriptor : Descriptor, descriptor.source.val = source →
      Phi81ProductPlan.evalState assignment (output interface descriptor.invocation) =
        PiRLCValues.output initial values descriptor.invocation := by
    intro source
    induction source with
    | zero =>
        intro descriptor atZero
        have accepted := recurrence descriptor.invocation
        rw [PiRLCValues.output_step]
        simp only [PiRLCProductRingSchedule.descriptor_invocation, atZero, ↓reduceDIte]
        rw [accepted]
        apply congrArg₂ ringFAdd
        · simp only [prior, PiRLCProductRingSchedule.descriptor_invocation, atZero, ↓reduceDIte]
          funext lane
          exact SparseForm.empty_eval _
        · rw [challengeEq, atZero]
          rfl
    | succ source ih =>
        intro descriptor atNext
        have nonzero : descriptor.source.val ≠ 0 := by omega
        have accepted := recurrence descriptor.invocation
        rw [PiRLCValues.output_step]
        simp only [PiRLCProductRingSchedule.descriptor_invocation, nonzero, ↓reduceDIte]
        rw [accepted]
        apply congrArg₂ ringFAdd
        · simp only [prior, PiRLCProductRingSchedule.descriptor_invocation, nonzero, ↓reduceDIte]
          exact ih (previous descriptor nonzero) (by change descriptor.source.val - 1 = source; omega)
        · rw [challengeEq]
          rfl
  have result := all (PiRLCProductRingSchedule.descriptor ring).source.val
    (PiRLCProductRingSchedule.descriptor ring) rfl
  simpa only [PiRLCProductRingSchedule.invocation_descriptor] using result

end NightstreamFPrime.Export.Stage1.Wide.PiRLCSoundness
