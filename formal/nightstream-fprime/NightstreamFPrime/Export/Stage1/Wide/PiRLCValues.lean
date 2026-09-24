import NightstreamFPrime.Export.Stage1.Wide.PiRLCGeometry

/-! Direct values for the retained ring outputs and quotient coefficients.
The construction reads the initial transcript and each right operand. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCValues

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint
open PiRLCGeometry

abbrev Initial := Fin 8 → F
abbrev Values := RingIndex → RingF

/-- The field-state view used by the compact sampler's soundness theorem. -/
def challenge (initial : Initial) (source : Nat) : RingF :=
  Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
    (List.ofFn initial) source

def term (initial : Initial) (values : Values) (descriptor : Descriptor) (source : Nat) : RingF :=
  if bounded : source < 17 then
    ringFMul (challenge initial source) (values (withSource descriptor ⟨source, bounded⟩).invocation)
  else fun _ => 0

def partialSum (terms : Nat → RingF) : Nat → RingF
  | 0 => fun _ => 0
  | count + 1 => ringFAdd (partialSum terms count) (terms count)

def output (initial : Initial) (values : Values) (ring : RingIndex) : RingF :=
  let descriptor := PiRLCProductRingSchedule.descriptor ring
  partialSum (term initial values descriptor) (descriptor.source.val + 1)

def quotient (initial : Initial) (values : Values) (ring : RingIndex) : RingF :=
  fun lane => Phi81Relation.QuotientProduct.quotientCoeff
    (challenge initial (PiRLCProductRingSchedule.descriptor ring).source.val) (values ring) lane

theorem term_withSource (initial : Initial) (values : Values) (descriptor : Descriptor) (source : Fin 17) :
    term initial values (withSource descriptor source) = term initial values descriptor := by
  funext index
  unfold term
  split <;> rfl

theorem output_step (initial : Initial) (values : Values) (ring : RingIndex) :
    output initial values ring =
      ringFAdd
        (if first : (PiRLCProductRingSchedule.descriptor ring).source.val = 0 then fun _ => 0
          else output initial values (previous (PiRLCProductRingSchedule.descriptor ring) first).invocation)
        (ringFMul (challenge initial (PiRLCProductRingSchedule.descriptor ring).source.val) (values ring)) := by
  let descriptor := PiRLCProductRingSchedule.descriptor ring
  have termEq : term initial values descriptor descriptor.source.val =
      ringFMul (challenge initial descriptor.source.val) (values ring) := by
    rw [term, dif_pos (show descriptor.source.val < 17 from descriptor.source.isLt)]
    change ringFMul (challenge initial descriptor.source.val)
      (values (withSource descriptor descriptor.source).invocation) = _
    rw [withSource_self]
    change ringFMul _ (values (PiRLCProductRingSchedule.descriptor ring).invocation) = _
    rw [PiRLCProductRingSchedule.invocation_descriptor]
  change partialSum (term initial values descriptor) (descriptor.source.val + 1) = _
  rw [partialSum, termEq]
  apply congrArg (fun prior => ringFAdd prior _)
  split_ifs with first
  · rw [show descriptor.source.val = 0 from first]
    rfl
  · unfold output
    rw [PiRLCProductRingSchedule.descriptor_invocation]
    change partialSum (term initial values descriptor) descriptor.source.val =
      partialSum (term initial values (withSource descriptor ⟨descriptor.source.val - 1, _⟩))
        (descriptor.source.val - 1 + 1)
    rw [term_withSource]
    have nonzero : descriptor.source.val ≠ 0 := first
    rw [Nat.sub_add_cancel (by omega : 1 ≤ descriptor.source.val)]

theorem quotient_eq (initial : Initial) (values : Values) (ring : RingIndex) :
    quotient initial values ring =
      Phi81Relation.QuotientProduct.quotient
        (challenge initial (PiRLCProductRingSchedule.descriptor ring).source.val) (values ring) := by
  funext lane
  exact Phi81Relation.QuotientProduct.quotientCoeff_eq_quotient _ _ lane

def fieldValue (initial : Initial) (values : Values) (index : Fin fieldBlock.slotCount) : F :=
  if inOutput : index.val < outputCount then
    let slot : Fin PiRLCProductSchedule.invocationCount := ⟨index.val, inOutput⟩
    output initial values (PiRLCProductRingSchedule.ringInvocation slot)
      (PiRLCProductSchedule.descriptor slot).lane
  else
    let slot : Fin PiRLCProductSchedule.invocationCount := ⟨index.val - outputCount, by
      have upper := index.isLt
      change index.val < 104652 at upper
      change ¬index.val < 52326 at inOutput
      change index.val - 52326 < 52326
      omega⟩
    quotient initial values (PiRLCProductRingSchedule.ringInvocation slot)
      (PiRLCProductSchedule.descriptor slot).lane

theorem fieldValue_output (initial : Initial) (values : Values) (ring : RingIndex) (lane : Fin ringDegree) :
    fieldValue initial values (outputSlot ring lane) = output initial values ring lane := by
  have bound : (PiRLCProductRingSchedule.laneInvocation ring lane).val < outputCount :=
    (PiRLCProductRingSchedule.laneInvocation ring lane).isLt
  unfold fieldValue outputSlot
  rw [dif_pos bound]
  change output initial values
    (PiRLCProductRingSchedule.ringInvocation (PiRLCProductRingSchedule.laneInvocation ring lane))
    (PiRLCProductSchedule.descriptor (PiRLCProductRingSchedule.laneInvocation ring lane)).lane = _
  rw [PiRLCProductRingSchedule.ringInvocation_laneInvocation,
    PiRLCProductRingSchedule.descriptor_laneInvocation]
  rfl

theorem fieldValue_quotient (initial : Initial) (values : Values) (ring : RingIndex) (lane : Fin ringDegree) :
    fieldValue initial values (quotientSlot ring lane) = quotient initial values ring lane := by
  unfold fieldValue quotientSlot
  rw [dif_neg (by dsimp only; omega)]
  simp only [Nat.add_sub_cancel_left]
  change quotient initial values
    (PiRLCProductRingSchedule.ringInvocation (PiRLCProductRingSchedule.laneInvocation ring lane))
    (PiRLCProductSchedule.descriptor (PiRLCProductRingSchedule.laneInvocation ring lane)).lane = _
  rw [PiRLCProductRingSchedule.ringInvocation_laneInvocation,
    PiRLCProductRingSchedule.descriptor_laneInvocation]
  rfl

end NightstreamFPrime.Export.Stage1.Wide.PiRLCValues
