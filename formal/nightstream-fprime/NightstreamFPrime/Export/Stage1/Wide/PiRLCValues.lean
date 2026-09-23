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
    output initial values ⟨index.val / 54, by change _ < 969; change index.val < 52326 at inOutput; omega⟩
      ⟨index.val % 54, Nat.mod_lt _ (by decide)⟩
  else
    quotient initial values ⟨(index.val - outputCount) / 54, by
      have upper := index.isLt
      change index.val < 104652 at upper
      change ¬ index.val < 52326 at inOutput
      change (index.val - 52326) / 54 < 969
      omega⟩ ⟨(index.val - outputCount) % 54, Nat.mod_lt _ (by decide)⟩

theorem fieldValue_output (initial : Initial) (values : Values) (ring : RingIndex) (lane : Fin ringDegree) :
    fieldValue initial values (outputSlot ring lane) = output initial values ring lane := by
  have r : ring.val < 969 := ring.isLt
  have l : lane.val < 54 := lane.isLt
  unfold fieldValue outputSlot
  rw [dif_pos (by change ring.val * 54 + lane.val < 52326; omega)]
  have quo : (ring.val * 54 + lane.val) / 54 = ring.val := by omega
  have rem : (ring.val * 54 + lane.val) % 54 = lane.val := by omega
  exact congrArg₂ (output initial values) (Fin.ext quo) (Fin.ext rem)

theorem fieldValue_quotient (initial : Initial) (values : Values) (ring : RingIndex) (lane : Fin ringDegree) :
    fieldValue initial values (quotientSlot ring lane) = quotient initial values ring lane := by
  have r : ring.val < 969 := ring.isLt
  have l : lane.val < 54 := lane.isLt
  unfold fieldValue quotientSlot
  rw [dif_neg (by change ¬52326 + ring.val * 54 + lane.val < 52326; omega)]
  have quo : (52326 + ring.val * 54 + lane.val - 52326) / 54 = ring.val := by omega
  have rem : (52326 + ring.val * 54 + lane.val - 52326) % 54 = lane.val := by omega
  exact congrArg₂ (quotient initial values) (Fin.ext quo) (Fin.ext rem)

end NightstreamFPrime.Export.Stage1.Wide.PiRLCValues
