import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold

/-! Retain all carried Pad values of one complete block and apply its first
adjacent prefix fold. The original block order and every tail lane remain.
Later folds operate on the concatenated 27-value block outputs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPadPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

private theorem dot_zero (basis : Vector K ringDegree) :
    PiCCSWeightedBasis.dotK basis (Vector.replicate ringDegree K.zero).get =
      extensionOps.zero := by
  have reads : (Vector.replicate ringDegree K.zero).get = fun _ => extensionOps.zero := by
    funext lane
    change (Vector.replicate ringDegree K.zero)[lane.val] = K.zero
    exact Vector.getElem_replicate lane.isLt
  rw [reads]
  unfold PiCCSWeightedBasis.dotK
  simp only [extensionLaws.mul_zero, sumMap_zero extensionOps extensionLaws]

/-- Materialize each original forward dot product once. Only equality of the
complete combined block with zero skips the arithmetic. -/
def blockValues (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) : Vector K ringDegree :=
  if values = Vector.replicate ringDegree K.zero then
    Vector.replicate ringDegree K.zero
  else Vector.ofFn fun lane => PiCCSWeightedBasis.dotK (basis.get lane) values.get

/-- Every retained lane is the same dot product used by blockMoment, with
no support, source-validity, or nonzero premise. -/
theorem blockValues_value (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (lane : Fin ringDegree) :
    (blockValues basis values).get lane =
      PiCCSWeightedBasis.dotK (basis.get lane) values.get := by
  by_cases empty : values = Vector.replicate ringDegree K.zero
  · subst values
    rw [blockValues, if_pos rfl, dot_zero]
    change (Vector.replicate ringDegree K.zero)[lane.val] = K.zero
    exact Vector.getElem_replicate lane.isLt
  · rw [blockValues, if_neg empty]
    change (Vector.ofFn _)[lane.val] = _
    rw [Vector.getElem_ofFn]

/-- The existing array view includes every lane and has an exact zero suffix. -/
theorem blockValues_getD (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (index : Nat) :
    (blockValues basis values).toArray.getD index K.zero =
      if bounded : index < ringDegree then
        PiCCSWeightedBasis.dotK (basis.get ⟨index, bounded⟩) values.get
      else K.zero := by
  rw [Array.getD_eq_getD_getElem?]
  by_cases bounded : index < ringDegree
  · have live : index < (blockValues basis values).toArray.size := by
      rw [Vector.size_toArray]
      exact bounded
    rw [Array.getElem?_eq_getElem live, Option.getD_some, dif_pos bounded]
    exact blockValues_value basis values ⟨index, bounded⟩
  · have outside : (blockValues basis values).toArray.size ≤ index := by
      rw [Vector.size_toArray]
      exact Nat.le_of_not_gt bounded
    rw [Array.getElem?_eq_none outside, Option.getD_none, dif_neg bounded]

/-- Fix the first challenge through the existing executable prefix fold.
A complete 54-lane input block supplies exactly 27 adjacent pairs. -/
def foldedBlock (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (challenge : K) : Array K :=
  PrefixFold.foldOne extensionOps (blockValues basis values).toArray challenge

theorem foldedBlock_size (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (challenge : K) :
    (foldedBlock basis values challenge).size = 27 := by
  rw [foldedBlock, PrefixFold.foldOne_size, Vector.size_toArray]
  rfl

/-- Total coefficient equality, including reads beyond the stored block.
The low and high operands are the exact original forward Pad dot products. -/
theorem foldedBlock_getD (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (challenge : K) (index : Nat) :
    (foldedBlock basis values challenge).getD index K.zero =
      PrefixFold.interpolate extensionOps challenge
        (if bounded : 2 * index < ringDegree then
          PiCCSWeightedBasis.dotK (basis.get ⟨2 * index, bounded⟩) values.get
        else K.zero)
        (if bounded : 2 * index + 1 < ringDegree then
          PiCCSWeightedBasis.dotK (basis.get ⟨2 * index + 1, bounded⟩) values.get
        else K.zero) := by
  simpa only [foldedBlock, show extensionOps.zero = K.zero from rfl,
    blockValues_getD] using
    (PrefixFold.foldOne_getD extensionOps extensionLaws
      (blockValues basis values).toArray challenge index)

/-- Missing coefficients use the same zero default as the full prefix fold. -/
theorem foldedBlock_getD_of_outside
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (values : Vector K ringDegree) (challenge : K)
    (index : Nat) (outside : 27 ≤ index) :
    (foldedBlock basis values challenge).getD index K.zero = K.zero := by
  have beyond : (foldedBlock basis values challenge).size ≤ index := by
    rw [foldedBlock_size]
    exact outside
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none beyond,
    Option.getD_none]

end NightstreamFPrime.Export.Stage1.PiCCSPadPrefix
