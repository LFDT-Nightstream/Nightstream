import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum

/-! One already combined complete block supplies both carried Pad moments.
Only an exact zero of the whole K block skips the 54 basis dot products. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPadBlockMoment

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NumericCompletionSum (numericSum)

private def paritySum (count : Nat) (term : Nat → K) : K × K :=
  Nat.fold count (fun index _ accumulated =>
    if index % 2 = 0 then
      (extensionOps.add accumulated.1 (term index), accumulated.2)
    else (accumulated.1, extensionOps.add accumulated.2 (term index)))
    (extensionOps.zero, extensionOps.zero)

private theorem paritySum_value (count : Nat) (term : Nat → K) :
    paritySum count term =
      (numericSum extensionOps count (fun index =>
        if index % 2 = 0 then term index else extensionOps.zero),
       numericSum extensionOps count (fun index =>
        if index % 2 = 0 then extensionOps.zero else term index)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [paritySum, Nat.fold_succ] at ⊢
      change (if count % 2 = 0 then
          (extensionOps.add (paritySum count term).1 (term count), (paritySum count term).2)
        else ((paritySum count term).1, extensionOps.add (paritySum count term).2 (term count))) = _
      rw [inductionHypothesis]
      by_cases even : count % 2 = 0
      · simp only [if_pos even, numericSum, Nat.fold_succ, extensionLaws.add_zero]
      · simp only [if_neg even, numericSum, Nat.fold_succ, extensionLaws.add_zero]

private theorem paritySum_zero (count : Nat) :
    paritySum count (fun _ => extensionOps.zero) =
      (extensionOps.zero, extensionOps.zero) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      change (if count % 2 = 0 then
          (extensionOps.add (paritySum count (fun _ => extensionOps.zero)).1 extensionOps.zero,
            (paritySum count (fun _ => extensionOps.zero)).2)
        else ((paritySum count (fun _ => extensionOps.zero)).1,
          extensionOps.add (paritySum count (fun _ => extensionOps.zero)).2 extensionOps.zero)) = _
      rw [inductionHypothesis]
      split <;> simp only [extensionLaws.add_zero]

private theorem dot_zero (basis : Vector K ringDegree) :
    PiCCSWeightedBasis.dotK basis (Vector.replicate ringDegree K.zero).get = extensionOps.zero := by
  have reads : (Vector.replicate ringDegree K.zero).get = fun _ => extensionOps.zero := by
    funext lane
    change (Vector.replicate ringDegree K.zero)[lane.val] = K.zero
    exact Vector.getElem_replicate lane.isLt
  rw [reads]
  unfold PiCCSWeightedBasis.dotK
  simp only [extensionLaws.mul_zero, sumMap_zero extensionOps extensionLaws]

/-- Combine the source block before calling this kernel. The weight uses the
absolute pair index; a 54-column block is not treated as a binary-aligned chunk. -/
def blockMoment (basis : FixedArray (Vector K ringDegree) ringDegree)
    (weight : Nat → K) (block : Nat) (values : Vector K ringDegree) : K × K :=
  if values = Vector.replicate ringDegree K.zero then (K.zero, K.zero)
  else paritySum ringDegree fun index =>
    if bounded : index < ringDegree then
      extensionOps.mul (weight ((block * ringDegree + index) / 2))
        (PiCCSWeightedBasis.dotK (basis.get ⟨index, bounded⟩) values.get)
    else extensionOps.zero

/-- Both outputs equal the complete original 54-column parity sums for any
basis, weight and combined block. The zero branch adds no source assumption. -/
theorem blockMoment_value (basis : FixedArray (Vector K ringDegree) ringDegree)
    (weight : Nat → K) (block : Nat) (values : Vector K ringDegree) :
    let column := fun index =>
      if bounded : index < ringDegree then
        extensionOps.mul (weight ((block * ringDegree + index) / 2))
          (PiCCSWeightedBasis.dotK (basis.get ⟨index, bounded⟩) values.get)
      else extensionOps.zero
    blockMoment basis weight block values =
      (numericSum extensionOps ringDegree (fun index =>
        if index % 2 = 0 then column index else extensionOps.zero),
       numericSum extensionOps ringDegree (fun index =>
        if index % 2 = 0 then extensionOps.zero else column index)) := by
  dsimp only
  let column : Nat → K := fun index =>
    if bounded : index < ringDegree then
      extensionOps.mul (weight ((block * ringDegree + index) / 2))
        (PiCCSWeightedBasis.dotK (basis.get ⟨index, bounded⟩) values.get)
    else extensionOps.zero
  change blockMoment basis weight block values =
    (numericSum extensionOps ringDegree (fun index =>
      if index % 2 = 0 then column index else extensionOps.zero),
     numericSum extensionOps ringDegree (fun index =>
      if index % 2 = 0 then extensionOps.zero else column index))
  by_cases empty : values = Vector.replicate ringDegree K.zero
  · subst values
    rw [blockMoment, if_pos rfl]
    have columnsZero : column = fun _ => extensionOps.zero := by
      funext index
      dsimp only [column]
      split <;> simp only [dot_zero, extensionLaws.mul_zero]
    have sumZero := paritySum_zero ringDegree
    rw [← columnsZero] at sumZero
    exact sumZero.symm.trans (paritySum_value ringDegree column)
  · rw [blockMoment, if_neg empty]
    exact paritySum_value ringDegree column

end NightstreamFPrime.Export.Stage1.PiCCSPadBlockMoment
