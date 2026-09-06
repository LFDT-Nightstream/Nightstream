import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Owns structural elimination of zero blocks from the exact Ajtai row sum.
The assignment still has the complete carrier domain. This module changes
neither the key nor the commitment relation and emits no circuit rows.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.CommitmentSparse

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open Commitment

private theorem zero_add (value : RingF) : ringFAdd ringFZero value = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

private theorem add_zero (value : RingF) : ringFAdd value ringFZero = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem sum_zero {count : Nat} :
    ringFSum (fun _ : Fin count => ringFZero) = ringFZero := by
  induction count with
  | zero => rfl
  | succ count previous => rw [ringFSum, previous, zero_add]

/-- A single supported summand is recovered by structural induction, without
evaluating an instantiated production-size finite domain. -/
theorem ringFSum_eq_single {count : Nat} (terms : Fin count → RingF)
    (selected : Fin count)
    (unsupported : ∀ index, index ≠ selected → terms index = ringFZero) :
    ringFSum terms = terms selected := by
  induction count with
  | zero => exact Fin.elim0 selected
  | succ count previous =>
      revert unsupported
      refine Fin.cases ?_ (fun selected => ?_) selected
      · intro unsupported
        have tail : (fun index : Fin count => terms index.succ) =
            (fun _ => ringFZero) := by
          funext index
          exact unsupported index.succ (Fin.succ_ne_zero index)
        rw [ringFSum, tail, sum_zero, add_zero]
      · intro unsupported
        have head : terms 0 = ringFZero :=
          unsupported 0 (Ne.symm (Fin.succ_ne_zero selected))
        have tail : ringFSum (fun index : Fin count => terms index.succ) =
            terms selected.succ := by
          apply previous (fun index => terms index.succ) selected
          intro index different
          apply unsupported index.succ
          intro equal
          exact different (Fin.succ_inj.mp equal)
        rw [ringFSum, head, tail, zero_add]

/-- One ring block at its original full-carrier address. -/
def singleBlock {shape : Shape}
    (selected : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (value : RingF) : Assignment shape :=
  fun column =>
    let packed := Phi81ColumnLayout.decode column
    if packed.1 = selected then value packed.2 else 0

theorem assignmentBlock_singleBlock {shape : Shape}
    (selected block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (value : RingF) :
    CarrierAction.assignmentBlock (singleBlock selected value) block =
      if block = selected then value else ringFZero := by
  funext lane
  have decoded : Phi81ColumnLayout.decode
      (CarrierAction.carrierColumn (logicalWidth := shape.logicalWidth) block lane) = (block, lane) :=
    Phi81CarrierLayout.decode_carrierColumn (logicalWidth := shape.logicalWidth) block lane
  dsimp only [CarrierAction.assignmentBlock, singleBlock]
  apply Eq.trans (congrArg
    (fun packed : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) × Fin ringDegree =>
      if packed.1 = selected then value packed.2 else (0 : F)) decoded)
  by_cases equal : block = selected <;> simp [equal, ringFZero]

/-- The exact original commitment of a single supported block uses only its
selected key element. No equality of commitments is an input premise. -/
theorem commit_singleBlock {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows)
    (selected : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (value : RingF) :
    commit key (singleBlock selected value) =
      fun row => ringFMul (key row selected) value := by
  funext row
  unfold commit ajtaiRow blockSum
  rw [ringFSum_eq_single _ selected]
  · rw [assignmentBlock_singleBlock, if_pos rfl]
  · intro block different
    rw [assignmentBlock_singleBlock, if_neg different]
    exact CarrierAction.ringFMul_zero_right _

/-- A one-coefficient block occupies precisely the canonical flat address,
including positions in the final complete carrier block. -/
theorem singleBlock_monomial_coordinate {shape : Shape}
    (selected : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) (scalar : F)
    (column : Fin shape.carrierWidth) :
    singleBlock selected (ringFMonomial lane.val scalar) column =
      if column = Phi81CarrierLayout.carrierColumn (logicalWidth := shape.logicalWidth) selected lane then scalar else 0 := by
  by_cases same : column = Phi81CarrierLayout.carrierColumn (logicalWidth := shape.logicalWidth) selected lane
  · subst column
    simp only [if_pos rfl]
    change CarrierAction.assignmentBlock
      (singleBlock selected (ringFMonomial lane.val scalar)) selected lane = scalar
    rw [assignmentBlock_singleBlock, if_pos rfl]
    simp [ringFMonomial]
  · have different : Phi81ColumnLayout.decode column ≠ (selected, lane) := by
      intro packed
      apply same
      apply Fin.ext
      have flattened := Phi81ColumnLayout.flatIndex_decode column
      rw [packed] at flattened
      exact flattened.symm
    by_cases block : (Phi81ColumnLayout.decode column).1 = selected
    · have coefficient : (Phi81ColumnLayout.decode column).2.val ≠ lane.val := by
        intro equal
        exact different (Prod.ext block (Fin.ext equal))
      simp [singleBlock, block, ringFMonomial, coefficient, same]
    · simp [singleBlock, block, same]

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.CommitmentSparse
