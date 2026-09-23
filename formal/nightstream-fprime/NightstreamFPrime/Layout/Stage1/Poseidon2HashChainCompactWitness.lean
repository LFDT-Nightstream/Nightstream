import NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompact
import NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness

/-! Construct the 258 retained application S-box values and prove their encoding. -/

namespace NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompactWitness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.Stage1
open Poseidon2HashChainCompact

def blockValue (block : Fin 4 → F) (lane : Fin 8) : F :=
  if bound : lane.val < 4 then block ⟨lane.val, bound⟩ else 0

def firstInput (priorState : Fin 4 → F) : Fin 8 → F := fun lane =>
  Poseidon2HashChainV1Prefix.constantState.getD lane.val 0 + blockValue priorState lane

def secondInput (priorState message : Fin 4 → F) : Fin 8 → F := fun lane =>
  PoseidonCompactWitness.output (firstInput priorState) lane + blockValue message lane

def finalInput (priorState message : Fin 4 → F) : Fin 8 → F := fun lane =>
  let value := PoseidonCompactWitness.output (secondInput priorState message) lane
  if lane.val = 0 then value + 1 else value

def permutationInput (priorState message : Fin 4 → F) (invocation : Fin 3) : Fin 8 → F :=
  if invocation.val = 0 then firstInput priorState
  else if invocation.val = 1 then secondInput priorState message
  else finalInput priorState message

/-- The concrete field values that the compact low-norm block must encode. -/
def witness (priorState message : Fin 4 → F) (invocation : Fin 3)
    (row : Fin PoseidonRetainedSlots.rows.length) : F :=
  PoseidonCompactWitness.retained (permutationInput priorState message invocation) row

/-- Encoding the constructed fields completes every application row for each
valid application transition. No S-box equation is assumed as input. -/
theorem complete_of_encoding {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (priorState message : Fin 4 → F)
    (one : assignment interface.oneColumn = 1)
    (prior : ∀ lane, (interface.priorState lane).eval assignment = priorState lane)
    (messages : ∀ lane, (interface.message lane).eval assignment = message lane)
    (digest : (List.ofFn fun lane => (interface.digest lane).eval assignment) =
      Poseidon2HashChainV1.step (List.ofFn priorState) (List.ofFn message))
    (encoded : ∀ invocation row, (interface.sbox invocation row).eval assignment =
      witness priorState message invocation row) :
    (plan interface).RowsZero assignment := by
  have first := PoseidonCompactWitness.family_member (family interface) 0 assignment
    (firstInput priorState) one (by
      funext lane
      by_cases bound : lane.val < 4 <;>
        simp [family, input, SparseLayer.evalState, firstInput, blockLane, blockValue, bound,
          SparseLayer.eval_constant assignment interface.oneColumn one, prior])
    (encoded 0)
  have firstOutput : SparseLayer.evalState assignment (output interface 0) =
      PoseidonCompactWitness.output (firstInput priorState) := first.2
  have second := PoseidonCompactWitness.family_member (family interface) 1 assignment
    (secondInput priorState message) one (by
      funext lane
      change ((output interface 0 lane).add (blockLane interface.message lane)).eval assignment = _
      rw [SparseForm.add_eval]
      rw [show (output interface 0 lane).eval assignment = _ from congrFun firstOutput lane]
      by_cases bound : lane.val < 4 <;>
        simp [secondInput, blockLane, blockValue, bound, messages])
    (encoded 1)
  have secondOutput : SparseLayer.evalState assignment (output interface 1) =
      PoseidonCompactWitness.output (secondInput priorState message) := second.2
  have final := PoseidonCompactWitness.family_member (family interface) 2 assignment
    (finalInput priorState message) one (by
      funext lane
      change (if lane.val = 0 then
        SparseLayer.addConstant interface.oneColumn (output interface 1 lane) 1
        else output interface 1 lane).eval assignment = finalInput priorState message lane
      by_cases zero : lane.val = 0
      · rw [if_pos zero, SparseLayer.eval_addConstant assignment interface.oneColumn one]
        rw [show (output interface 1 lane).eval assignment = _ from congrFun secondOutput lane]
        simp only [finalInput, zero, ↓reduceIte]
      · rw [if_neg zero]
        rw [show (output interface 1 lane).eval assignment = _ from congrFun secondOutput lane]
        simp only [finalInput, if_neg zero])
    (encoded 2)
  have familyRows : (PoseidonSboxFamilyPlan.plan (family interface) (by decide)).RowsZero assignment := by
    apply (PoseidonSboxFamilyPlan.planRowsZero_iff _ _ assignment).mpr
    intro invocation
    fin_cases invocation
    · exact first.1
    · exact second.1
    · exact final.1
  apply (Plan.append_rowsZero_iff _ _ _ assignment).mpr
  refine ⟨familyRows, ?_⟩
  apply (PinFamilyPlan.planRowsZero_iff _ _ assignment one).mpr
  have computed := output_eq_step interface assignment one familyRows
  have same : (List.ofFn fun lane => (interface.digest lane).eval assignment) =
      (List.ofFn (SparseLayer.evalState assignment (output interface 2))).take 4 := by
    rw [computed]
    simpa only [prior, messages] using digest
  have takeEq :
      (List.ofFn (SparseLayer.evalState assignment (output interface 2))).take 4 =
        List.ofFn (fun lane : Fin 4 =>
          (output interface 2 ⟨lane.val, by have := lane.isLt; omega⟩).eval assignment) := by
    simp [List.ofFn_succ, SparseLayer.evalState]
  have equal := List.ofFn_inj.mp (same.trans takeEq)
  intro lane
  have observed := congrFun equal lane
  simp only [pins, SparseForm.add_eval, SparseForm.scale_eval, neg_one_mul]
  simpa only [sub_eq_add_neg] using sub_eq_zero.mpr observed

end NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompactWitness
