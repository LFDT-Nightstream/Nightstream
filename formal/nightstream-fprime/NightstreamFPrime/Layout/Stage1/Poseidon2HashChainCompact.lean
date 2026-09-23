import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1Prefix

/-! Three compact permutations and four digest pins for the exact application hash. -/

namespace NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompact

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.Stage1

structure Interface (columns : Nat) where
  oneColumn : Fin columns
  priorState : Fin 4 → SparseForm columns
  message : Fin 4 → SparseForm columns
  digest : Fin 4 → SparseForm columns
  sbox : Fin 3 → Fin PoseidonRetainedSlots.rows.length → SparseForm columns

def output {columns : Nat} (interface : Interface columns) (invocation : Fin 3) :
    SparseLayer.State columns :=
  SparseLayer.external fun lane => interface.sbox invocation (PoseidonRetainedSlots.finalRow lane)

def blockLane {columns : Nat} (block : Fin 4 → SparseForm columns) (lane : Fin 8) :
    SparseForm columns :=
  if bound : lane.val < 4 then block ⟨lane.val, bound⟩ else .empty

def input {columns : Nat} (interface : Interface columns) (invocation : Fin 3) :
    SparseLayer.State columns := fun lane =>
  if invocation.val = 0 then
    SparseForm.add
      (SparseLayer.constant interface.oneColumn
        (Poseidon2HashChainV1Prefix.constantState.getD lane.val 0))
      (blockLane interface.priorState lane)
  else if invocation.val = 1 then
    SparseForm.add (output interface 0 lane) (blockLane interface.message lane)
  else if lane.val = 0 then
    SparseLayer.addConstant interface.oneColumn (output interface 1 lane) 1
  else output interface 1 lane

def family {columns : Nat} (interface : Interface columns) :
    PoseidonSboxFamilyPlan.Interface columns 3 where
  oneColumn := interface.oneColumn
  input := input interface
  sboxOutput := interface.sbox

def pins {columns : Nat} (interface : Interface columns) :
    PinFamilyPlan.Interface columns 4 where
  oneColumn := interface.oneColumn
  value := fun lane => SparseForm.add (interface.digest lane)
    (SparseForm.scale (-1) (output interface 2 ⟨lane.val, by have := lane.isLt; omega⟩))

def plan {columns : Nat} (interface : Interface columns) : ProductionRelation.Plan columns :=
  Plan.append (PoseidonSboxFamilyPlan.plan (family interface) (by decide))
    (PinFamilyPlan.plan (pins interface) (by decide)) (by
      change 258 + 4 ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables
      decide)

theorem plan_rowCount {columns : Nat} (interface : Interface columns) :
    (plan interface).rowCount = 262 := by rfl

theorem private_coordinate_count : (4 + 3 * 86) * 41 = 10742 := by decide

private theorem ofFn_eight {α : Type} (f : Fin 8 → α) :
    List.ofFn f = [f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7] := by
  simp [List.ofFn_succ]

private theorem ofFn_four {α : Type} (f : Fin 4 → α) :
    List.ofFn f = [f 0, f 1, f 2, f 3] := by
  simp [List.ofFn_succ]

private theorem first_input {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1) :
    List.ofFn (SparseLayer.evalState assignment (input interface 0)) =
      (List.range Poseidon2.width).map (fun index =>
        Poseidon2HashChainV1Prefix.constantState.getD index 0 +
          (List.ofFn fun lane => (interface.priorState lane).eval assignment).getD index 0) := by
  rw [ofFn_eight, ofFn_four]
  simp [SparseLayer.evalState, input, blockLane, SparseForm.add_eval,
    SparseLayer.eval_constant assignment interface.oneColumn one,
    Poseidon2.width, List.range_succ]

private theorem second_input {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) :
    List.ofFn (SparseLayer.evalState assignment (input interface 1)) =
      (List.range Poseidon2.width).map (fun index =>
        (List.ofFn (SparseLayer.evalState assignment (output interface 0))).getD index 0 +
          (List.ofFn fun lane => (interface.message lane).eval assignment).getD index 0) := by
  rw [ofFn_eight, ofFn_eight, ofFn_four]
  simp [SparseLayer.evalState, input, blockLane, Poseidon2.width, List.range_succ]

private theorem padding_input {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1) :
    List.ofFn (SparseLayer.evalState assignment (input interface 2)) =
      (List.range Poseidon2.width).map (fun index =>
        if index = 0 then
          (List.ofFn (SparseLayer.evalState assignment (output interface 1))).getD 0 0 + 1
        else (List.ofFn (SparseLayer.evalState assignment (output interface 1))).getD index 0) := by
  rw [ofFn_eight, ofFn_eight]
  simp [SparseLayer.evalState, input,
    SparseLayer.eval_addConstant assignment interface.oneColumn one,
    Poseidon2.width, List.range_succ]

/-- The compact permutation family produces the exact application digest. -/
theorem output_eq_step {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (permutations : (PoseidonSboxFamilyPlan.plan (family interface) (by decide)).RowsZero assignment) :
    (List.ofFn (SparseLayer.evalState assignment (output interface 2))).take 4 =
      Poseidon2HashChainV1.step
        (List.ofFn fun lane => (interface.priorState lane).eval assignment)
        (List.ofFn fun lane => (interface.message lane).eval assignment) := by
  have permutes := PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (family interface) (by decide) assignment one permutations
  have first := permutes 0
  have second := permutes 1
  have final := permutes 2
  change List.ofFn (SparseLayer.evalState assignment (output interface 0)) =
    Poseidon2.permute (List.ofFn (SparseLayer.evalState assignment (input interface 0))) at first
  change List.ofFn (SparseLayer.evalState assignment (output interface 1)) =
    Poseidon2.permute (List.ofFn (SparseLayer.evalState assignment (input interface 1))) at second
  change List.ofFn (SparseLayer.evalState assignment (output interface 2)) =
    Poseidon2.permute (List.ofFn (SparseLayer.evalState assignment (input interface 2))) at final
  rw [first_input interface assignment one] at first
  rw [second_input, first] at second
  rw [padding_input interface assignment one, second] at final
  rw [final]
  exact Poseidon2HashChainV1Prefix.threePermutations_eq_step _ _
    (List.length_ofFn) (List.length_ofFn)

/-- Every satisfying assignment gives the unchanged application transition. -/
theorem soundness {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan interface).RowsZero assignment) :
    (List.ofFn fun lane => (interface.digest lane).eval assignment) =
      Poseidon2HashChainV1.step
        (List.ofFn fun lane => (interface.priorState lane).eval assignment)
        (List.ofFn fun lane => (interface.message lane).eval assignment) := by
  obtain ⟨permutations, bindings⟩ := (Plan.append_rowsZero_iff _ _ _ assignment).mp rows
  have pinned := (PinFamilyPlan.planRowsZero_iff (pins interface) (by decide) assignment one).mp bindings
  have digestEq : (List.ofFn fun lane => (interface.digest lane).eval assignment) =
      (List.ofFn (SparseLayer.evalState assignment (output interface 2))).take 4 := by
    have lanes : ∀ lane : Fin 4, (interface.digest lane).eval assignment =
        (output interface 2 ⟨lane.val, by have := lane.isLt; omega⟩).eval assignment := by
      intro lane
      have zero := pinned lane
      simp only [pins, SparseForm.add_eval, SparseForm.scale_eval] at zero
      exact sub_eq_zero.mp (by simpa only [neg_one_mul, sub_eq_add_neg] using zero)
    rw [ofFn_four, ofFn_eight]
    simp only [List.take_succ_cons, List.take_zero]
    simp only [lanes, SparseLayer.evalState]
    rfl
  rw [digestEq]
  exact output_eq_step interface assignment one permutations

end NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompact
