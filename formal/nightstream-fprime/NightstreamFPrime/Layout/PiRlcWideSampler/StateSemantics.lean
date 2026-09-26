import NightstreamFPrime.Layout.PiRlcWideSampler.BatchSemantics

/-! Exact one-window transcript semantics of the candidate CCS Poseidon
schedule. The input is the caller's previous transcript state. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.StateSemantics

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation BatchPlan BatchSemantics
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev reference := Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt
abbrev enter := Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter

def state {columns : Nat} (assignment : Assignment F columns) (forms : PoseidonSboxPlan.State columns) :
    Poseidon2.State := List.ofFn (SparseLayer.evalState assignment forms)

def boundary {columns : Nat} (interface : Interface columns) (count : Nat) : PoseidonSboxPlan.State columns :=
  if bounded : 0 < count ∧ count ≤ 17 then
    outputState interface ⟨2 * count - 1, by omega⟩
  else interface.initialState

theorem boundary_zero {columns : Nat} (interface : Interface columns) :
    boundary interface 0 = interface.initialState := by simp [boundary]

theorem boundary_succ {columns : Nat} (interface : Interface columns) (source : Fin 17) :
    boundary interface (source.val + 1) = outputState interface ⟨source.val * 2 + 1, by omega⟩ := by
  rw [boundary, dif_pos (by constructor <;> omega)]
  congr 1
  apply Fin.ext
  dsimp only
  omega

theorem prior_entry {columns : Nat} (interface : Interface columns) (source : Fin 17) :
    priorState interface ⟨source.val * 2, by omega⟩ = boundary interface source.val := by
  by_cases first : source.val = 0
  · simp [priorState, boundary, first]
  · rw [priorState, dif_neg (by simp only; omega), boundary, dif_pos (by constructor <;> omega)]
    congr 1
    apply Fin.ext
    dsimp only
    omega

theorem prior_advance {columns : Nat} (interface : Interface columns) (source : Fin 17) :
    priorState interface ⟨source.val * 2 + 1, by omega⟩ = outputState interface ⟨source.val * 2, by omega⟩ := by
  rw [priorState, dif_neg (by simp only; omega)]
  congr 1

theorem enter_ofFn (values : Fin 8 → F) (source : Nat) :
    Poseidon2.permute (List.ofFn (fun lane => values lane + entryWord source lane)) =
      enter (List.ofFn values) source := by
  unfold enter Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter Poseidon2.absorbBlock
  apply congrArg Poseidon2.permute
  simp [Poseidon2.width, List.range_succ, List.ofFn_succ, entryWord]

theorem entered_state {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.oneColumn = 1) (rows : (poseidonPlan interface).RowsZero assignment)
    (source : Fin 17) :
    state assignment (outputState interface ⟨source.val * 2, by omega⟩) =
      enter (state assignment (boundary interface source.val)) source.val := by
  have permutation := PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (poseidonInterface interface) (by decide) assignment one rows ⟨source.val * 2, by omega⟩
  change state assignment (outputState interface _) = Poseidon2.permute _ at permutation
  rw [permutation]
  unfold state
  rw [← enter_ofFn]
  apply congrArg Poseidon2.permute
  apply congrArg List.ofFn
  funext lane
  simp only [poseidonInterface, SparseLayer.evalState]
  rw [if_pos (Nat.mul_mod_left _ _)]
  rw [SparseLayer.addConstant, SparseLayer.add, SparseForm.add_eval,
    SparseLayer.constant, SparseForm.singleton_eval, one, mul_one, prior_entry]
  rw [show source.val * 2 / 2 = source.val by omega]

theorem advanced_state {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.oneColumn = 1) (rows : (poseidonPlan interface).RowsZero assignment)
    (source : Fin 17) :
    state assignment (boundary interface (source.val + 1)) =
      Poseidon2.permute (state assignment (outputState interface ⟨source.val * 2, by omega⟩)) := by
  rw [boundary_succ]
  have permutation := PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (poseidonInterface interface) (by decide) assignment one rows ⟨source.val * 2 + 1, by omega⟩
  change state assignment (outputState interface _) = Poseidon2.permute _ at permutation
  rw [permutation]
  apply congrArg Poseidon2.permute
  apply congrArg List.ofFn
  funext lane
  simp only [poseidonInterface, SparseLayer.evalState]
  rw [if_neg (by omega), prior_advance]

theorem boundary_exact {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.oneColumn = 1) (rows : (poseidonPlan interface).RowsZero assignment)
    (count : Nat) (bounded : count ≤ 17) :
    state assignment (boundary interface count) = reference (state assignment interface.initialState) count := by
  induction count with
  | zero => rw [boundary_zero]; rfl
  | succ count ih =>
      have step := advanced_state interface assignment one rows ⟨count, by omega⟩
      rw [entered_state interface assignment one rows, ih (by omega)] at step
      exact step

theorem range_draw {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (source : Fin 17) :
    NightstreamFPrime.Gadgets.Sampling.WideReduction.drawOf RangePlan.interface
      (rangeEnv interface assignment source) 1408 =
      Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.block
        (state assignment (outputState interface ⟨source.val * 2, by omega⟩)) := by
  funext lane
  have bounded : lane.val < 4 := lane.isLt
  change rangeEnv interface assignment source lane.val = _
  unfold rangeEnv
  rw [SourceCompiler.sourceEnv_at _ ⟨lane.val, by omega⟩]
  simp only [rangeSource, Retained.sourceMap, dif_pos bounded]
  change (outputState interface _ _).eval assignment =
    (List.ofFn (SparseLayer.evalState assignment (outputState interface _))).getD lane.val 0
  rw [List.getD_eq_get _ _ ⟨lane.val, by simp only [List.length_ofFn]; omega⟩, List.get_ofFn]
  rfl

/-- Every accepted candidate assignment samples the prescribed scalar from
exactly the prescribed one-window transcript schedule. -/
theorem sampled_digits {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment) (source : Fin 17)
    (position : Fin 54) :
    NightstreamFPrime.Gadgets.Sampling.WideReduction.digitValue
      (rangeEnv interface assignment source) 1408 position.val =
        (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt
          (state assignment interface.initialState) source.val position).val := by
  obtain ⟨permutations, reductions⟩ := (rowsZero_iff compiled interface assignment).mp rows
  have checked := range_sound compiled interface assignment one source
    ((rangeFamily_zero_iff compiled interface assignment).mp reductions source) position
  rw [range_draw, entered_state interface assignment one permutations,
    boundary_exact interface assignment one permutations source.val source.isLt.le] at checked
  exact checked

end NightstreamFPrime.Layout.PiRlcWideSampler.StateSemantics
