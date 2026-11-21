import NightstreamFPrime.Gadgets.Sampling.WideReduction
import NightstreamFPrime.Circuit.Execution

/-! The witness program of the five canonical input children. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range

theorem witnesses_childOps (interface : Interface) (offset : Nat) :
    witnesses (childOps interface offset) =
      (List.finRange fieldCount).flatMap fun index =>
        witnesses (CanonicalU64.operations (childInterface interface offset index)
          (childOffset offset index)) := by
  unfold witnesses childOps
  rw [List.flatMap_map]
  rfl

theorem ordered (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset) :
    Witness.Ordered offset (witnesses (childOps interface offset)) (quotientStart offset) := by
  rw [witnesses_childOps, quotientStart, Nat.mul_comm]
  refine Witness.Ordered.finRange_flatMap fun index =>
    (CanonicalU64.ordered (childInterface interface offset index) (childOffset offset index)
      (fun _ => 0) (Expr.VarsBelow.mono _ (inputs index) (by unfold childOffset; omega))).mono
      ?_ ?_ <;>
    simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount] <;> omega

theorem rows (interface : Interface) (env : Env) (offset : Nat)
    (computed : Witness.Computed env (witnesses (childOps interface offset))) :
    holdsFlat env (childOps interface offset) := by
  rw [witnesses_childOps, Witness.computed_flatMap] at computed
  intro expression member
  obtain ⟨operation, operationMember, member⟩ := List.mem_flatMap.mp member
  obtain ⟨index, _, rfl⟩ := List.mem_map.mp operationMember
  exact CanonicalU64.computed_holdsFlat _ env _ (computed index (List.mem_finRange index))
    expression member

theorem scope (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset) := by
  intro expression member
  obtain ⟨operation, operationMember, member⟩ := List.mem_flatMap.mp member
  obtain ⟨index, _, rfl⟩ := List.mem_map.mp operationMember
  have below := CanonicalU64.flatConstraints_varsBelow (childInterface interface offset index)
    (childOffset offset index) (Expr.VarsBelow.mono _ (inputs index) (by unfold childOffset; omega))
    expression member
  apply Expr.VarsBelow.mono _ below
  have indexBound := index.isLt
  simp only [childOffset, quotientStart, childWidth, CanonicalU64.auxiliaryCount, fieldCount] at *
  omega

theorem specifications (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (rows : holdsFlat env (childOps interface offset)) :
    ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index) (childOffset offset index)
      env := by
  intro index
  apply CanonicalU64.soundness
  · exact Expr.VarsBelow.mono _ (inputs index) (by unfold childOffset; omega)
  · apply holdsFlat_implies_holds
    intro expression member
    exact rows expression (List.mem_flatMap.mpr ⟨_, List.mem_map.mpr ⟨index, List.mem_finRange index, rfl⟩,
      member⟩)

end NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren
