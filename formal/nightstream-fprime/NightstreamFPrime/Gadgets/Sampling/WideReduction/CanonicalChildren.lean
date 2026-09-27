import NightstreamFPrime.Gadgets.Sampling.WideReduction.Completeness

/-! Concrete execution of the four canonical input children. Each call
uses the existing proved hint-and-recipe construction. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range

def values (interface : Interface) (base : Env) (offset : Nat) : Nat → Env
  | 0 => base
  | count + 1 =>
      if inside : count < fieldCount then
        CanonicalU64.completeEnv (childInterface interface offset ⟨count, inside⟩)
          (values interface base offset count) (childOffset offset count)
      else values interface base offset count

theorem correct (interface : Interface) (base : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (count : Nat) (bound : count ≤ fieldCount) :
    AgreesOutside base (values interface base offset count) offset (childWidth * count) ∧
      ∀ index : Fin fieldCount, index.val < count →
        holdsFlat (values interface base offset count)
          (CanonicalU64.operations (childInterface interface offset index) (childOffset offset index)) := by
  induction count with
  | zero =>
      refine ⟨?_, ?_⟩
      · intro index _
        rfl
      · intro index impossible
        omega
  | succ count ih =>
      have inside : count < fieldCount := by omega
      have previous := ih (by omega)
      have sourceBelow : CanonicalU64.Assumptions
          (childInterface interface offset ⟨count, inside⟩) (childOffset offset count)
          (values interface base offset count) :=
        Expr.VarsBelow.mono _ (inputs ⟨count, inside⟩) (by unfold childOffset; omega)
      have child := CanonicalU64.completeEnv_correct _ _ _ sourceBelow
      rw [values, dif_pos inside]
      refine ⟨?_, ?_⟩
      · have agreement := previous.1.append child.1
        convert agreement using 1
        rw [Nat.mul_succ]
        rfl
      · intro index earlier
        by_cases last : index.val = count
        · have same : index = ⟨count, inside⟩ := Fin.ext last
          subst index
          exact child.2.1
        · apply constraintsHold_of_agree_below (values interface base offset count) _ _
            (childOffset offset index + CanonicalU64.auxiliaryCount)
          · exact CanonicalU64.flatConstraints_varsBelow _ _
              (Expr.VarsBelow.mono _ (inputs index) (by unfold childOffset; omega))
          · intro column below
            apply child.1 column (Or.inl ?_)
            simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount] at *
            omega
          · exact previous.2 index (by omega)

theorem rows (interface : Interface) (base : Env) (offset : Nat)
    (inputs : Assumptions interface offset) :
    holdsFlat (values interface base offset fieldCount) (childOps interface offset) := by
  have children := (correct interface base offset inputs fieldCount (by rfl)).2
  intro expression member
  obtain ⟨operation, operationMember, member⟩ := List.mem_flatMap.mp member
  obtain ⟨index, _, rfl⟩ := List.mem_map.mp operationMember
  exact children index index.isLt expression member

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

theorem specifications (interface : Interface) (base : Env) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index) (childOffset offset index)
      (values interface base offset fieldCount) := by
  intro index
  apply CanonicalU64.soundness
  · exact Expr.VarsBelow.mono _ (inputs index) (by unfold childOffset; omega)
  · exact holdsFlat_implies_holds _ _
      ((correct interface base offset inputs fieldCount (by rfl)).2 index index.isLt)

end NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren
