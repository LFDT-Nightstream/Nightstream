import NightstreamFPrime.Spec.Algebra

/-!
Paper-level `b = 2` norm range semantics for the joint `Pi_CCS` polynomial.

Owns: the exact base-field residual `(z + 1) * z * (z - 1)`, its zero set,
and its equivalence with the authoritative strict centered norm
`Concrete.centeredMagnitude z < 2`.

Does not own: the paper's inconsistent displayed product bounds, extension-
field `Q`, Boolean-table placement, SumCheck, Fiat--Shamir, Rust, R1CS rows,
or permission to remove constraints.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange

open NightstreamFPrime.Spec

def representedMinusOne : F :=
  ⟨goldilocksModulus - 1, by simp [goldilocksModulus]⟩

def representedRoots : List F :=
  [representedMinusOne, 0, 1]

def BaseFieldNoZeroDivisors : Prop :=
  ∀ left right : F, left * right = 0 → left = 0 ∨ right = 0

def GoldilocksModulusEuclid : Prop :=
  ∀ left right : Nat,
    left * right % goldilocksModulus = 0 →
      left % goldilocksModulus = 0 ∨ right % goldilocksModulus = 0

theorem baseFieldNoZeroDivisors_of_modulusEuclid
    (euclid : GoldilocksModulusEuclid) :
    BaseFieldNoZeroDivisors := by
  intro left right productZero
  have productValuesZero :
      left.val * right.val % goldilocksModulus = 0 := by
    have values := congrArg Fin.val productZero
    simpa [Fin.val_mul] using values
  rcases euclid left.val right.val productValuesZero with
    leftZero | rightZero
  · left
    apply Fin.eq_of_val_eq
    simpa [Nat.mod_eq_of_lt left.isLt] using leftZero
  · right
    apply Fin.eq_of_val_eq
    simpa [Nat.mod_eq_of_lt right.isLt] using rightZero

/-- Exact base-field factor selected by the semantic strict centered range
`|z| < 2`; no displayed paper product bound is consulted. -/
def cubicResidual (value : F) : F :=
  (value + 1) * value * (value - 1)

theorem representedRoots_values :
    representedRoots.map Fin.val =
      [goldilocksModulus - 1, 0, 1] := by
  rfl

theorem representedRoots_nodup :
    representedRoots.Nodup := by
  decide

private theorem add_one_eq_zero_implies_representedMinusOne
    {value : F}
    (zero : value + 1 = 0) :
    value = representedMinusOne := by
  apply Fin.eq_of_val_eq
  have modularZero :
      (value.val + 1) % goldilocksModulus = 0 := by
    have values := congrArg Fin.val zero
    simpa [Fin.val_add] using! values
  have valueSuccLe : value.val + 1 ≤ goldilocksModulus := by
    omega
  rcases Nat.lt_or_eq_of_le valueSuccLe with strict | equal
  · rw [Nat.mod_eq_of_lt strict] at modularZero
    omega
  · simpa [representedMinusOne] using! congrArg (fun n => n - 1) equal

theorem cubicResidual_eq_zero_iff_representedRoot
    (noZeroDivisors : BaseFieldNoZeroDivisors)
    (value : F) :
    cubicResidual value = 0 ↔
      value = representedMinusOne ∨ value = 0 ∨ value = 1 := by
  constructor
  · intro productZero
    rcases noZeroDivisors _ _ productZero with leadingZero | trailingZero
    · rcases noZeroDivisors _ _ leadingZero with negative | zero
      · exact Or.inl
          (add_one_eq_zero_implies_representedMinusOne negative)
      · exact Or.inr (Or.inl zero)
    · have one : value = 1 :=
        Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp trailingZero
      exact Or.inr (Or.inr one)
  · intro root
    rcases root with negative | zero | one
    · rw [negative]
      decide
    · rw [zero]
      rfl
    · rw [one]
      rfl

theorem strictNormTwo_iff_representedRoot (value : F) :
    centeredMagnitude value < 2 ↔
      value = representedMinusOne ∨ value = 0 ∨ value = 1 := by
  constructor
  · intro bounded
    unfold centeredMagnitude at bounded
    by_cases lowerHalf :
        value.val ≤ goldilocksModulus - value.val
    · rw [Nat.min_eq_left lowerHalf] at bounded
      rcases Nat.eq_zero_or_pos value.val with zero | positive
      · exact Or.inr (Or.inl (Fin.eq_of_val_eq zero))
      · have one : value.val = 1 := by omega
        exact Or.inr (Or.inr (Fin.eq_of_val_eq one))
    · have upperHalf :
          goldilocksModulus - value.val ≤ value.val := by
        omega
      rw [Nat.min_eq_right upperHalf] at bounded
      have differencePositive :
          0 < goldilocksModulus - value.val :=
        Nat.sub_pos_of_lt value.isLt
      have differenceOne :
          goldilocksModulus - value.val = 1 := by
        omega
      have valueIsMinusOne :
          value.val = goldilocksModulus - 1 := by
        omega
      exact Or.inl (Fin.eq_of_val_eq valueIsMinusOne)
  · intro root
    rcases root with negative | zero | one
    · rw [negative]
      decide
    · rw [zero]
      decide
    · rw [one]
      decide

theorem cubicResidual_eq_zero_iff_strictNormTwo
    (noZeroDivisors : BaseFieldNoZeroDivisors)
    (value : F) :
    cubicResidual value = 0 ↔ centeredMagnitude value < 2 := by
  rw [cubicResidual_eq_zero_iff_representedRoot noZeroDivisors value]
  exact (strictNormTwo_iff_representedRoot value).symm

theorem allCubicResidualsZero_iff_normBoundedTwo
    (noZeroDivisors : BaseFieldNoZeroDivisors)
    (assignment : List F) :
    (∀ value ∈ assignment, cubicResidual value = 0) ↔
      normBounded 2 assignment := by
  constructor <;> intro accepted value member
  · exact (cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors value).mp (accepted value member)
  · exact (cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors value).mpr (accepted value member)

theorem embed_cubicResidual (value : F) :
    K.mul (K.mul (K.add (K.embed value) (K.embed 1)) (K.embed value))
        (K.sub (K.embed value) (K.embed 1)) =
      K.embed (cubicResidual value) := by
  simp [cubicResidual, K.add, K.mul, K.sub, K.embed,
    Fin.add_zero, Fin.mul_zero, Fin.zero_mul]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange
