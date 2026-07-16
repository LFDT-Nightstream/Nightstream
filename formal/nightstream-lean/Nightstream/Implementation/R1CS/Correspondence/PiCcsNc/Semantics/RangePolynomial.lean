import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.NormDischarged

/-!
Contract: model the production Π_CCS NC `b = 2` range factor on embedded
Goldilocks assignment coordinates.

Owns: the factored extension-field polynomial `(z + 1) * z * (z - 1)`, its
specialization to `K.embed(F)`, and the exact equivalence between vanishing of
that specialization and the authoritative strict `normBounded 2` predicate.

Does not own: arbitrary-`K` root classification, assignment packing, NC
mixing, SumCheck soundness, transcript challenges, production row lowering,
or permission to remove centered-alphabet rows.

Emits constraints: no. This file states mathematical semantics only.

Authority boundary: the list theorem quantifies the verifier-bound embedded
fresh assignment. A terminal NC claim, sampled evaluation, digest, or
unconnected extension-field value is not a substitute for that assignment.

| Predicate / theorem | Rust stage | Mathematical guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `rangeProductB2` | Rust row family `nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.range_products` | exact `(z + 1) * z * (z - 1)` factor | concrete `K` arithmetic | no |
| `rangeProductB2_embed` | same | embedded inputs stay in the base field | input is `K.embed(value)` | no |
| `rangeProductB2_embed_eq_zero_iff_centered` | same | roots are exactly `{-1, 0, 1}` | Goldilocks Euclid property | no |
| `rangeProductB2_embed_eq_zero_iff_normTwo` | same | one root check equals strict `b = 2` norm | embedded base-field input | no |
| `assignment_rangeProductB2_zero_iff_normBoundedTwo` | Π_CCS NC semantics | pointwise roots equal `normBounded 2` | exact fresh assignment ordering still external | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.SuperNeo.Concrete

/-- Production Π_CCS NC range factor specialized to `b = 2`.

The zero-set theorems below deliberately cover only values embedded from the
Goldilocks base field. -/
def rangeProductB2 (value : K) : K :=
  K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1))

/-- The factored extension-field computation preserves an embedded base-field
input and computes the same factor in `F`. -/
theorem rangeProductB2_embed (value : F) :
    rangeProductB2 (K.embed value) =
      K.embed ((value + 1) * value * (value - 1)) := by
  simp [rangeProductB2, K.add, K.mul, K.sub, K.embed,
    Fin.add_zero, Fin.mul_zero, Fin.zero_mul]

private theorem embed_eq_zero_iff (value : F) :
    K.embed value = K.zero ↔ value = 0 := by
  constructor
  · intro equal
    have low := congrArg (fun item : K => item.c0) equal
    simpa [K.embed, K.zero] using low
  · rintro rfl
    rfl

private theorem fmul_eq_zero
    (prime : EuclidPrime goldilocksP)
    {left right : F}
    (zero : left * right = 0) :
    left = 0 ∨ right = 0 := by
  have productZero : left.val * right.val % goldilocksP = 0 := by
    have values := congrArg Fin.val zero
    simpa [Fin.val_mul, goldilocksP, goldilocksModulus] using values
  rcases prime left.val right.val productZero with leftZero | rightZero
  · left
    apply Fin.eq_of_val_eq
    have leftLt : left.val < goldilocksP := by
      exact left.isLt
    simpa [Nat.mod_eq_of_lt leftLt] using leftZero
  · right
    apply Fin.eq_of_val_eq
    have rightLt : right.val < goldilocksP := by
      exact right.isLt
    simpa [Nat.mod_eq_of_lt rightLt] using rightZero

private theorem add_one_eq_zero_implies_negative
    {value : F} (zero : value + 1 = 0) :
    value.val = goldilocksP - 1 := by
  have modularZero : (value.val + 1) % goldilocksP = 0 := by
    have values := congrArg Fin.val zero
    simpa [Fin.val_add, goldilocksP, goldilocksModulus] using values
  have valueLt : value.val < goldilocksP := by
    exact value.isLt
  have valueSuccLe : value.val + 1 ≤ goldilocksP := by omega
  rcases Nat.lt_or_eq_of_le valueSuccLe with strict | equal
  · rw [Nat.mod_eq_of_lt strict] at modularZero
    omega
  · omega

/-- On embedded Goldilocks values, the production factor vanishes exactly on
the three centered residues. -/
theorem rangeProductB2_embed_eq_zero_iff_centered
    (prime : EuclidPrime goldilocksP) (value : F) :
    rangeProductB2 (K.embed value) = K.zero ↔
      CenteredResidue value.val := by
  rw [rangeProductB2_embed]
  rw [embed_eq_zero_iff]
  constructor
  · intro productZero
    rcases fmul_eq_zero prime productZero with leadingZero | trailingZero
    · rcases fmul_eq_zero prime leadingZero with negative | zero
      · exact Or.inl (add_one_eq_zero_implies_negative negative)
      · exact Or.inr (Or.inl (congrArg Fin.val zero))
    · have one : value = 1 :=
        Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp trailingZero
      exact Or.inr (Or.inr (congrArg Fin.val one))
  · intro centered
    rcases centered with negative | zero | one
    · have plusZero : value + 1 = 0 := by
        apply Fin.eq_of_val_eq
        simp only [Fin.val_add, negative, Fin.val_zero]
        change (goldilocksP - 1 + 1) % goldilocksP = 0
        have modulusIdentity :
            goldilocksP - 1 + 1 = goldilocksP := by
          simp [goldilocksP]
        rw [modulusIdentity, Nat.mod_self]
      rw [plusZero, Fin.zero_mul, Fin.zero_mul]
    · have valueEq : value = 0 := by exact Fin.ext zero
      rw [valueEq]
      simp only [Fin.mul_zero, Fin.zero_mul]
    · have valueEq : value = 1 := by exact Fin.ext one
      rw [valueEq]
      simp only [Fin.sub_self, Fin.mul_zero]

/-- Vanishing of the embedded production factor is exactly the concrete
SuperNeo strict norm window for one Goldilocks coordinate. -/
theorem rangeProductB2_embed_eq_zero_iff_normTwo
    (prime : EuclidPrime goldilocksP) (value : F) :
    rangeProductB2 (K.embed value) = K.zero ↔
      Nightstream.SuperNeo.Concrete.centeredMagnitude value < 2 := by
  rw [rangeProductB2_embed_eq_zero_iff_centered prime value]
  exact (concrete_norm_two_iff_centeredResidue value).symm

/-- Pointwise vanishing on the exact embedded assignment is equivalent to the
paper's authoritative `‖z‖∞ < 2` predicate. -/
theorem assignment_rangeProductB2_zero_iff_normBoundedTwo
    (prime : EuclidPrime goldilocksP) (assignment : List F) :
    (∀ value ∈ assignment,
      rangeProductB2 (K.embed value) = K.zero) ↔
      Nightstream.SuperNeo.Concrete.normBounded 2 assignment := by
  constructor <;> intro accepted value member
  · exact (rangeProductB2_embed_eq_zero_iff_normTwo prime value).mp
      (accepted value member)
  · exact (rangeProductB2_embed_eq_zero_iff_normTwo prime value).mpr
      (accepted value member)

end Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial
