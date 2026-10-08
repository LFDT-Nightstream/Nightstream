import Mathlib.Algebra.QuadraticAlgebra.Defs
import Mathlib.Algebra.Ring.TransferInstance
import NightstreamFPrime.Spec.GoldilocksExtension
import NightstreamFPrime.Spec.FieldTower

/-! Owns the Mathlib structure on the challenge field `K = F_q[U]/(U² − 7)`:
commutative ring, domain, finite type, and characteristic `q`, as one scoped
instance block, with the equations that match it to `K`'s own operations. A
consumer opens `NightstreamFPrime.Spec.GoldilocksExtensionRing`. -/

namespace NightstreamFPrime.Spec.GoldilocksExtensionRing

/-- `K` as Mathlib's quadratic algebra over `ZMod q`. -/
def carrierEquiv : K ≃ QuadraticAlgebra (ZMod goldilocksModulus) 7 0 where
  toFun value := ⟨value.c0, value.c1⟩
  invFun value := ⟨value.re, value.im⟩
  left_inv _ := rfl
  right_inv _ := rfl

scoped instance : CommRing K := carrierEquiv.commRing

theorem zero_eq : (0 : K) = K.zero := rfl

theorem one_eq : (1 : K) = K.one := rfl

theorem add_eq (left right : K) : left + right = K.add left right := rfl

theorem mul_eq (left right : K) : left * right = K.mul left right := by
  change K.mk _ (left.c0 * right.c1 + left.c1 * right.c0 + 0 * left.c1 * right.c1) = _
  simp only [Fin.zero_mul, Fin.add_zero]
  rfl

theorem neg_eq (value : K) : -value = ⟨-value.c0, -value.c1⟩ := rfl

theorem sub_eq (left right : K) : left - right = K.sub left right := by
  change K.mk _ _ = K.mk _ _
  simp only [sub_eq_add_neg]
  rfl

/-- A natural number in `K` is its residue in the first coordinate. -/
theorem natCast_eq (n : ℕ) :
    (n : K) = ⟨⟨n % goldilocksModulus, Nat.mod_lt _ (by decide)⟩, 0⟩ := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Nat.cast_succ, ih, add_eq, one_eq]
    simp only [K.add, K.one, K.mk.injEq, add_zero, and_true]
    apply Fin.ext
    simp [Fin.val_add, Nat.add_mod]

scoped instance : Nontrivial K := ⟨⟨K.zero, K.one, by
  intro same
  have : (0 : F) = 1 := congrArg K.c0 same
  exact (by decide : (0 : F) ≠ 1) this⟩⟩

scoped instance : NoZeroDivisors K where
  eq_zero_or_eq_zero_of_mul_eq_zero := by
    intro left right productZero
    exact GoldilocksExtension.extensionNoZeroDivisors left right
      (by simpa only [mul_eq, zero_eq] using productZero)

scoped instance : IsDomain K := NoZeroDivisors.to_isDomain K

/-- `K` is finite: one element per pair of coordinates. -/
@[reducible] noncomputable def kFintype : Fintype K := Fintype.ofEquiv (F × F) {
  toFun value := ⟨value.1, value.2⟩
  invFun value := (value.c0, value.c1)
  left_inv _ := rfl
  right_inv _ := rfl }

attribute [scoped instance] kFintype

scoped instance : CharP K goldilocksModulus := by
  haveI : CharP (QuadraticAlgebra (ZMod goldilocksModulus) 7 0) goldilocksModulus :=
    charP_of_injective_algebraMap QuadraticAlgebra.algebraMap_injective _
  exact charP_of_injective_ringHom (f := carrierEquiv.ringEquiv.symm.toRingHom)
    carrierEquiv.ringEquiv.symm.injective _

/-- `K` has `q²` elements. -/
theorem card_K : Fintype.card K = goldilocksModulus ^ 2 := by
  rw [← Nat.card_eq_fintype_card]
  exact FieldTower.extension_cardinality

end NightstreamFPrime.Spec.GoldilocksExtensionRing
