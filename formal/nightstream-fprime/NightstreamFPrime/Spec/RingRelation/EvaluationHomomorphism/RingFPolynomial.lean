import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring
import Mathlib.RingTheory.AdjoinRoot
import NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFLaws

namespace NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFPolynomial

open NightstreamFPrime.Spec
open Polynomial

attribute [local implicit_reducible] ZMod NightstreamFPrime.Spec.fieldModulus

local instance : Fact (1 < fieldModulus) := ⟨by decide⟩

private abbrev Base := ZMod fieldModulus

local instance : Nontrivial Base := ZMod.nontrivial fieldModulus

noncomputable def modulus : Polynomial Base :=
  X ^ ringDegree + 1

noncomputable def toPolynomial (value : RingF) : Polynomial Base :=
  ∑ index : Fin ringDegree, C (R := Base) (value index) * X ^ index.val

theorem coeff_toPolynomial (value : RingF) (index : Fin ringDegree) :
    (toPolynomial value).coeff index.val = value index := by
  classical
  simp only [toPolynomial, finsetSum_coeff, C_mul_X_pow_eq_monomial,
    coeff_monomial]
  rw [Finset.sum_eq_single index]
  · simp
  · intro other _ different
    exact ite_eq_right (fun equal => different (Fin.ext equal))
  · simp

private theorem coeff_toPolynomial_of_ge (value : RingF) (index : Nat)
    (outside : ringDegree ≤ index) : (toPolynomial value).coeff index = 0 := by
  classical
  simp only [toPolynomial, finsetSum_coeff, C_mul_X_pow_eq_monomial,
    coeff_monomial]
  apply Finset.sum_eq_zero
  intro other _
  exact ite_eq_right (by have := other.isLt; omega)

theorem toPolynomial_injective : Function.Injective toPolynomial := by
  intro left right equal
  funext index
  have atIndex := congrArg (fun p => p.coeff index.val) equal
  simpa only [coeff_toPolynomial] using! atIndex

theorem toPolynomial_zero : toPolynomial ringFZero = 0 := by
  ext index
  rw [coeff_zero]
  by_cases inside : index < ringDegree
  · rw [coeff_toPolynomial ringFZero ⟨index, inside⟩]
    rfl
  · rw [coeff_toPolynomial_of_ge ringFZero index (Nat.le_of_not_gt inside)]

private theorem toPolynomial_basis (degree : Nat) (inside : degree < ringDegree) :
    toPolynomial (RingFLaws.basis degree) = X ^ degree := by
  ext index
  by_cases indexInside : index < ringDegree
  · rw [coeff_toPolynomial _ ⟨index, indexInside⟩]
    by_cases same : index = degree
    · subst index
      simp only [RingFLaws.basis, ringFMonomial, coeff_X_pow, ↓reduceIte]
    · simp only [RingFLaws.basis, ringFMonomial, coeff_X_pow, same,
        ↓reduceIte]
  · rw [coeff_toPolynomial_of_ge _ index (Nat.le_of_not_gt indexInside)]
    symm
    apply coeff_eq_zero_of_degree_lt
    rw [degree_X_pow]
    exact_mod_cast (show degree < index by omega)

theorem toPolynomial_one : toPolynomial ringFOne = 1 := by
  change toPolynomial (RingFLaws.basis 0) = 1
  rw [toPolynomial_basis 0 (by decide), pow_zero]

private theorem toPolynomial_add (left right : RingF) :
    toPolynomial (ringFAdd left right) = toPolynomial left + toPolynomial right := by
  ext index
  rw [coeff_add]
  by_cases inside : index < ringDegree
  · rw [coeff_toPolynomial _ ⟨index, inside⟩,
      coeff_toPolynomial _ ⟨index, inside⟩, coeff_toPolynomial _ ⟨index, inside⟩]
    rfl
  · have outside := Nat.le_of_not_gt inside
    rw [coeff_toPolynomial_of_ge _ index outside,
      coeff_toPolynomial_of_ge _ index outside, coeff_toPolynomial_of_ge _ index outside,
      add_zero]

private theorem toPolynomial_scale (scalar : F) (value : RingF) :
    toPolynomial (CarrierAction.ringFScale scalar value) =
      C (R := Base) scalar * toPolynomial value := by
  ext index
  rw [coeff_C_mul]
  by_cases inside : index < ringDegree
  · rw [coeff_toPolynomial _ ⟨index, inside⟩, coeff_toPolynomial _ ⟨index, inside⟩]
    rfl
  · have outside := Nat.le_of_not_gt inside
    rw [coeff_toPolynomial_of_ge _ index outside,
      coeff_toPolynomial_of_ge _ index outside, mul_zero]

private theorem degree_one_lt :
    (1 : Polynomial Base).degree < (ringDegree : WithBot Nat) := by
  rw [degree_one]
  exact_mod_cast (show 0 < ringDegree by decide)

theorem modulus_monic : modulus.Monic :=
  monic_X_pow_add degree_one_lt

theorem degree_modulus : modulus.degree = (ringDegree : WithBot Nat) := by
  rw [modulus, degree_add_eq_left_of_degree_lt, degree_X_pow]
  rw [degree_X_pow]
  exact degree_one_lt

private theorem degree_toPolynomial (value : RingF) :
    (toPolynomial value).degree < (ringDegree : WithBot Nat) :=
  degree_sum_fin_lt (R := Base) value

private theorem toPolynomial_mod (value : RingF) :
    toPolynomial value %ₘ modulus = toPolynomial value := by
  apply (modByMonic_eq_self_iff modulus_monic).mpr
  rw [degree_modulus]
  exact degree_toPolynomial value

abbrev QuotientRing := AdjoinRoot modulus

noncomputable def image (value : RingF) : QuotientRing :=
  AdjoinRoot.mk modulus (toPolynomial value)

private theorem image_zero : image ringFZero = 0 := by
  rw [image, toPolynomial_zero, map_zero]

private theorem image_add (left right : RingF) :
    image (ringFAdd left right) = image left + image right := by
  rw [image, toPolynomial_add, map_add]
  rfl

private theorem image_scale (scalar : F) (value : RingF) :
    image (CarrierAction.ringFScale scalar value) =
      AdjoinRoot.of modulus (scalar : Base) * image value := by
  rw [image, toPolynomial_scale, map_mul, AdjoinRoot.mk_C]
  rfl

private theorem image_basis (degree : Nat) (inside : degree < ringDegree) :
    image (RingFLaws.basis degree) = AdjoinRoot.root modulus ^ degree := by
  rw [image, toPolynomial_basis degree inside, map_pow, AdjoinRoot.mk_X]

private theorem root_relation :
    AdjoinRoot.root modulus ^ ringDegree = -1 := by
  have relation := AdjoinRoot.mk_self (f := modulus)
  rw [modulus, map_add, map_pow, map_one, AdjoinRoot.mk_X] at relation
  exact eq_neg_of_add_eq_zero_left relation

private theorem root_period : AdjoinRoot.root modulus ^ (2 * ringDegree) = 1 := by
  rw [mul_comm, pow_mul, root_relation, neg_one_sq]

theorem root_pow_mod (degree : Nat) :
    AdjoinRoot.root modulus ^ degree =
      AdjoinRoot.root modulus ^ (degree % (2 * ringDegree)) := by
  conv_lhs => rw [← Nat.mod_add_div degree (2 * ringDegree)]
  rw [pow_add, pow_mul, root_period, one_pow, mul_one]

private theorem image_monomialReduce (degree : Nat) :
    image (RingFLaws.monomialReduce degree) = AdjoinRoot.root modulus ^ degree := by
  rw [root_pow_mod]
  have residual : degree % (2 * ringDegree) < 2 * ringDegree := Nat.mod_lt _ (by decide)
  unfold RingFLaws.monomialReduce
  dsimp only
  split_ifs with low
  · exact image_basis _ low
  · rw [image_scale, image_basis _ (by omega)]
    change AdjoinRoot.of modulus (-1 : Base) *
      AdjoinRoot.root modulus ^ (degree % (2 * ringDegree) - ringDegree) = _
    have highPower : AdjoinRoot.root modulus ^ (degree % (2 * ringDegree)) =
        AdjoinRoot.root modulus ^ (degree % (2 * ringDegree) - ringDegree) *
          AdjoinRoot.root modulus ^ ringDegree := by
      rw [← pow_add]
      congr 1
      omega
    rw [map_neg, map_one, highPower, root_relation]
    ring

private theorem linear_image_eq
    (leftMap rightMap : RingF → QuotientRing)
    (leftZero : leftMap ringFZero = 0) (rightZero : rightMap ringFZero = 0)
    (leftAdd : ∀ left right, leftMap (ringFAdd left right) = leftMap left + leftMap right)
    (rightAdd : ∀ left right, rightMap (ringFAdd left right) = rightMap left + rightMap right)
    (leftScale : ∀ scalar value, leftMap (CarrierAction.ringFScale scalar value) =
      AdjoinRoot.of modulus (scalar : Base) * leftMap value)
    (rightScale : ∀ scalar value, rightMap (CarrierAction.ringFScale scalar value) =
      AdjoinRoot.of modulus (scalar : Base) * rightMap value)
    (onBasis : ∀ index : Fin ringDegree,
      leftMap (RingFLaws.basis index.val) = rightMap (RingFLaws.basis index.val))
    (value : RingF) : leftMap value = rightMap value := by
  classical
  let piece (index : Fin ringDegree) : RingF :=
    CarrierAction.ringFScale (value index) (RingFLaws.basis index.val)
  have expansion : (∑ index : Fin ringDegree, piece index) = value := by
    funext column
    simp only [Finset.sum_apply]
    change (∑ index : Fin ringDegree,
      (value index : Base) * (if column.val = index.val then 1 else 0)) = value column
    rw [Finset.sum_eq_single column]
    · simp
    · intro other _ different
      rw [ite_eq_right (fun equal => different (Fin.ext equal.symm)), mul_zero]
    · simp
  let leftHom : RingF →+ QuotientRing :=
    { toFun := leftMap, map_zero' := leftZero, map_add' := leftAdd }
  let rightHom : RingF →+ QuotientRing :=
    { toFun := rightMap, map_zero' := rightZero, map_add' := rightAdd }
  change leftHom value = rightHom value
  rw [← expansion, map_sum, map_sum]
  apply Finset.sum_congr rfl
  intro index _
  change leftMap (CarrierAction.ringFScale (value index) (RingFLaws.basis index.val)) =
    rightMap (CarrierAction.ringFScale (value index) (RingFLaws.basis index.val))
  rw [leftScale, rightScale, onBasis]

private theorem image_mul_basis (index : Fin ringDegree) (value : RingF) :
    image (ringFMul (RingFLaws.basis index.val) value) =
      image (RingFLaws.basis index.val) * image value := by
  apply linear_image_eq
    (fun value => image (ringFMul (RingFLaws.basis index.val) value))
    (fun value => image (RingFLaws.basis index.val) * image value)
  · rw [CarrierAction.ringFMul_zero_right, image_zero]
  · rw [image_zero, mul_zero]
  · intro left right
    rw [CarrierAction.ringFMul_add_right, image_add]
  · intro left right
    rw [image_add, mul_add]
  · intro scalar value
    rw [CarrierAction.ringFMul_scale_right, image_scale]
  · intro scalar value
    rw [image_scale]
    ring
  · intro other
    rw [RingFLaws.ringFMul_basis_basis, image_monomialReduce,
      image_basis _ index.isLt, image_basis _ other.isLt, pow_add]

theorem image_mul (left right : RingF) :
    image (ringFMul left right) = image left * image right := by
  apply linear_image_eq
    (fun value => image (ringFMul value right))
    (fun value => image value * image right)
  · rw [RingFLaws.ringFMul_comm, CarrierAction.ringFMul_zero_right, image_zero]
  · rw [image_zero, zero_mul]
  · intro first second
    rw [CarrierAction.ringFMul_add_left, image_add]
  · intro first second
    rw [image_add, add_mul]
  · intro scalar value
    rw [CarrierAction.ringFMul_scale_left, image_scale]
  · intro scalar value
    rw [image_scale]
    ring
  · intro index
    exact image_mul_basis index right

theorem toPolynomial_ringFMul_mod (left right : RingF) :
    toPolynomial (ringFMul left right) = (toPolynomial left * toPolynomial right) %ₘ modulus := by
  have quotient : AdjoinRoot.mk modulus (toPolynomial (ringFMul left right)) =
      AdjoinRoot.mk modulus (toPolynomial left * toPolynomial right) := by
    rw [map_mul]
    exact image_mul left right
  have remainder := congrArg (AdjoinRoot.modByMonicHom modulus_monic) quotient
  simpa only [AdjoinRoot.modByMonicHom_mk, toPolynomial_mod] using remainder

end NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFPolynomial
