import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring
import Mathlib.RingTheory.AdjoinRoot
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-! Polynomial views of the existing 54-coefficient RingF carrier. These
noncomputable views serve only the multiplication and inverse correctness
proofs. They define no executable inverse or protocol operation. -/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

open NightstreamFPrime.Spec
open Polynomial

local instance : Fact (1 < goldilocksModulus) := ⟨by decide⟩

private abbrev Base := ZMod goldilocksModulus

local instance : Nontrivial Base := ZMod.nontrivial goldilocksModulus

noncomputable def modulus : Polynomial Base :=
  X ^ ringDegree + (X ^ ringMiddleDegree + 1)

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
    exact if_neg (fun equal => different (Fin.ext equal))
  · simp

private theorem coeff_toPolynomial_of_ge (value : RingF) (index : Nat)
    (outside : ringDegree ≤ index) : (toPolynomial value).coeff index = 0 := by
  classical
  simp only [toPolynomial, finsetSum_coeff, C_mul_X_pow_eq_monomial,
    coeff_monomial]
  apply Finset.sum_eq_zero
  intro other _
  exact if_neg (by have := other.isLt; omega)

theorem toPolynomial_injective : Function.Injective toPolynomial := by
  intro left right equal
  funext index
  have atIndex := congrArg (fun p => p.coeff index.val) equal
  simpa only [coeff_toPolynomial] using atIndex

theorem toPolynomial_zero : toPolynomial ringFZero = 0 := by
  ext index
  rw [coeff_zero]
  by_cases inside : index < ringDegree
  · exact coeff_toPolynomial ringFZero ⟨index, inside⟩
  · exact coeff_toPolynomial_of_ge ringFZero index (Nat.le_of_not_gt inside)

private theorem toPolynomial_basis (degree : Nat) (inside : degree < ringDegree) :
    toPolynomial (RingFLaws.basis degree) = X ^ degree := by
  ext index
  by_cases indexInside : index < ringDegree
  · rw [coeff_toPolynomial _ ⟨index, indexInside⟩]
    by_cases same : index = degree
    · subst index
      simp only [RingFLaws.basis, ringFMonomial, coeff_X_pow, ↓reduceIte]
      rfl
    · simp only [RingFLaws.basis, ringFMonomial, coeff_X_pow, same,
        Ne.symm same, ↓reduceIte]
      rfl
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

private theorem degree_lower :
    (X ^ ringMiddleDegree + 1 : Polynomial Base).degree = (27 : WithBot Nat) := by
  simpa [ringMiddleDegree] using
    (degree_X_pow_add_C (R := Base) (n := 27) (by decide) (1 : Base))

theorem modulus_monic : modulus.Monic := by
  apply monic_X_pow_add
  rw [degree_lower]
  decide

private theorem degree_modulus : modulus.degree = (ringDegree : WithBot Nat) := by
  rw [modulus, degree_add_eq_left_of_degree_lt]
  · exact degree_X_pow ringDegree
  · rw [degree_lower, degree_X_pow]
    decide

private theorem degree_toPolynomial (value : RingF) :
    (toPolynomial value).degree < (ringDegree : WithBot Nat) :=
  degree_sum_fin_lt (R := Base) value

private theorem toPolynomial_mod (value : RingF) :
    toPolynomial value %ₘ modulus = toPolynomial value := by
  apply (modByMonic_eq_self_iff modulus_monic).mpr
  rw [degree_modulus]
  exact degree_toPolynomial value

private abbrev QuotientRing := AdjoinRoot modulus

private noncomputable def image (value : RingF) : QuotientRing :=
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
    AdjoinRoot.root modulus ^ 54 + (AdjoinRoot.root modulus ^ 27 + 1) = 0 := by
  have relation := AdjoinRoot.mk_self (f := modulus)
  change AdjoinRoot.mk modulus ((X : Polynomial Base) ^ 54 + (X ^ 27 + 1)) = 0 at relation
  simpa only [map_add, map_pow, map_one, AdjoinRoot.mk_X] using relation

private theorem root_period : AdjoinRoot.root modulus ^ 81 = 1 := by
  apply sub_eq_zero.mp
  calc
    _ = (AdjoinRoot.root modulus ^ 27 - 1) *
        (AdjoinRoot.root modulus ^ 54 + (AdjoinRoot.root modulus ^ 27 + 1)) := by ring
    _ = 0 := by rw [root_relation, mul_zero]

private theorem root_pow_mod (degree : Nat) :
    AdjoinRoot.root modulus ^ degree = AdjoinRoot.root modulus ^ (degree % 81) := by
  conv_lhs => rw [← Nat.mod_add_div degree 81]
  rw [pow_add, pow_mul, root_period, one_pow, mul_one]

private theorem image_monomialReduce (degree : Nat) :
    image (RingFLaws.monomialReduce degree) = AdjoinRoot.root modulus ^ degree := by
  rw [root_pow_mod]
  have residual : degree % 81 < 81 := Nat.mod_lt _ (by decide)
  unfold RingFLaws.monomialReduce
  dsimp only
  split_ifs with low
  · exact image_basis _ low
  · have high : 54 ≤ degree % 81 := by simpa [ringDegree] using Nat.le_of_not_gt low
    rw [image_add, image_scale, image_scale,
      image_basis _ (by change degree % 81 - 54 < 54; omega),
      image_basis _ (by change degree % 81 - 27 < 54; omega)]
    change AdjoinRoot.of modulus (-1 : Base) * AdjoinRoot.root modulus ^ (degree % 81 - 54) +
      AdjoinRoot.of modulus (-1 : Base) * AdjoinRoot.root modulus ^ (degree % 81 - 27) = _
    rw [map_neg, map_one]
    have highPower : AdjoinRoot.root modulus ^ (degree % 81) =
        AdjoinRoot.root modulus ^ (degree % 81 - 54) * AdjoinRoot.root modulus ^ 54 := by
      rw [← pow_add]
      congr 1
      omega
    have middlePower : AdjoinRoot.root modulus ^ (degree % 81 - 27) =
        AdjoinRoot.root modulus ^ (degree % 81 - 54) * AdjoinRoot.root modulus ^ 27 := by
      rw [← pow_add]
      congr 1
      omega
    have relation : AdjoinRoot.root modulus ^ 54 = -(AdjoinRoot.root modulus ^ 27 + 1) :=
      eq_neg_of_add_eq_zero_left root_relation
    rw [highPower, middlePower, relation]
    ring

/-- Lift equality through the finite coefficient expansion. Both maps keep
only the existing base-linear RingF operations as hypotheses. -/
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
      rw [if_neg (fun equal => different (Fin.ext equal.symm)), mul_zero]
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

private theorem image_mul (left right : RingF) :
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

/-- The protocol's executed schoolbook multiplication has exactly the
polynomial product's remainder modulo Phi81. -/
theorem toPolynomial_ringFMul_mod (left right : RingF) :
    toPolynomial (ringFMul left right) = (toPolynomial left * toPolynomial right) %ₘ modulus := by
  have quotient : AdjoinRoot.mk modulus (toPolynomial (ringFMul left right)) =
      AdjoinRoot.mk modulus (toPolynomial left * toPolynomial right) := by
    rw [map_mul]
    exact image_mul left right
  have remainder := congrArg (AdjoinRoot.modByMonicHom modulus_monic) quotient
  simpa only [AdjoinRoot.modByMonicHom_mk, toPolynomial_mod] using remainder

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
