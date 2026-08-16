import Mathlib.Data.List.GetD
import Nightstream.Implementation.R1CS.Core.SeededPhi81
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-!
Contract: exact algebraic meaning of the compact seeded-Phi81 rotation
compiler.

Assurance tier: implementation-to-algebra bridge.

Owns the canonical Goldilocks interpretation of sampled coefficient lists,
the proof that one compact rotation is multiplication by `X` modulo Phi81,
and the proof that `n < 54` rotations are multiplication by `X^n`.

Does not own coefficient sampling, Rust `rand_chacha` conformance, input
selector columns, output rows, commitment binding, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SeededPhi81RingRefinement

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

def residueNat (value : Nat) : F :=
  ⟨value % goldilocksP, by
    simpa [goldilocksP, goldilocksModulus] using
      (Nat.mod_lt value (by decide : 0 < goldilocksP))⟩

@[simp]
theorem residueNat_val (value : Nat) :
    (residueNat value).val = value % goldilocksP := rfl

@[simp]
theorem residueNat_fin_val (value : F) :
    residueNat value.val = value := by
  apply Fin.ext
  change value.val % goldilocksP = value.val
  exact Nat.mod_eq_of_lt value.isLt

@[simp]
theorem residueNat_mod (value : Nat) :
    residueNat (value % goldilocksP) = residueNat value := by
  apply Fin.ext
  change value % goldilocksP % goldilocksP = value % goldilocksP
  exact Nat.mod_mod _ _

theorem residueNat_add (left right : Nat) :
    residueNat (left + right) = residueNat left + residueNat right := by
  apply Fin.ext
  change (left + right) % goldilocksP =
    (left % goldilocksP + right % goldilocksP) % goldilocksP
  exact Nat.add_mod left right goldilocksP

theorem residueNat_mul (left right : Nat) :
    residueNat (left * right) = residueNat left * residueNat right := by
  apply Fin.ext
  change (left * right) % goldilocksP =
    (left % goldilocksP * (right % goldilocksP)) % goldilocksP
  exact Nat.mul_mod left right goldilocksP

theorem residueNat_fieldNeg (value : Nat) :
    residueNat (SeededPhi81.fieldNeg value) = -residueNat value := by
  apply Fin.ext
  rw [Fin.val_neg]
  change SeededPhi81.fieldNeg value % goldilocksP =
    if residueNat value = 0 then 0
    else goldilocksP - (residueNat value).val
  by_cases zero : value % goldilocksP = 0
  · have fieldZero : residueNat value = 0 := by
      apply Fin.ext
      exact zero
    simp [SeededPhi81.fieldNeg, zero, fieldZero]
  · have nonzero : residueNat value ≠ 0 := by
      intro equal
      have valuesEqual := congrArg Fin.val equal
      simp [residueNat, zero] at valuesEqual
    rw [if_neg nonzero]
    simp only [SeededPhi81.fieldNeg, zero, ↓reduceIte, residueNat_val]
    have residueLt : value % goldilocksP < goldilocksP :=
      Nat.mod_lt _ (by decide)
    have differenceLt :
        goldilocksP - value % goldilocksP < goldilocksP := by
      omega
    exact Nat.mod_eq_of_lt differenceLt

theorem residueNat_fieldSub (left right : Nat) :
    residueNat (SeededPhi81.fieldSub left right) =
      residueNat left - residueNat right := by
  rw [Fin.sub_eq_add_neg]
  unfold SeededPhi81.fieldSub
  rw [residueNat_mod, residueNat_add, residueNat_mod,
    residueNat_fieldNeg]

/-- Coefficient-list interpretation used by the sampled verifier key. -/
def ringOfList (values : List Nat) : RingF :=
  fun lane => residueNat (values.getD lane.val 0)

/-- Explicit coefficient action of multiplication by `X`. -/
def mulX (value : RingF) : RingF :=
  fun output =>
    if output.val = 0 then
      -value ⟨53, by decide⟩
    else if output.val = 27 then
      value ⟨26, by decide⟩ - value ⟨53, by decide⟩
    else
      value ⟨output.val - 1, by
        have outputLt := output.isLt
        simp only [ringDegree] at outputLt ⊢
        omega⟩

private theorem add_sub_pair
    (a₁ a₂ b₁ b₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) =
      (a₁ - b₁) + (a₂ - b₂) := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_comm⟩
  ac_rfl

private theorem scale_sub (scalar a b : F) :
    scalar * a - scalar * b = scalar * (a - b) := by
  have mulNeg : scalar * -b = -(scalar * b) := by
    calc
      scalar * -b = (-b) * scalar := Fin.mul_comm _ _
      _ = -(b * scalar) := Lean.Grind.Fin.neg_mul _ _
      _ = -(scalar * b) := by rw [Fin.mul_comm b scalar]
  calc
    scalar * a - scalar * b = scalar * a + -(scalar * b) :=
      Fin.sub_eq_add_neg _ _
    _ = scalar * a + scalar * -b := by rw [mulNeg]
    _ = scalar * (a + -b) :=
      (ConcreteCarrier.baseLaws.left_distrib scalar a (-b)).symm
    _ = scalar * (a - b) := by rw [Fin.sub_eq_add_neg]

private theorem mul_neg_right (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := by rw [Fin.mul_comm right left]

theorem mulX_zero : mulX ringFZero = ringFZero := by
  funext output
  by_cases zero : output.val = 0
  · simp [mulX, zero, ringFZero]
    exact Lean.Grind.AddCommGroup.neg_zero
  · by_cases middle : output.val = 27
    · simp [mulX, zero, middle, ringFZero]
    · simp [mulX, zero, middle, ringFZero]

theorem mulX_add (left right : RingF) :
    mulX (ringFAdd left right) =
      ringFAdd (mulX left) (mulX right) := by
  funext output
  by_cases zero : output.val = 0
  · simp only [mulX, zero, if_pos, ringFAdd,
      Lean.Grind.AddCommGroup.neg_add]
  · by_cases middle : output.val = 27
    · simp only [mulX, zero, if_neg, middle, if_pos, ringFAdd]
      exact add_sub_pair _ _ _ _
    · simp [mulX, zero, middle, ringFAdd]

theorem mulX_scale (scalar : F) (value : RingF) :
    mulX (CarrierAction.ringFScale scalar value) =
      CarrierAction.ringFScale scalar (mulX value) := by
  funext output
  by_cases zero : output.val = 0
  · simp [mulX, zero, CarrierAction.ringFScale]
    exact (mul_neg_right _ _).symm
  · by_cases middle : output.val = 27
    · simp only [mulX, zero, if_neg, middle, if_pos,
        CarrierAction.ringFScale]
      exact scale_sub _ _ _
    · simp [mulX, zero, middle, CarrierAction.ringFScale]

private theorem mulX_basis (index : Fin ringDegree) :
    mulX (RingFLaws.basis index.val) =
      RingFLaws.monomialReduce (1 + index.val) := by
  by_cases last : index.val = 53
  · have reduced :
        RingFLaws.monomialReduce (1 + index.val) =
          ringFAdd
            (CarrierAction.ringFScale (-1) (RingFLaws.basis 0))
            (CarrierAction.ringFScale (-1) (RingFLaws.basis 27)) := by
      simp [RingFLaws.monomialReduce, ringDegree, ringMiddleDegree, last]
    rw [reduced]
    funext output
    unfold mulX RingFLaws.basis ringFMonomial ringFAdd
      CarrierAction.ringFScale
    simp only
    by_cases zero : output.val = 0
    · have outputNe27 : output.val ≠ 27 := by omega
      rw [if_pos zero, if_pos (by omega : 53 = index.val),
        if_pos zero, if_neg outputNe27]
      simp only [Fin.mul_one, Fin.mul_zero, Fin.add_zero]
    · by_cases middle : output.val = 27
      · rw [if_neg zero, if_pos middle,
          if_neg (by omega : 26 ≠ index.val),
          if_pos (by omega : 53 = index.val),
          if_neg (by omega : output.val ≠ 0), if_pos middle]
        simp only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.mul_zero,
          Fin.mul_one]
      · have priorNe : output.val - 1 ≠ 53 := by
          have outputLt := output.isLt
          simp only [ringDegree] at outputLt
          omega
        have priorIndexNe : output.val - 1 ≠ index.val := by omega
        rw [if_neg zero, if_neg middle, if_neg priorIndexNe,
          if_neg (by omega : output.val ≠ 0), if_neg middle]
        simp only [Fin.mul_zero, Fin.add_zero,
          Lean.Grind.AddCommGroup.neg_zero]
  · have indexLt53 : index.val < 53 := by
      have indexLt := index.isLt
      simp only [ringDegree] at indexLt
      omega
    have sumLt : 1 + index.val < ringDegree := by
      simp only [ringDegree]
      omega
    have sumLt54 : 1 + index.val < 54 := by omega
    have residue : (1 + index.val) % 81 = 1 + index.val :=
      Nat.mod_eq_of_lt (by omega)
    have reduced :
        RingFLaws.monomialReduce (1 + index.val) =
          RingFLaws.basis (1 + index.val) := by
      unfold RingFLaws.monomialReduce
      rw [residue, if_pos sumLt]
    rw [reduced]
    funext output
    unfold mulX RingFLaws.basis ringFMonomial
    by_cases zero : output.val = 0
    · rw [if_pos zero, if_neg (by omega : 53 ≠ index.val),
        if_neg (by omega : output.val ≠ 1 + index.val)]
      exact Lean.Grind.AddCommGroup.neg_zero
    · by_cases middle : output.val = 27
      · by_cases index26 : index.val = 26
        · rw [if_neg zero, if_pos middle, if_pos (by omega : 26 = index.val),
            if_neg (by omega : 53 ≠ index.val),
            if_pos (by omega : output.val = 1 + index.val)]
          simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
            Fin.add_zero]
        · rw [if_neg zero, if_pos middle,
            if_neg (by omega : 26 ≠ index.val),
            if_neg (by omega : 53 ≠ index.val),
            if_neg (by omega : output.val ≠ 1 + index.val)]
          exact Fin.sub_self
      · by_cases prior : output.val - 1 = index.val
        · have outputEq : output.val = 1 + index.val := by omega
          rw [if_neg zero, if_neg middle, if_pos prior, if_pos outputEq]
        · have outputNe : output.val ≠ 1 + index.val := by omega
          rw [if_neg zero, if_neg middle, if_neg prior, if_neg outputNe]

private theorem basisOne_mul (value : RingF) :
    ringFMul (RingFLaws.basis 1) value = mulX value := by
  apply RingFLaws.ringF_linear_eq_of_basis
  · exact CarrierAction.ringFMul_zero_right _
  · exact mulX_zero
  · exact CarrierAction.ringFMul_add_right _
  · exact mulX_add
  · exact CarrierAction.ringFMul_scale_right _
  · exact mulX_scale
  · intro index
    let one : Fin ringDegree := ⟨1, by decide⟩
    change ringFMul (RingFLaws.basis one.val)
      (RingFLaws.basis index.val) = mulX (RingFLaws.basis index.val)
    rw [RingFLaws.ringFMul_basis_basis one index]
    simpa [one] using (mulX_basis index).symm

/-- One list rotation is one quotient-ring multiplication by `X`. -/
theorem ringOfList_rotatePhi81 (values : List Nat) :
    ringOfList (SeededPhi81.rotatePhi81 values) =
      ringFMul (ringOfList values) (ringFMonomial 1 1) := by
  rw [RingFLaws.ringFMul_comm]
  change ringOfList (SeededPhi81.rotatePhi81 values) =
    ringFMul (RingFLaws.basis 1) (ringOfList values)
  rw [basisOne_mul]
  funext output
  have outputLt : output.val < SeededPhi81.dimension := by
    have := output.isLt
    simpa [SeededPhi81.dimension, SeededPhi81Sampler.dimension,
      ringDegree] using this
  have mappedBound :
      output.val <
        ((List.range SeededPhi81.dimension).map fun coordinate =>
          if coordinate = 0 then
            SeededPhi81.fieldNeg (values.getD (SeededPhi81.dimension - 1) 0)
          else if coordinate = 27 then
            SeededPhi81.fieldSub (values.getD 26 0)
              (values.getD (SeededPhi81.dimension - 1) 0)
          else values.getD (coordinate - 1) 0).length := by
    simp [outputLt]
  unfold ringOfList SeededPhi81.rotatePhi81
  rw [List.getD_eq_getElem _ _ mappedBound]
  simp only [List.getElem_map, List.getElem_range]
  by_cases zero : output.val = 0
  · simp [mulX, zero, residueNat_fieldNeg, SeededPhi81.dimension,
      SeededPhi81Sampler.dimension]
  · by_cases middle : output.val = 27
    · simp [mulX, zero, middle, residueNat_fieldSub,
        SeededPhi81.dimension, SeededPhi81Sampler.dimension]
    · simp [mulX, zero, middle]

private theorem monomial_one_mul
    (count : Nat) (successorLt : count + 1 < ringDegree) :
    ringFMul (ringFMonomial 1 1) (ringFMonomial count 1) =
      ringFMonomial (count + 1) 1 := by
  change ringFMul (RingFLaws.basis 1) (RingFLaws.basis count) =
    RingFLaws.basis (count + 1)
  let one : Fin ringDegree := ⟨1, by decide⟩
  let current : Fin ringDegree := ⟨count, by
    simp only [ringDegree] at successorLt ⊢
    omega⟩
  have product := RingFLaws.ringFMul_basis_basis one current
  have successorLt54 : count + 1 < 54 := by
    simpa only [ringDegree] using successorLt
  have residue : (1 + count) % 81 = 1 + count :=
    Nat.mod_eq_of_lt (by omega)
  simp only [one, current] at product
  rw [product]
  unfold RingFLaws.monomialReduce
  simp only
  rw [residue, if_pos (by omega : 1 + count < ringDegree)]
  simp only [Nat.add_comm 1 count]

/-- Fewer than 54 compact rotations are multiplication by the matching
coefficient basis monomial. -/
theorem ringOfList_rotatePow
    (count : Nat) (countLt : count < ringDegree) (values : List Nat) :
    ringOfList (SeededPhi81.rotatePow count values) =
      ringFMul (ringOfList values) (ringFMonomial count 1) := by
  induction count generalizing values with
  | zero =>
      simp only [SeededPhi81.rotatePow]
      simpa [ringFOne] using
        (RingFLaws.ringFMul_one_right (ringOfList values)).symm
  | succ count inductionHypothesis =>
      have countLt' : count < ringDegree := by omega
      have successorLt : count + 1 < ringDegree := by omega
      simp only [SeededPhi81.rotatePow]
      rw [inductionHypothesis countLt']
      rw [ringOfList_rotatePhi81]
      rw [RingFLaws.ringFMul_assoc]
      rw [monomial_one_mul count successorLt]

end Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
