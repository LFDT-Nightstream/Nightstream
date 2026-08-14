import Mathlib.Data.Nat.Digits.Lemmas
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

/-!
Model-level canonical radix-four decomposition for the Phi81 `PiDEC` algebra.

This file owns the scalar and complete-assignment split into seven common-sign
base-four digits. It proves exact recomposition and both norm directions. It
does not select a production profile, emit circuit rows, or claim Rust
conformance.
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev params := Concrete.Radix4Candidate.globalParams
abbrev ChildIndex := Fin params.k

def combinedBound : Nat := params.bigB

theorem parameter_values :
    params.b = 4 ∧ params.k = 7 ∧ combinedBound = 16384 := by
  decide

abbrev fieldOfNat := Radix.fieldOfNat

def magnitudeDigitList (value : F) : List Nat :=
  Nat.digitsAppend 4 params.k (centeredMagnitude value)

theorem magnitudeDigitList_length (value : F)
    (bounded : centeredMagnitude value < combinedBound) :
    (magnitudeDigitList value).length = params.k := by
  apply Nat.length_digitsAppend (by decide)
  simpa [combinedBound, params, Concrete.Radix4Candidate.globalParams,
    GlobalParams.bigB] using bounded

def magnitudeDigit (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) : Nat :=
  (magnitudeDigitList value).get ⟨index.val, by
    rw [magnitudeDigitList_length value bounded]
    exact index.isLt⟩

theorem magnitudeDigit_lt_four (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) : magnitudeDigit value bounded index < 4 := by
  apply Nat.lt_of_mem_digitsAppend (by decide) params.k
  change (magnitudeDigitList value).get _ ∈ magnitudeDigitList value
  exact List.get_mem (magnitudeDigitList value) _

private theorem magnitudeDigits_list (value : F)
    (bounded : centeredMagnitude value < combinedBound) :
    List.ofFn (magnitudeDigit value bounded) = magnitudeDigitList value := by
  apply List.ext_get
  · simpa using (magnitudeDigitList_length value bounded).symm
  · intro index leftBound rightBound
    simp [magnitudeDigit]

private theorem magnitudeDigitList_recompose (value : F) :
    Nat.ofDigits 4 (magnitudeDigitList value) = centeredMagnitude value := by
  rw [magnitudeDigitList, Nat.digitsAppend,
    Nat.ofDigits_append_replicate_zero, Nat.ofDigits_digits]

/-! ## Field Horner recomposition -/

def recomposeList : List F → F
  | [] => 0
  | digit :: digits => digit + fieldOfNat 4 * recomposeList digits

def recomposeScalar (digits : ChildIndex → F) : F :=
  recomposeList (List.ofFn digits)

/-- Verifier-owned radix-four weight for one of the seven children. -/
def radixWeight (index : ChildIndex) : F :=
  fieldOfNat (4 ^ index.val)

private def combineScalars : {count : Nat} →
    (Fin count → F) → (Fin count → F) → F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        combineScalars
          (fun index => weights index.succ)
          (fun index => values index.succ)

private theorem combineScalars_scale_weights (factor : F) {count : Nat}
    (weights values : Fin count → F) :
    combineScalars (fun index => factor * weights index) values =
      factor * combineScalars weights values := by
  induction count with
  | zero => exact Fin.mul_zero factor |>.symm
  | succ count inductionHypothesis =>
      simp only [combineScalars]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      calc
        (factor * weights 0) * values 0 +
              factor * combineScalars
                (fun index => weights index.succ)
                (fun index => values index.succ) =
            factor * (weights 0 * values 0) +
              factor * combineScalars
                (fun index => weights index.succ)
                (fun index => values index.succ) := by
          rw [Fin.mul_assoc]
        _ = factor *
              (weights 0 * values 0 +
                combineScalars
                  (fun index => weights index.succ)
                  (fun index => values index.succ)) :=
          (Lean.Grind.Fin.left_distrib _ _ _).symm

private theorem combineScalars_radix_eq_recomposeList {count : Nat}
    (values : Fin count → F) :
    combineScalars
        (fun index : Fin count => fieldOfNat (4 ^ index.val)) values =
      recomposeList (List.ofFn values) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [combineScalars, List.ofFn_succ, recomposeList,
        Fin.val_zero, Nat.pow_zero]
      rw [show fieldOfNat 1 = (1 : F) by rfl, Fin.one_mul]
      have tailWeights :
          (fun index : Fin count => fieldOfNat (4 ^ index.succ.val)) =
            (fun index : Fin count =>
              fieldOfNat 4 * fieldOfNat (4 ^ index.val)) := by
        funext index
        simp only [Fin.val_succ]
        rw [Nat.pow_succ']
        change Radix.fieldOfNat (4 * 4 ^ index.val) =
          Radix.fieldOfNat 4 * Radix.fieldOfNat (4 ^ index.val)
        exact Radix.fieldOfNat_mul 4 (4 ^ index.val)
      rw [tailWeights, combineScalars_scale_weights,
        inductionHypothesis (fun index => values index.succ)]

theorem weighted_recomposeScalar (values : ChildIndex → F) :
    combineScalars radixWeight values = recomposeScalar values := by
  exact combineScalars_radix_eq_recomposeList values

/-- Exact seven-child Horner form used by the radix-four circuit compiler. -/
theorem recomposeScalar_seven (values : ChildIndex → F) :
    recomposeScalar values =
      values ⟨0, by decide⟩ + fieldOfNat 4 *
        (values ⟨1, by decide⟩ + fieldOfNat 4 *
          (values ⟨2, by decide⟩ + fieldOfNat 4 *
            (values ⟨3, by decide⟩ + fieldOfNat 4 *
              (values ⟨4, by decide⟩ + fieldOfNat 4 *
                (values ⟨5, by decide⟩ +
                  fieldOfNat 4 * values ⟨6, by decide⟩))))) := by
  simp [recomposeScalar, recomposeList, params,
    Concrete.Radix4Candidate.globalParams, Fin.mul_zero, Fin.add_zero]

private theorem recomposeList_fieldOfNat (digits : List Nat) :
    recomposeList (digits.map fieldOfNat) =
      fieldOfNat (Nat.ofDigits 4 digits) := by
  induction digits with
  | nil => rfl
  | cons digit digits inductionHypothesis =>
      simp only [List.map_cons, recomposeList, Nat.ofDigits]
      rw [inductionHypothesis]
      change Radix.fieldOfNat digit +
          Radix.fieldOfNat 4 *
            Radix.fieldOfNat (Nat.ofDigits 4 digits : Nat) =
        Radix.fieldOfNat (digit + 4 * Nat.ofDigits 4 digits)
      rw [← Radix.fieldOfNat_mul, ← Radix.fieldOfNat_add]

private theorem recomposeList_neg (digits : List F) :
    recomposeList (digits.map fun digit => -digit) =
      -(recomposeList digits) := by
  induction digits with
  | nil => exact Lean.Grind.AddCommGroup.neg_zero.symm
  | cons digit digits inductionHypothesis =>
      simp only [List.map_cons, recomposeList]
      rw [inductionHypothesis]
      have mulNeg : fieldOfNat 4 * -recomposeList digits =
          -(fieldOfNat 4 * recomposeList digits) := by
        calc
          fieldOfNat 4 * -recomposeList digits =
              -recomposeList digits * fieldOfNat 4 := Fin.mul_comm _ _
          _ = -(recomposeList digits * fieldOfNat 4) :=
            Lean.Grind.Fin.neg_mul _ _
          _ = -(fieldOfNat 4 * recomposeList digits) := by
            rw [Fin.mul_comm (recomposeList digits) (fieldOfNat 4)]
      rw [mulNeg]
      exact (Lean.Grind.AddCommGroup.neg_add _ _).symm

def isNonnegative (value : F) : Prop :=
  value.val <= Centered.halfModulus

instance (value : F) : Decidable (isNonnegative value) :=
  inferInstanceAs (Decidable (value.val <= Centered.halfModulus))

def boundedDigit (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) : F :=
  if isNonnegative value then
    fieldOfNat (magnitudeDigit value bounded index)
  else
    -(fieldOfNat (magnitudeDigit value bounded index))

/-- One radix-four digit selected by a shared centered sign. -/
def signedDigit (negative : Bool) (digit : Nat) : F :=
  if negative then -(fieldOfNat digit) else fieldOfNat digit

theorem recomposeScalar_signed
    (negative : Bool) (digits : ChildIndex → Nat) :
    recomposeScalar (fun index => signedDigit negative (digits index)) =
      if negative then
        -(fieldOfNat (Nat.ofDigits 4 (List.ofFn digits)))
      else fieldOfNat (Nat.ofDigits 4 (List.ofFn digits)) := by
  unfold recomposeScalar
  cases negative with
  | false =>
      have listForm :
          List.ofFn (fun index => signedDigit false (digits index)) =
            (List.ofFn digits).map fieldOfNat := by
        rw [List.map_ofFn]
        rfl
      simp only [Bool.false_eq_true, ↓reduceIte]
      rw [listForm, recomposeList_fieldOfNat]
  | true =>
      have listForm :
          List.ofFn (fun index => signedDigit true (digits index)) =
            ((List.ofFn digits).map fieldOfNat).map
              (fun digit => -digit) := by
        rw [List.map_map, List.map_ofFn]
        rfl
      simp only [↓reduceIte]
      rw [listForm, recomposeList_neg, recomposeList_fieldOfNat]

def fallbackDigit (value : F) (index : ChildIndex) : F :=
  if index.val = 0 then value else 0

def splitScalar (value : F) (index : ChildIndex) : F :=
  if bounded : centeredMagnitude value < combinedBound then
    boundedDigit value bounded index
  else
    fallbackDigit value index

private theorem centeredMagnitude_eq_val_of_nonnegative (value : F)
    (nonnegative : isNonnegative value) :
    centeredMagnitude value = value.val := by
  unfold isNonnegative at nonnegative
  rw [Centered.centeredMagnitude_eq_distance]
  simp [Centered.distance, nonnegative]

private theorem centeredMagnitude_eq_complement_of_negative (value : F)
    (negative : ¬isNonnegative value) :
    centeredMagnitude value = goldilocksModulus - value.val := by
  unfold isNonnegative at negative
  rw [Centered.centeredMagnitude_eq_distance]
  simp [Centered.distance, negative]

private theorem fieldOfNat_val (value : F) :
    fieldOfNat value.val = value := by
  apply Fin.ext
  simp [fieldOfNat, Radix.fieldOfNat, Nat.mod_eq_of_lt value.isLt]

private theorem neg_fieldOfNat_complement (value : F) :
    -(fieldOfNat (goldilocksModulus - value.val)) = value := by
  by_cases valueZero : value = 0
  · subst value
    exact Lean.Grind.AddCommGroup.neg_zero
  · have valNonzero : value.val ≠ 0 := by
      intro equal
      apply valueZero
      apply Fin.ext
      simpa using equal
    have complementPositive : 0 < goldilocksModulus - value.val := by omega
    have complementLt : goldilocksModulus - value.val < goldilocksModulus := by
      omega
    have embeddedNonzero :
        fieldOfNat (goldilocksModulus - value.val) ≠ 0 := by
      intro equal
      have valuesEqual := congrArg Fin.val equal
      simp [fieldOfNat, Radix.fieldOfNat,
        Nat.mod_eq_of_lt complementLt] at valuesEqual
      omega
    apply Fin.ext
    rw [Fin.val_neg, if_neg embeddedNonzero]
    simp [fieldOfNat, Radix.fieldOfNat,
      Nat.mod_eq_of_lt complementLt]
    omega

private theorem boundedDigit_recompose (value : F)
    (bounded : centeredMagnitude value < combinedBound) :
    recomposeScalar (boundedDigit value bounded) = value := by
  have digitList := magnitudeDigits_list value bounded
  by_cases nonnegative : isNonnegative value
  · have listForm :
        List.ofFn (boundedDigit value bounded) =
          (magnitudeDigitList value).map fieldOfNat := by
      rw [← digitList]
      rw [List.map_ofFn]
      apply congrArg List.ofFn
      funext index
      simp [boundedDigit, nonnegative]
    unfold recomposeScalar
    rw [listForm, recomposeList_fieldOfNat,
      magnitudeDigitList_recompose,
      centeredMagnitude_eq_val_of_nonnegative value nonnegative,
      fieldOfNat_val]
  · have listForm :
        List.ofFn (boundedDigit value bounded) =
          (magnitudeDigitList value).map
            (fun digit => -(fieldOfNat digit)) := by
      rw [← digitList]
      rw [List.map_ofFn]
      apply congrArg List.ofFn
      funext index
      simp [boundedDigit, nonnegative]
    unfold recomposeScalar
    rw [listForm]
    have mapForm :
        (magnitudeDigitList value).map
            (fun digit => -(fieldOfNat digit)) =
          ((magnitudeDigitList value).map fieldOfNat).map
            (fun digit => -digit) := by
      simp
    rw [mapForm, recomposeList_neg, recomposeList_fieldOfNat,
      magnitudeDigitList_recompose,
      centeredMagnitude_eq_complement_of_negative value nonnegative,
      neg_fieldOfNat_complement]

private theorem fallbackDigit_recompose (value : F) :
    recomposeScalar (fallbackDigit value) = value := by
  simp [recomposeScalar, fallbackDigit, recomposeList, params,
    Concrete.Radix4Candidate.globalParams]
  simp only [Fin.mul_zero, Fin.add_zero]

theorem splitScalar_recompose (value : F) :
    recomposeScalar (splitScalar value) = value := by
  by_cases bounded : centeredMagnitude value < combinedBound
  · rw [show splitScalar value = boundedDigit value bounded by
      funext index
      simp [splitScalar, bounded]]
    exact boundedDigit_recompose value bounded
  · rw [show splitScalar value = fallbackDigit value by
      funext index
      simp [splitScalar, bounded]]
    exact fallbackDigit_recompose value

/-! ## Complete assignment split -/

def splitAssignment {shape : Shape}
    (assignment : Assignment shape) (index : ChildIndex) : Assignment shape :=
  fun column => splitScalar (assignment column) index

def recomposeAssignment {shape : Shape}
    (assignments : ChildIndex → Assignment shape) : Assignment shape :=
  fun column => recomposeScalar (fun child => assignments child column)

/-- The same assignment operation expressed with the generic base-linear
combiner used by the concrete PiDEC evaluation homomorphism. -/
def weightedRecomposeAssignment {shape : Shape}
    (assignments : ChildIndex → Assignment shape) : Assignment shape :=
  EvaluationHomomorphism.BaseLinear.combineAssignments radixWeight assignments

private theorem combineAssignments_apply {shape : Shape} {count : Nat}
    (weights : Fin count → F)
    (assignments : Fin count → Assignment shape)
    (column : Fin shape.carrierWidth) :
    EvaluationHomomorphism.BaseLinear.combineAssignments
        weights assignments column =
      combineScalars weights (fun index => assignments index column) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [EvaluationHomomorphism.BaseLinear.combineAssignments,
        EvaluationHomomorphism.BaseLinear.assignmentAdd,
        EvaluationHomomorphism.BaseLinear.assignmentScale,
        EvaluationHomomorphism.BaseLinear.Raw.assignmentAdd,
        EvaluationHomomorphism.BaseLinear.Raw.assignmentScale,
        combineScalars]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => assignments index.succ)]

theorem recomposeAssignment_eq_weighted {shape : Shape}
    (assignments : ChildIndex → Assignment shape) :
    recomposeAssignment assignments = weightedRecomposeAssignment assignments := by
  funext column
  unfold recomposeAssignment weightedRecomposeAssignment
  rw [combineAssignments_apply, weighted_recomposeScalar]

theorem split_recompose {shape : Shape} (assignment : Assignment shape) :
    recomposeAssignment (splitAssignment assignment) = assignment := by
  funext column
  exact splitScalar_recompose (assignment column)

/-! ## Split norm -/

private theorem centeredMagnitude_fieldOfNat_lt_four {digit : Nat}
    (digitLt : digit < 4) :
    centeredMagnitude (fieldOfNat digit) < 4 := by
  interval_cases digit <;> decide

theorem boundedDigit_norm (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) :
    centeredMagnitude (boundedDigit value bounded index) < params.b := by
  have digitBound := centeredMagnitude_fieldOfNat_lt_four
    (magnitudeDigit_lt_four value bounded index)
  by_cases nonnegative : isNonnegative value
  · simpa [boundedDigit, nonnegative, params,
      Concrete.Radix4Candidate.globalParams] using digitBound
  · simpa [boundedDigit, nonnegative, params,
      Concrete.Radix4Candidate.globalParams,
      Centered.centeredMagnitude_neg] using digitBound

theorem splitScalar_norm (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) :
    centeredMagnitude (splitScalar value index) < params.b := by
  simpa [splitScalar, bounded] using boundedDigit_norm value bounded index

theorem split_norm {shape : Shape} (assignment : Assignment shape)
    (bounded : assignmentNormBounded params.bigB assignment) :
    forall index,
      assignmentNormBounded params.b (splitAssignment assignment index) := by
  intro index column
  apply splitScalar_norm
  simpa [combinedBound] using bounded column

/-! ## Recomposition norm -/

private theorem two_mul (value : F) :
    fieldOfNat 2 * value = value + value := by
  have twoEq : fieldOfNat 2 = 1 + 1 := by decide
  rw [twoEq]
  calc
    ((1 : F) + 1) * value = value * ((1 : F) + 1) :=
      Fin.mul_comm _ _
    _ = value * 1 + value * 1 := Lean.Grind.Fin.left_distrib _ _ _
    _ = value + value := by simp only [Lean.Grind.Fin.mul_one]

private theorem four_mul (value : F) :
    fieldOfNat 4 * value = (value + value) + (value + value) := by
  have fourEq : fieldOfNat 4 = fieldOfNat 2 * fieldOfNat 2 := by
    exact (Radix.fieldOfNat_mul 2 2).symm
  rw [fourEq]
  calc
    (fieldOfNat 2 * fieldOfNat 2) * value =
        fieldOfNat 2 * (fieldOfNat 2 * value) := by
      rw [Fin.mul_assoc]
    _ = (fieldOfNat 2 * value) + (fieldOfNat 2 * value) :=
      two_mul _
    _ = (value + value) + (value + value) := by
      rw [two_mul]

private theorem centeredMagnitude_four_mul_le (value : F) :
    centeredMagnitude (fieldOfNat 4 * value) <=
      4 * centeredMagnitude value := by
  rw [four_mul]
  calc
    centeredMagnitude ((value + value) + (value + value)) <=
        centeredMagnitude (value + value) +
          centeredMagnitude (value + value) :=
      Centered.centeredMagnitude_add_le _ _
    _ <= (centeredMagnitude value + centeredMagnitude value) +
          (centeredMagnitude value + centeredMagnitude value) := by
      exact Nat.add_le_add
        (Centered.centeredMagnitude_add_le value value)
        (Centered.centeredMagnitude_add_le value value)
    _ = 4 * centeredMagnitude value := by omega

private theorem recomposeList_norm (digits : List F)
    (bounded : forall digit, digit ∈ digits → centeredMagnitude digit < 4) :
    centeredMagnitude (recomposeList digits) < 4 ^ digits.length := by
  induction digits with
  | nil => simp [recomposeList, Centered.centeredMagnitude_zero]
  | cons digit digits inductionHypothesis =>
      have digitBound : centeredMagnitude digit < 4 :=
        bounded digit (by simp)
      have tailBound : centeredMagnitude (recomposeList digits) <
          4 ^ digits.length := by
        apply inductionHypothesis
        intro tailDigit member
        exact bounded tailDigit (by simp [member])
      have digitAtMost : centeredMagnitude digit <= 3 := by omega
      have powerPositive : 0 < 4 ^ digits.length := by positivity
      have tailAtMost : centeredMagnitude (recomposeList digits) <=
          4 ^ digits.length - 1 := by omega
      simp only [recomposeList, List.length_cons]
      calc
        centeredMagnitude
            (digit + fieldOfNat 4 * recomposeList digits) <=
            centeredMagnitude digit +
              centeredMagnitude (fieldOfNat 4 * recomposeList digits) :=
          Centered.centeredMagnitude_add_le _ _
        _ <= centeredMagnitude digit +
              4 * centeredMagnitude (recomposeList digits) := by
          exact Nat.add_le_add_left
            (centeredMagnitude_four_mul_le (recomposeList digits)) _
        _ <= 3 + 4 * (4 ^ digits.length - 1) := by
          exact Nat.add_le_add digitAtMost
            (Nat.mul_le_mul_left 4 tailAtMost)
        _ < 4 ^ (digits.length + 1) := by
          rw [Nat.pow_succ]
          omega

theorem recomposeScalar_norm
    (digits : ChildIndex → F)
    (bounded : forall index, centeredMagnitude (digits index) < params.b) :
    centeredMagnitude (recomposeScalar digits) < params.bigB := by
  unfold recomposeScalar
  apply (show centeredMagnitude (recomposeList (List.ofFn digits)) <
      4 ^ (List.ofFn digits).length by
    apply recomposeList_norm
    intro digit member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    simpa [params, Concrete.Radix4Candidate.globalParams] using bounded index)

theorem recompose_norm {shape : Shape}
    (assignments : ChildIndex → Assignment shape)
    (bounded : forall index,
      assignmentNormBounded params.b (assignments index)) :
    assignmentNormBounded params.bigB
      (recomposeAssignment assignments) := by
  intro column
  apply recomposeScalar_norm
  intro index
  exact bounded index column

/-! ## Canonical split uniqueness -/

private theorem centeredMagnitude_fieldOfNat_eq_of_combinedBound
    {value : Nat} (bounded : value < combinedBound) :
    centeredMagnitude (fieldOfNat value) = value := by
  have belowModulus : value < goldilocksModulus := by
    norm_num [combinedBound, params,
      Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
      goldilocksModulus] at bounded ⊢
    omega
  have belowHalf : value <= Centered.halfModulus := by
    norm_num [combinedBound, params,
      Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
      Centered.halfModulus, goldilocksModulus] at bounded ⊢
    omega
  rw [Centered.centeredMagnitude_eq_distance]
  simp [Centered.distance, fieldOfNat, Radix.fieldOfNat,
    Nat.mod_eq_of_lt belowModulus, belowHalf]

private theorem fieldOfNat_nonnegative_of_combinedBound
    {value : Nat} (bounded : value < combinedBound) :
    isNonnegative (fieldOfNat value) := by
  have belowModulus : value < goldilocksModulus := by
    norm_num [combinedBound, params,
      Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
      goldilocksModulus] at bounded ⊢
    omega
  unfold isNonnegative
  simp [fieldOfNat, Radix.fieldOfNat, Nat.mod_eq_of_lt belowModulus]
  norm_num [combinedBound, params,
    Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
    Centered.halfModulus, goldilocksModulus] at bounded ⊢
  omega

private theorem neg_fieldOfNat_negative_of_positive_combinedBound
    {value : Nat} (positive : 0 < value)
    (bounded : value < combinedBound) :
    ¬isNonnegative (-(fieldOfNat value)) := by
  have belowModulus : value < goldilocksModulus := by
    norm_num [combinedBound, params,
      Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
      goldilocksModulus] at bounded ⊢
    omega
  have nonzero : fieldOfNat value ≠ 0 := by
    intro equal
    have valuesEqual := congrArg Fin.val equal
    simp [fieldOfNat, Radix.fieldOfNat,
      Nat.mod_eq_of_lt belowModulus] at valuesEqual
    omega
  unfold isNonnegative
  rw [Fin.val_neg]
  simp only [nonzero, ↓reduceIte]
  simp [fieldOfNat, Radix.fieldOfNat,
    Nat.mod_eq_of_lt belowModulus]
  norm_num [combinedBound, params,
    Concrete.Radix4Candidate.globalParams, GlobalParams.bigB,
    Centered.halfModulus, goldilocksModulus] at bounded ⊢
  omega

/-- Common-sign radix-four digits and exact recomposition determine the
verifier-computed public split. This rejects alternate child vectors that
recompose to the same parent. -/
theorem splitScalar_eq_signed_of_recompose
    (value : F) (negative : Bool) (digits : ChildIndex → Nat)
    (digitBound : forall index, digits index < 4)
    (recomposes :
      recomposeScalar (fun index => signedDigit negative (digits index)) =
        value) :
    forall index,
      splitScalar value index = signedDigit negative (digits index) := by
  let magnitude := Nat.ofDigits 4 (List.ofFn digits)
  have magnitudeBound : magnitude < combinedBound := by
    have rawBound := Nat.ofDigits_lt_base_pow_length
      (b := 4) (l := List.ofFn digits) (by decide : 1 < 4) (by
        intro digit member
        rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
        exact digitBound position)
    simpa [magnitude, combinedBound, params,
      Concrete.Radix4Candidate.globalParams, GlobalParams.bigB] using rawBound
  have valueBound : centeredMagnitude value < combinedBound := by
    rw [← recomposes]
    apply recomposeScalar_norm
    intro index
    have small := centeredMagnitude_fieldOfNat_lt_four (digitBound index)
    cases negative <;>
      simpa [signedDigit, params, Concrete.Radix4Candidate.globalParams,
        Centered.centeredMagnitude_neg] using small
  have signedRecomposition := recomposeScalar_signed negative digits
  have magnitudeExact : centeredMagnitude value = magnitude := by
    cases negative with
    | false =>
        have valueExact : fieldOfNat magnitude = value := by
          simpa [magnitude] using signedRecomposition.symm.trans recomposes
        rw [← valueExact]
        exact centeredMagnitude_fieldOfNat_eq_of_combinedBound magnitudeBound
    | true =>
        have valueExact : -(fieldOfNat magnitude) = value := by
          simpa [magnitude] using signedRecomposition.symm.trans recomposes
        rw [← valueExact, Centered.centeredMagnitude_neg]
        exact centeredMagnitude_fieldOfNat_eq_of_combinedBound magnitudeBound
  have canonicalListsEqual :
      magnitudeDigitList value = List.ofFn digits := by
    apply Nat.injOn_ofDigits (b := 4) (by decide) params.k
    · exact ⟨magnitudeDigitList_length value valueBound, by
        intro digit member
        exact Nat.lt_of_mem_digitsAppend (by decide) params.k digit member⟩
    · exact ⟨by simp [params, Concrete.Radix4Candidate.globalParams], by
        intro digit member
        rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
        exact digitBound position⟩
    · rw [magnitudeDigitList_recompose, magnitudeExact]
  have digitFunction : magnitudeDigit value valueBound = digits := by
    apply List.ofFn_injective
    exact (magnitudeDigits_list value valueBound).trans canonicalListsEqual
  intro index
  have digitExact : magnitudeDigit value valueBound index = digits index :=
    congrFun digitFunction index
  rw [splitScalar, dif_pos valueBound]
  cases negative with
  | false =>
      have valueExact : fieldOfNat magnitude = value := by
        simpa [magnitude] using signedRecomposition.symm.trans recomposes
      have nonnegative : isNonnegative value := by
        rw [← valueExact]
        exact fieldOfNat_nonnegative_of_combinedBound magnitudeBound
      simp [boundedDigit, signedDigit, nonnegative, digitExact]
  | true =>
      by_cases magnitudeZero : magnitude = 0
      · have valueExact : value = 0 := by
          have equation : -(fieldOfNat magnitude) = value := by
            simpa [magnitude] using signedRecomposition.symm.trans recomposes
          simpa [magnitudeZero] using equation.symm
        have suppliedListZero :
            List.ofFn digits = List.replicate params.k 0 := by
          apply Nat.injOn_ofDigits (b := 4) (by decide) params.k
          · exact ⟨by simp [params, Concrete.Radix4Candidate.globalParams], by
              intro digit member
              rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
              exact digitBound position⟩
          · exact ⟨by simp, by simp⟩
          · simp [magnitude, magnitudeZero]
        have zeroFunctions : digits = (fun _ : ChildIndex => 0) := by
          apply List.ofFn_injective
          calc
            List.ofFn digits = List.replicate params.k 0 := suppliedListZero
            _ = List.ofFn (fun _ : ChildIndex => 0) := by
              simp [params, Concrete.Radix4Candidate.globalParams]
        have digitZero : digits index = 0 := congrFun zeroFunctions index
        have canonicalDigitZero :
            magnitudeDigit value valueBound index = 0 :=
          digitExact.trans digitZero
        have nonnegative : isNonnegative value := by
          simp [valueExact, isNonnegative]
        simp only [boundedDigit, nonnegative, ↓reduceIte, signedDigit]
        rw [canonicalDigitZero, digitZero]
        rfl
      · have magnitudePositive : 0 < magnitude :=
          Nat.pos_of_ne_zero magnitudeZero
        have valueExact : -(fieldOfNat magnitude) = value := by
          simpa [magnitude] using signedRecomposition.symm.trans recomposes
        have negativeValue : ¬isNonnegative value := by
          rw [← valueExact]
          exact neg_fieldOfNat_negative_of_positive_combinedBound
            magnitudePositive magnitudeBound
        simp [boundedDigit, signedDigit, negativeValue, digitExact]

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4
