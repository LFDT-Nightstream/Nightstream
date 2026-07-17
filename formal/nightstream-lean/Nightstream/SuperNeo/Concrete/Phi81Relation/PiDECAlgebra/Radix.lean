import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-!
Canonical production radix decomposition for the typed Phi81 `PiDEC` algebra.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 14`.
Phase: complete-assignment split and recomposition.
Constraint family: semantic digit and norm obligations only; this file emits no
rows.

Owns: a deterministic signed-binary split inside the strict combined bound; a
total exact fallback outside that precondition; coordinate and assignment
recomposition; fresh-child norm preservation; and the reverse norm bound for
arbitrary fourteen-child recompositions.

Does not own: commitment, public-input, or evaluation homomorphisms; child CE
membership; transcript or NIFS acceptance; Rust/R1CS refinement; row removal;
or constraint counts.

Emits constraints: no.

Authority boundary: `PiDEC.Algebra.split_recompose` is unconditional, while
`split_norm` assumes a strictly `B`-bounded parent. The bounded branch is the
canonical fourteen-position signed binary expansion. Outside that assumption,
child zero retains the original coefficient and the other children are zero.
That fallback is exact but deliberately carries no shortness claim.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.radix.parameters` | radix two, fourteen children, `B = 16384` | verifier fixed | `production_parameters` |
| `nifs.pi_dec.verify.radix.scalar.digits` | digit `i` is `(magnitude / 2^i) mod 2` | computed | `magnitudeDigit`, `magnitudeDigit_lt_two` |
| `nifs.pi_dec.verify.radix.scalar.sign` | both centered signs use the same magnitude digits | computed | `boundedDigit` |
| `nifs.pi_dec.verify.radix.scalar.total` | unbounded values retain an exact first-child fallback | computed | `splitScalar` |
| `nifs.pi_dec.verify.radix.recompose` | every field value and complete assignment recomposes exactly | derived | `splitScalar_recompose`, `split_recompose` |
| `nifs.pi_dec.verify.radix.split_norm` | a strict-`B` parent produces strict-`2` children | derived | `split_norm` |
| `nifs.pi_dec.verify.radix.recompose_norm` | fourteen strict-`2` children recompose below `B` | derived | `recompose_norm` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ChildIndex := Fin productionGlobalParams.k

/-- The production combined-witness bound `B = 2^14`. -/
def combinedBound : Nat := productionGlobalParams.bigB

theorem production_parameters :
    productionGlobalParams.b = 2 /\
      productionGlobalParams.k = 14 /\
      combinedBound = 16384 := by
  decide

/-! ## Local natural and field arithmetic -/

/-- Canonical embedding of a natural into the active Goldilocks residue type. -/
def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

@[simp] theorem fieldOfNat_zero : fieldOfNat 0 = (0 : F) := by
  rfl

@[simp] theorem fieldOfNat_one : fieldOfNat 1 = (1 : F) := by
  rfl

theorem fieldOfNat_add (left right : Nat) :
    fieldOfNat (left + right) = fieldOfNat left + fieldOfNat right := by
  apply Fin.ext
  simp [fieldOfNat, Fin.val_add, Nat.add_mod]

theorem fieldOfNat_mul (left right : Nat) :
    fieldOfNat (left * right) = fieldOfNat left * fieldOfNat right := by
  apply Fin.ext
  simp [fieldOfNat, Fin.val_mul, Nat.mul_mod]

private theorem fieldOfNat_succ (value : Nat) :
    fieldOfNat (value + 1) = fieldOfNat value + 1 := by
  rw [show (1 : F) = fieldOfNat 1 by rfl]
  exact fieldOfNat_add value 1

private theorem fieldOfNat_val (value : F) :
    fieldOfNat value.val = value := by
  apply Fin.ext
  simp [fieldOfNat, Nat.mod_eq_of_lt value.isLt]

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
    have complementPositive : 0 < goldilocksModulus - value.val := by
      omega
    have complementLt :
        goldilocksModulus - value.val < goldilocksModulus := by
      omega
    have embeddedNonzero :
        fieldOfNat (goldilocksModulus - value.val) ≠ 0 := by
      intro equal
      have valuesEqual := congrArg Fin.val equal
      simp [fieldOfNat, Nat.mod_eq_of_lt complementLt] at valuesEqual
      omega
    apply Fin.ext
    rw [Fin.val_neg, if_neg embeddedNonzero]
    simp [fieldOfNat, Nat.mod_eq_of_lt complementLt]
    omega

/-! ## Exact fixed-position binary digits -/

/-- Natural binary digit at verifier-owned little-endian position `index`. -/
def natBit (value index : Nat) : Nat :=
  (value / 2 ^ index) % 2

@[simp] private theorem natBit_zero (value : Nat) :
    natBit value 0 = value % 2 := by
  simp [natBit]

private theorem natBit_succ (value index : Nat) :
    natBit value (index + 1) = natBit (value / 2) index := by
  simp [natBit, Nat.div_div_eq_div_mul, Nat.pow_succ']

theorem natBit_lt_two (value index : Nat) : natBit value index < 2 := by
  exact Nat.mod_lt _ (by decide)

/-- One binary digit of a centered coefficient magnitude. -/
def magnitudeDigit (value : F) (index : ChildIndex) : Nat :=
  natBit (centeredMagnitude value) index.val

theorem magnitudeDigit_lt_two (value : F) (index : ChildIndex) :
    magnitudeDigit value index < 2 := by
  exact natBit_lt_two _ _

/-- Head-first natural sum over a fixed verifier-owned child count. -/
private def sumNats : {count : Nat} -> (Fin count -> Nat) -> Nat
  | 0, _ => 0
  | _ + 1, values =>
      values 0 + sumNats (fun index => values index.succ)

private theorem sumNats_mul (factor : Nat) {count : Nat}
    (values : Fin count -> Nat) :
    sumNats (fun index => factor * values index) =
      factor * sumNats values := by
  induction count with
  | zero => simp [sumNats]
  | succ count inductionHypothesis =>
      simp only [sumNats]
      rw [inductionHypothesis (fun index => values index.succ)]
      exact (Nat.mul_add factor _ _).symm

/-- Core fixed-radix identity. It uses only natural division and remainder,
not an imported digit library. -/
private theorem sumBits_reconstruct (count value : Nat)
    (bounded : value < 2 ^ count) :
    sumNats (fun index : Fin count =>
      2 ^ index.val * natBit value index.val) = value := by
  induction count generalizing value with
  | zero =>
      have valueZero : value = 0 := by simpa using bounded
      subst value
      rfl
  | succ count inductionHypothesis =>
      have tailBound : value / 2 < 2 ^ count := by
        apply (Nat.div_lt_iff_lt_mul (by decide)).2
        simpa [Nat.pow_succ] using bounded
      simp only [sumNats, Fin.val_zero, Nat.pow_zero, Nat.one_mul,
        natBit_zero]
      have tailTerms :
          (fun index : Fin count =>
              2 ^ index.succ.val * natBit value index.succ.val) =
            (fun index : Fin count =>
              2 * (2 ^ index.val * natBit (value / 2) index.val)) := by
        funext index
        simp only [Fin.val_succ]
        rw [natBit_succ]
        simp [Nat.pow_succ', Nat.mul_assoc]
      rw [tailTerms, sumNats_mul,
        inductionHypothesis (value / 2) tailBound]
      exact Nat.mod_add_div value 2

/-! ## Field recomposition -/

private def combineScalars : {count : Nat} ->
    (Fin count -> F) -> (Fin count -> F) -> F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        combineScalars
          (fun index => weights index.succ)
          (fun index => values index.succ)

private theorem combineAssignments_apply {shape : Shape} {count : Nat}
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape)
    (column : Fin shape.carrierWidth) :
    BaseLinear.combineAssignments weights assignments column =
      combineScalars weights (fun index => assignments index column) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineAssignments, BaseLinear.assignmentAdd,
        BaseLinear.assignmentScale, BaseLinear.Raw.assignmentAdd,
        BaseLinear.Raw.assignmentScale, combineScalars]
      rw [inductionHypothesis]

private theorem combineScalars_fieldOfNat {count : Nat}
    (weights digits : Fin count -> Nat) :
    combineScalars
        (fun index => fieldOfNat (weights index))
        (fun index => fieldOfNat (digits index)) =
      fieldOfNat (sumNats fun index => weights index * digits index) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [combineScalars, sumNats]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => digits index.succ)]
      rw [← fieldOfNat_mul, ← fieldOfNat_add]

private theorem radixWeight_eq_fieldOfNat (index : ChildIndex) :
    EvaluationHomomorphism.PiDEC.radixWeight index =
      fieldOfNat (2 ^ index.val) := by
  rfl

private theorem combineMagnitudeDigits (value : F)
    (bounded : centeredMagnitude value < combinedBound) :
    combineScalars EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => fieldOfNat (magnitudeDigit value index)) =
      fieldOfNat (centeredMagnitude value) := by
  rw [show EvaluationHomomorphism.PiDEC.radixWeight =
      (fun index : ChildIndex => fieldOfNat (2 ^ index.val)) by
    funext index
    exact radixWeight_eq_fieldOfNat index]
  rw [combineScalars_fieldOfNat]
  apply congrArg fieldOfNat
  apply sumBits_reconstruct
  simpa [combinedBound, productionGlobalParams, GlobalParams.bigB] using bounded

private theorem combineScalars_neg {count : Nat}
    (weights values : Fin count -> F) :
    combineScalars weights (fun index => -(values index)) =
      -(combineScalars weights values) := by
  induction count with
  | zero => exact Lean.Grind.AddCommGroup.neg_zero.symm
  | succ count inductionHypothesis =>
      simp only [combineScalars]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      have mulNeg : weights 0 * -values 0 = -(weights 0 * values 0) := by
        calc
          weights 0 * -values 0 = -values 0 * weights 0 :=
            Fin.mul_comm _ _
          _ = -(values 0 * weights 0) :=
            Lean.Grind.Fin.neg_mul _ _
          _ = -(weights 0 * values 0) := by
            rw [Fin.mul_comm (values 0) (weights 0)]
      rw [mulNeg]
      exact (Lean.Grind.AddCommGroup.neg_add _ _).symm

/-! ## Total signed split -/

/-- The lower centered half has a nonnegative canonical representative. -/
def isNonnegative (value : F) : Prop :=
  value.val <= Centered.halfModulus

instance (value : F) : Decidable (isNonnegative value) :=
  inferInstanceAs (Decidable (value.val <= Centered.halfModulus))

/-- One signed binary digit in the active Goldilocks field. -/
def boundedDigit (value : F) (index : ChildIndex) : F :=
  if isNonnegative value then
    fieldOfNat (magnitudeDigit value index)
  else
    -(fieldOfNat (magnitudeDigit value index))

/-- Total outside-bound fallback. Child zero retains the coefficient and all
other children are zero. It is exact but deliberately not short. -/
def fallbackDigit (value : F) (index : ChildIndex) : F :=
  if index.val = 0 then value else 0

/-- Deterministic total scalar split. Only the bounded branch carries a short
digit guarantee. -/
def splitScalar (value : F) (index : ChildIndex) : F :=
  if centeredMagnitude value < combinedBound then
    boundedDigit value index
  else
    fallbackDigit value index

/-- Coordinatewise split of the complete typed assignment. -/
def splitAssignment {shape : Shape}
    (assignment : Assignment shape) (index : ChildIndex) : Assignment shape :=
  fun column => splitScalar (assignment column) index

/-- Recomposition is the production base-field operation already used by the
exact evaluation homomorphism. -/
abbrev recomposeAssignment {shape : Shape}
    (assignments : ChildIndex -> Assignment shape) : Assignment shape :=
  EvaluationHomomorphism.PiDEC.recomposeAssignment assignments

private theorem centeredMagnitude_eq_val_of_nonnegative (value : F)
    (nonnegative : isNonnegative value) :
    centeredMagnitude value = value.val := by
  unfold isNonnegative at nonnegative
  rw [Centered.centeredMagnitude_eq_distance]
  simp [Centered.distance, nonnegative]

private theorem centeredMagnitude_eq_complement_of_negative (value : F)
    (negative : ¬ isNonnegative value) :
    centeredMagnitude value = goldilocksModulus - value.val := by
  unfold isNonnegative at negative
  rw [Centered.centeredMagnitude_eq_distance]
  simp [Centered.distance, negative]

private theorem boundedDigit_recompose (value : F)
    (bounded : centeredMagnitude value < combinedBound) :
    combineScalars EvaluationHomomorphism.PiDEC.radixWeight
        (boundedDigit value) = value := by
  by_cases nonnegative : isNonnegative value
  · rw [show boundedDigit value =
        (fun index => fieldOfNat (magnitudeDigit value index)) by
      funext index
      simp [boundedDigit, nonnegative]]
    rw [combineMagnitudeDigits value bounded,
      centeredMagnitude_eq_val_of_nonnegative value nonnegative,
      fieldOfNat_val]
  · rw [show boundedDigit value =
        (fun index => -(fieldOfNat (magnitudeDigit value index))) by
      funext index
      simp [boundedDigit, nonnegative]]
    rw [combineScalars_neg, combineMagnitudeDigits value bounded,
      centeredMagnitude_eq_complement_of_negative value nonnegative,
      neg_fieldOfNat_complement]

private theorem combineScalars_zero {count : Nat} (weights : Fin count -> F) :
    combineScalars weights (fun _ => 0) = 0 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [combineScalars]
      rw [Fin.mul_zero,
        inductionHypothesis (fun index => weights index.succ), Fin.zero_add]

private theorem combineScalars_head_only {count : Nat}
    (weights values : Fin (count + 1) -> F) (value : F)
    (head : values 0 = value)
    (tail : forall index : Fin count, values index.succ = 0) :
    combineScalars weights values = weights 0 * value := by
  rw [combineScalars, head]
  have tailZero :
      (fun index : Fin count => values index.succ) = (fun _ => 0) := by
    funext index
    exact tail index
  rw [tailZero, combineScalars_zero, Fin.add_zero]

private theorem fallbackDigit_recompose (value : F) :
    combineScalars EvaluationHomomorphism.PiDEC.radixWeight
        (fallbackDigit value) = value := by
  change combineScalars
      (count := 13 + 1)
      EvaluationHomomorphism.PiDEC.radixWeight
      (fallbackDigit value) = value
  rw [combineScalars_head_only
    EvaluationHomomorphism.PiDEC.radixWeight (fallbackDigit value) value
    (by simp [fallbackDigit])
    (by intro index; simp [fallbackDigit])]
  change (1 : F) * value = value
  rw [Fin.one_mul]

/-- Scalar form of the verifier-owned production assignment recomposition. -/
def recomposeScalar (values : ChildIndex -> F) : F :=
  combineScalars EvaluationHomomorphism.PiDEC.radixWeight values

theorem splitScalar_recompose (value : F) :
    recomposeScalar (splitScalar value) = value := by
  unfold recomposeScalar
  by_cases bounded : centeredMagnitude value < combinedBound
  · rw [show splitScalar value = boundedDigit value by
      funext index
      simp [splitScalar, bounded]]
    exact boundedDigit_recompose value bounded
  · rw [show splitScalar value = fallbackDigit value by
      funext index
      simp [splitScalar, bounded]]
    exact fallbackDigit_recompose value

/-- Unconditional exact recomposition required by `PiDEC.Algebra`. -/
theorem split_recompose {shape : Shape} (assignment : Assignment shape) :
    recomposeAssignment (splitAssignment assignment) = assignment := by
  funext column
  unfold recomposeAssignment EvaluationHomomorphism.PiDEC.recomposeAssignment
  rw [combineAssignments_apply]
  exact splitScalar_recompose (assignment column)

/-! ## Split norm -/

private theorem centeredMagnitude_fieldOfNat_lt_two {digit : Nat}
    (digitLt : digit < 2) :
    centeredMagnitude (fieldOfNat digit) < 2 := by
  have cases : digit = 0 ∨ digit = 1 := by omega
  rcases cases with rfl | rfl <;> decide

theorem boundedDigit_norm (value : F) (index : ChildIndex) :
    centeredMagnitude (boundedDigit value index) <
      productionGlobalParams.b := by
  have digitBound := centeredMagnitude_fieldOfNat_lt_two
    (magnitudeDigit_lt_two value index)
  by_cases nonnegative : isNonnegative value
  · simpa [boundedDigit, nonnegative, productionGlobalParams] using digitBound
  · simpa [boundedDigit, nonnegative, productionGlobalParams,
      Centered.centeredMagnitude_neg] using digitBound

theorem splitScalar_norm (value : F)
    (bounded : centeredMagnitude value < combinedBound)
    (index : ChildIndex) :
    centeredMagnitude (splitScalar value index) < productionGlobalParams.b := by
  simpa [splitScalar, bounded] using boundedDigit_norm value index

/-- A strictly combined-bounded complete assignment splits into fourteen
strictly fresh-bounded assignments. -/
theorem split_norm {shape : Shape} (assignment : Assignment shape)
    (bounded : assignmentNormBounded productionGlobalParams.bigB assignment) :
    forall index,
      assignmentNormBounded productionGlobalParams.b
        (splitAssignment assignment index) := by
  intro index column
  apply splitScalar_norm
  simpa [combinedBound] using bounded column

/-! ## Recomposition norm -/

private theorem sumNats_mono {count : Nat}
    {left right : Fin count -> Nat}
    (pointwise : forall index, left index <= right index) :
    sumNats left <= sumNats right := by
  induction count with
  | zero => exact Nat.le_refl 0
  | succ count inductionHypothesis =>
      simp only [sumNats]
      exact Nat.add_le_add (pointwise 0)
        (inductionHypothesis (fun index => pointwise index.succ))

private theorem centeredMagnitude_combineScalars_le {count : Nat}
    (weights values : Fin count -> F) :
    centeredMagnitude (combineScalars weights values) <=
      sumNats fun index => centeredMagnitude (weights index * values index) := by
  induction count with
  | zero => simp [combineScalars, sumNats, Centered.centeredMagnitude_zero]
  | succ count inductionHypothesis =>
      simp only [combineScalars, sumNats]
      calc
        centeredMagnitude
            (weights 0 * values 0 +
              combineScalars
                (fun index => weights index.succ)
                (fun index => values index.succ)) <=
            centeredMagnitude (weights 0 * values 0) +
              centeredMagnitude
                (combineScalars
                  (fun index => weights index.succ)
                  (fun index => values index.succ)) :=
          Centered.centeredMagnitude_add_le _ _
        _ <= centeredMagnitude (weights 0 * values 0) +
              sumNats (fun index =>
                centeredMagnitude
                  (weights index.succ * values index.succ)) := by
          exact Nat.add_le_add_left
            (inductionHypothesis
              (fun index => weights index.succ)
              (fun index => values index.succ)) _

private theorem centeredMagnitude_fieldOfNat_mul_le
    (factor : Nat) (value : F) :
    centeredMagnitude (fieldOfNat factor * value) <=
      factor * centeredMagnitude value := by
  induction factor with
  | zero =>
      rw [fieldOfNat_zero, Fin.zero_mul,
        Centered.centeredMagnitude_zero, Nat.zero_mul]
      exact Nat.le_refl 0
  | succ factor inductionHypothesis =>
      have expand : fieldOfNat (factor + 1) * value =
          fieldOfNat factor * value + value := by
        rw [fieldOfNat_succ]
        calc
          (fieldOfNat factor + 1) * value =
              value * (fieldOfNat factor + 1) := Fin.mul_comm _ _
          _ = value * fieldOfNat factor + value * 1 :=
            Lean.Grind.Fin.left_distrib _ _ _
          _ = fieldOfNat factor * value + value := by
            rw [Fin.mul_comm value (fieldOfNat factor),
              Lean.Grind.Fin.mul_one]
      rw [expand]
      calc
        centeredMagnitude (fieldOfNat factor * value + value) <=
            centeredMagnitude (fieldOfNat factor * value) +
              centeredMagnitude value :=
          Centered.centeredMagnitude_add_le _ _
        _ <= factor * centeredMagnitude value + centeredMagnitude value := by
          exact Nat.add_le_add_right inductionHypothesis _
        _ = (factor + 1) * centeredMagnitude value := by
          simpa [Nat.succ_eq_add_one] using
            (Nat.succ_mul factor (centeredMagnitude value)).symm

private theorem production_power_sum :
    sumNats (fun index : ChildIndex => 2 ^ index.val) =
      productionGlobalParams.bigB - 1 := by
  decide

private theorem recomposeScalar_norm
    (digits : ChildIndex -> F)
    (bounded : forall index,
      centeredMagnitude (digits index) < productionGlobalParams.b) :
    centeredMagnitude
        (combineScalars EvaluationHomomorphism.PiDEC.radixWeight digits) <
      productionGlobalParams.bigB := by
  have combinationBound := centeredMagnitude_combineScalars_le
    EvaluationHomomorphism.PiDEC.radixWeight digits
  have termsBound : forall index : ChildIndex,
      centeredMagnitude
          (EvaluationHomomorphism.PiDEC.radixWeight index * digits index) <=
        2 ^ index.val := by
    intro index
    rw [radixWeight_eq_fieldOfNat]
    calc
      centeredMagnitude (fieldOfNat (2 ^ index.val) * digits index) <=
          2 ^ index.val * centeredMagnitude (digits index) :=
        centeredMagnitude_fieldOfNat_mul_le _ _
      _ <= 2 ^ index.val := by
        have digitAtMostOne : centeredMagnitude (digits index) <= 1 := by
          have digitLt := bounded index
          simp only [productionGlobalParams] at digitLt
          omega
        simpa using Nat.mul_le_mul_left (2 ^ index.val) digitAtMostOne
  have sumBound := sumNats_mono termsBound
  rw [production_power_sum] at sumBound
  exact Nat.lt_of_le_of_lt (Nat.le_trans combinationBound sumBound) (by
    have positive : 0 < productionGlobalParams.bigB := by decide
    omega)

/-- Any fourteen strict-`2` assignments recompose to a strict-`16384`
assignment, independently of whether they came from `splitAssignment`. -/
theorem recompose_norm {shape : Shape}
    (assignments : ChildIndex -> Assignment shape)
    (bounded : forall index,
      assignmentNormBounded productionGlobalParams.b (assignments index)) :
    assignmentNormBounded productionGlobalParams.bigB
      (recomposeAssignment assignments) := by
  intro column
  unfold recomposeAssignment EvaluationHomomorphism.PiDEC.recomposeAssignment
  rw [combineAssignments_apply]
  apply recomposeScalar_norm
  intro index
  exact bounded index column

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
