import Mathlib.Data.Nat.Digits.Lemmas
import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns the exact canonical 64-bit decomposition of one Goldilocks field value.

Inputs:
- one caller-owned field expression below the allocation offset.

Outputs:
- 64 little-endian Boolean bits;
- one inverse hint and one derived high-word flag.

Invariants:
- the bits recompose to the input as one integer below the modulus;
- `0xffffffff : high32` forces `low32 = 0`, so `x + p` is rejected;
- hints are witness computations only. Every accepted value is bound by rows.

Provenance: the quadratic high-word flag schedule follows
`formal/nightstream-lean/Nightstream/Implementation/R1CS/Canonical/
CanonicalU64Recipe.lean` at commit
`fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`. The circuit and proofs below
are adapted to the F′ `FormalCircuit` DSL and use no frozen module.
-/

namespace NightstreamFPrime.Gadgets.Range.CanonicalU64

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.GoldilocksPrime
open NightstreamFPrime.Circuit

def bitCount : Nat := 64
def halfBitCount : Nat := 32
def auxiliaryCount : Nat := 66
def exactRowCount : Nat := 67
def highMax : Nat := 4294967295

structure Interface where
  source : Nat → Expr

def bitExpr (offset index : Nat) : Expr :=
  Expr.var (offset + index)

def inverseExpr (offset : Nat) : Expr :=
  Expr.var (offset + bitCount)

def highFlagExpr (offset : Nat) : Expr :=
  Expr.var (offset + bitCount + 1)

def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def weightedExpr (offset bitStart count : Nat) : Expr :=
  (List.range count).foldl (fun value index =>
    value + Expr.const (fieldOfNat (2 ^ index)) *
      bitExpr offset (bitStart + index)) 0

def lowExpr (offset : Nat) : Expr :=
  weightedExpr offset 0 halfBitCount

def highExpr (offset : Nat) : Expr :=
  weightedExpr offset halfBitCount halfBitCount

def wordExpr (offset : Nat) : Expr :=
  lowExpr offset + Expr.const (fieldOfNat (2 ^ halfBitCount)) * highExpr offset

def highDifferenceExpr (offset : Nat) : Expr :=
  highExpr offset - Expr.const (fieldOfNat highMax)

def flagRecipe (offset : Nat) : Expr :=
  1 - highDifferenceExpr offset * inverseExpr offset

def bitHints (interface : Interface) (offset : Nat) : List Hint :=
  (List.range bitCount).map fun index =>
    .bit (interface.source offset) index

def inverseHint (offset : Nat) : Hint :=
  .inverseOrZero (highDifferenceExpr offset)

def booleanConstraint (offset index : Nat) : Expr :=
  bitExpr offset index * (bitExpr offset index - 1)

def booleanOps (offset : Nat) : List Op :=
  (List.range bitCount).map fun index =>
    .assertZero (booleanConstraint offset index)

def booleanConstraints (offset : Nat) : List Expr :=
  (List.range bitCount).map fun index => booleanConstraint offset index

def recompositionConstraint (interface : Interface) (offset : Nat) : Expr :=
  interface.source offset - wordExpr offset

def canonicalityConstraint (offset : Nat) : Expr :=
  highFlagExpr offset * lowExpr offset

private theorem booleanConstraint_eval (env : Env) (offset index : Nat) :
    (booleanConstraint offset index).eval env =
      env (offset + index) * (env (offset + index) - 1) := by
  unfold booleanConstraint
  calc
    (bitExpr offset index * (bitExpr offset index - 1)).eval env =
        (bitExpr offset index).eval env *
          (bitExpr offset index - 1).eval env :=
      Expr.eval_mul env _ _
    _ = env (offset + index) * (env (offset + index) - 1) := by
      rw [Expr.eval_sub]
      rfl

private theorem canonicalityConstraint_eval (env : Env) (offset : Nat) :
    (canonicalityConstraint offset).eval env =
      (highFlagExpr offset).eval env * (lowExpr offset).eval env := by
  unfold canonicalityConstraint
  exact Expr.eval_mul env _ _

private theorem flagRecipe_eval (env : Env) (offset : Nat) :
    (flagRecipe offset).eval env =
      (1 : F) - (highDifferenceExpr offset).eval env *
        (inverseExpr offset).eval env := by
  unfold flagRecipe
  calc
    (1 - highDifferenceExpr offset * inverseExpr offset : Expr).eval env =
        (1 : Expr).eval env -
          (highDifferenceExpr offset * inverseExpr offset).eval env :=
      Expr.eval_sub env _ _
    _ = (1 : F) -
        (highDifferenceExpr offset * inverseExpr offset).eval env := by
      rfl
    _ = (1 : F) - (highDifferenceExpr offset).eval env *
        (inverseExpr offset).eval env :=
      congrArg (fun product => (1 : F) - product)
        (Expr.eval_mul env _ _)

def operations (interface : Interface) (offset : Nat) : List Op :=
  [ .witness (WitnessBatch.hinted offset (bitHints interface offset)),
    .witness (WitnessBatch.hinted (offset + bitCount) [inverseHint offset]),
    .witness (WitnessBatch.arithmetic (offset + bitCount + 1)
      [flagRecipe offset]) ] ++
    booleanOps offset ++
    [ .assertZero (recompositionConstraint interface offset),
      .assertZero (canonicalityConstraint offset) ]

private theorem flatConstraints_booleanOps (offset : Nat) :
    flatConstraints (booleanOps offset) = booleanConstraints offset := by
  unfold flatConstraints booleanOps booleanConstraints
  generalize List.range bitCount = indices
  induction indices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp [Op.flatConstraints, inductionHypothesis]

theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints (offset + bitCount + 1) [flagRecipe offset] ++
        (booleanConstraints offset ++
          [recompositionConstraint interface offset,
            canonicalityConstraint offset]) := by
  change recipeConstraints offset [] ++
      (recipeConstraints (offset + bitCount) [] ++
        (recipeConstraints (offset + bitCount + 1) [flagRecipe offset] ++
          (flatConstraints (booleanOps offset) ++
            [recompositionConstraint interface offset,
              canonicalityConstraint offset]))) = _
  rw [flatConstraints_booleanOps]
  rfl

private theorem localLength_booleanOps (offset : Nat) :
    localLength (booleanOps offset) = 0 := by
  unfold localLength booleanOps
  change (List.map (fun _ => 0) (List.range bitCount)).sum = 0
  simp

private theorem localLength_append' (left right : List Op) :
    localLength (left ++ right) = localLength left + localLength right := by
  simp [localLength]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = auxiliaryCount := by
  unfold operations
  rw [localLength_append', localLength_append', localLength_booleanOps]
  simp [localLength, Op.localLength, WitnessBatch.outputLength,
    bitHints, auxiliaryCount, bitCount]

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = exactRowCount := by
  rw [flatConstraints_operations]
  simp [booleanConstraints, recipeConstraints, exactRowCount, bitCount]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + auxiliaryCount, operations interface offset)

def bitValue (env : Env) (offset index : Nat) : Nat :=
  (env (offset + index)).val

def weightedValue (env : Env) (offset bitStart count : Nat) : Nat :=
  (List.range count).foldl (fun value index =>
    value + 2 ^ index * bitValue env offset (bitStart + index)) 0

def lowValue (env : Env) (offset : Nat) : Nat :=
  weightedValue env offset 0 halfBitCount

def highValue (env : Env) (offset : Nat) : Nat :=
  weightedValue env offset halfBitCount halfBitCount

def wordValue (env : Env) (offset : Nat) : Nat :=
  lowValue env offset + 2 ^ halfBitCount * highValue env offset

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.source offset).VarsBelow offset

/-- Exact integer result of one accepted decomposition. -/
structure Refines (interface : Interface) (offset : Nat) (env : Env) : Prop where
  source_eq : ((interface.source offset).eval env).val = wordValue env offset
  canonical : wordValue env offset < goldilocksModulus
  bit_lt_two : ∀ index, index < bitCount → bitValue env offset index < 2

abbrev SpecHolds := Refines

@[simp] private theorem fieldOfNat_val (value : Nat) :
    (fieldOfNat value).val = value % goldilocksModulus := by
  rfl

private theorem fieldOfNat_add (left right : Nat) :
    fieldOfNat left + fieldOfNat right = fieldOfNat (left + right) := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, Fin.val_add, Nat.add_mod]

private theorem fieldOfNat_mul (left right : Nat) :
    fieldOfNat left * fieldOfNat right = fieldOfNat (left * right) := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, Fin.val_mul, Nat.mul_mod]

private theorem fieldOfNat_val_self (value : F) :
    fieldOfNat value.val = value := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, Nat.mod_eq_of_lt value.isLt]

private theorem mul_hintInverse_eq_one (value : F) (nonzero : value ≠ 0) :
    value * Hint.inverse value = 1 := by
  have valuePositive : 0 < value.val := Nat.pos_of_ne_zero (by
    intro valueValZero
    apply nonzero
    apply Fin.eq_of_val_eq
    simpa using valueValZero)
  have notDvd : ¬goldilocksModulus ∣ value.val := by
    intro divides
    have lower := Nat.le_of_dvd valuePositive divides
    exact (not_le_of_gt value.isLt) lower
  have coprime : Nat.Coprime value.val goldilocksModulus :=
    (goldilocks_natPrime.coprime_iff_not_dvd.mpr notDvd).symm
  unfold Hint.inverse
  have gcdOne : Nat.gcd value.val goldilocksModulus = 1 :=
    Nat.coprime_iff_gcd_eq_one.mp coprime
  unfold F goldilocksModulus at value gcdOne ⊢
  have law := ZMod.mul_inv_eq_gcd
    (n := 18446744069414584321) value
  dsimp [ZMod, ZMod.val] at law
  rw [gcdOne] at law
  exact law

private theorem weightedExpr_succ (offset bitStart count : Nat) :
    weightedExpr offset bitStart (count + 1) =
      weightedExpr offset bitStart count +
        Expr.const (fieldOfNat (2 ^ count)) *
          bitExpr offset (bitStart + count) := by
  simp [weightedExpr, List.range_succ, List.foldl_append]

private theorem weightedValue_succ (env : Env) (offset bitStart count : Nat) :
    weightedValue env offset bitStart (count + 1) =
      weightedValue env offset bitStart count +
        2 ^ count * bitValue env offset (bitStart + count) := by
  simp [weightedValue, List.range_succ, List.foldl_append]

private theorem weightedExpr_eval (env : Env) (offset bitStart : Nat) :
    ∀ count,
      (weightedExpr offset bitStart count).eval env =
        fieldOfNat (weightedValue env offset bitStart count)
  | 0 => by
      rfl
  | count + 1 => by
      rw [weightedExpr_succ, weightedValue_succ]
      change (weightedExpr offset bitStart count).eval env +
          fieldOfNat (2 ^ count) * env (offset + (bitStart + count)) = _
      rw [weightedExpr_eval env offset bitStart count]
      rw [← fieldOfNat_val_self (env (offset + (bitStart + count))),
        fieldOfNat_mul, fieldOfNat_add]
      rfl

private theorem weightedValue_lt_twoPow
    (env : Env) (offset bitStart : Nat)
    (binary : ∀ index, index < bitCount → bitValue env offset index ≤ 1) :
    ∀ count, bitStart + count ≤ bitCount →
      weightedValue env offset bitStart count < 2 ^ count
  | 0, _ => by simp [weightedValue]
  | count + 1, within => by
      rw [weightedValue_succ, pow_succ]
      have prior := weightedValue_lt_twoPow env offset bitStart binary count (by omega)
      have current : bitValue env offset (bitStart + count) ≤ 1 :=
        binary _ (by omega)
      have positive : 0 < 2 ^ count := Nat.two_pow_pos count
      nlinarith

private theorem wordExpr_eval (env : Env) (offset : Nat) :
    (wordExpr offset).eval env = fieldOfNat (wordValue env offset) := by
  change (weightedExpr offset 0 halfBitCount).eval env +
      fieldOfNat (2 ^ halfBitCount) *
        (weightedExpr offset halfBitCount halfBitCount).eval env = _
  rw [weightedExpr_eval env offset 0 halfBitCount,
    weightedExpr_eval env offset halfBitCount halfBitCount]
  rw [fieldOfNat_mul, fieldOfNat_add]
  rfl

theorem bitExpr_varsBelow (offset index bound : Nat)
    (below : offset + index < bound) :
    (bitExpr offset index).VarsBelow bound := by
  simpa [bitExpr, Expr.VarsBelow] using below

theorem weightedExpr_varsBelow (offset bitStart count bound : Nat)
    (within : offset + bitStart + count ≤ bound) :
    (weightedExpr offset bitStart count).VarsBelow bound := by
  induction count with
  | zero => simp [weightedExpr, Expr.VarsBelow]
  | succ count inductionHypothesis =>
      rw [weightedExpr_succ]
      apply Expr.VarsBelow.add
      · exact inductionHypothesis (by omega)
      · apply Expr.VarsBelow.mul
        · exact trivial
        · apply bitExpr_varsBelow
          omega

private theorem highDifference_varsBelow (offset : Nat) :
    (highDifferenceExpr offset).VarsBelow (offset + bitCount) := by
  apply Expr.VarsBelow.sub
  · apply weightedExpr_varsBelow
    simp [halfBitCount, bitCount]
  · exact trivial

private theorem flagRecipe_varsBelow (offset : Nat) :
    (flagRecipe offset).VarsBelow (offset + bitCount + 1) := by
  apply Expr.VarsBelow.sub
  · exact trivial
  · apply Expr.VarsBelow.mul
    · exact Expr.VarsBelow.mono _ (highDifference_varsBelow offset) (by omega)
    · simp [inverseExpr, Expr.VarsBelow]

/-- Every canonical-decomposition constraint reads only the caller-owned
source and the exact 66-value logical interval. -/
theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat)
    (sourceBelow : (interface.source offset).VarsBelow offset) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + auxiliaryCount) := by
  intro expression member
  rw [flatConstraints_operations] at member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · simp only [recipeConstraints, List.mem_singleton] at recipeMember
    subst expression
    apply Expr.VarsBelow.sub
    · simp [Expr.VarsBelow, bitCount, auxiliaryCount]
    · exact Expr.VarsBelow.mono _ (flagRecipe_varsBelow offset) (by
        simp [bitCount, auxiliaryCount])
  · rcases List.mem_append.mp assertionMember with booleanMember | finalMember
    · rcases List.mem_map.mp booleanMember with ⟨index, indexMember, rfl⟩
      have bounded := List.mem_range.mp indexMember
      unfold booleanConstraint
      apply Expr.VarsBelow.mul
      · apply bitExpr_varsBelow
        simp [bitCount, auxiliaryCount] at bounded ⊢
        omega
      · apply Expr.VarsBelow.sub
        · apply bitExpr_varsBelow
          simp [bitCount, auxiliaryCount] at bounded ⊢
          omega
        · exact trivial
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at finalMember
      rcases finalMember with rfl | rfl
      · apply Expr.VarsBelow.sub
        · exact Expr.VarsBelow.mono _ sourceBelow (by
            simp [auxiliaryCount])
        · unfold wordExpr lowExpr highExpr
          apply Expr.VarsBelow.add
          · apply weightedExpr_varsBelow
            simp [halfBitCount, bitCount, auxiliaryCount]
          · apply Expr.VarsBelow.mul
            · exact trivial
            · apply weightedExpr_varsBelow
              simp [halfBitCount, bitCount, auxiliaryCount]
      · unfold canonicalityConstraint highFlagExpr lowExpr
        apply Expr.VarsBelow.mul
        · simp [Expr.VarsBelow, bitCount, auxiliaryCount]
        · apply weightedExpr_varsBelow
          simp [halfBitCount, bitCount, auxiliaryCount]

theorem bitExpr_varsSatisfy (offset index : Nat) (allowed : Nat → Prop)
    (supported : allowed (offset + index)) :
    (bitExpr offset index).VarsSatisfy allowed := by
  exact supported

theorem weightedExpr_varsSatisfy (offset bitStart count : Nat)
    (allowed : Nat → Prop)
    (supported : ∀ index, index < count →
      allowed (offset + bitStart + index)) :
    (weightedExpr offset bitStart count).VarsSatisfy allowed := by
  induction count with
  | zero => simp [weightedExpr, Expr.VarsSatisfy]
  | succ count inductionHypothesis =>
      rw [weightedExpr_succ]
      refine ⟨inductionHypothesis (fun index bounded =>
        supported index (by omega)), ?_⟩
      exact ⟨trivial, bitExpr_varsSatisfy offset (bitStart + count) allowed (by
        simpa [Nat.add_assoc] using supported count (by omega))⟩

private theorem highDifference_varsSatisfy (offset : Nat)
    (allowed : Nat → Prop)
    (localSupported : ∀ index, index < auxiliaryCount →
      allowed (offset + index)) :
    (highDifferenceExpr offset).VarsSatisfy allowed := by
  unfold highDifferenceExpr highExpr
  apply Expr.VarsSatisfy.sub
  · simpa [Nat.add_assoc] using
      weightedExpr_varsSatisfy offset halfBitCount halfBitCount allowed (by
        intro index bounded
        simpa [Nat.add_assoc] using
          localSupported (halfBitCount + index) (by
            norm_num [halfBitCount, auxiliaryCount] at bounded ⊢
            omega))
  · trivial

private theorem flagRecipe_varsSatisfy (offset : Nat)
    (allowed : Nat → Prop)
    (localSupported : ∀ index, index < auxiliaryCount →
      allowed (offset + index)) :
    (flagRecipe offset).VarsSatisfy allowed := by
  unfold flagRecipe
  apply Expr.VarsSatisfy.sub
  · trivial
  · apply Expr.VarsSatisfy.mul
    · exact highDifference_varsSatisfy offset allowed localSupported
    · exact localSupported bitCount (by norm_num [bitCount, auxiliaryCount])

/-- Every canonical-decomposition constraint reads only the caller-supported
source and the exact 66 local columns. -/
theorem flatConstraints_varsSatisfy
    (interface : Interface) (offset : Nat) (allowed : Nat → Prop)
    (sourceSupported : (interface.source offset).VarsSatisfy allowed)
    (localSupported : ∀ index, index < auxiliaryCount →
      allowed (offset + index)) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [flatConstraints_operations] at member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · simp only [recipeConstraints, List.mem_singleton] at recipeMember
    subst expression
    apply Expr.VarsSatisfy.sub
    · exact localSupported (bitCount + 1) (by
        norm_num [bitCount, auxiliaryCount])
    · exact flagRecipe_varsSatisfy offset allowed localSupported
  · rcases List.mem_append.mp assertionMember with booleanMember | finalMember
    · rcases List.mem_map.mp booleanMember with ⟨index, indexMember, rfl⟩
      have bounded := List.mem_range.mp indexMember
      have bit := bitExpr_varsSatisfy offset index allowed (localSupported index (by
        simp only [bitCount] at bounded
        norm_num [auxiliaryCount] at bounded ⊢
        omega))
      unfold booleanConstraint
      exact Expr.VarsSatisfy.mul _ _ allowed bit
        (Expr.VarsSatisfy.sub _ _ allowed bit trivial)
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at finalMember
      rcases finalMember with rfl | rfl
      · unfold recompositionConstraint wordExpr lowExpr highExpr
        apply Expr.VarsSatisfy.sub
        · exact sourceSupported
        · apply Expr.VarsSatisfy.add
          · exact weightedExpr_varsSatisfy offset 0 halfBitCount allowed (by
              intro index bounded
              exact localSupported index (by
                norm_num [halfBitCount, auxiliaryCount] at bounded ⊢
                omega))
          · apply Expr.VarsSatisfy.mul
            · trivial
            · simpa [Nat.add_assoc] using
                weightedExpr_varsSatisfy offset halfBitCount halfBitCount allowed (by
                  intro index bounded
                  simpa [Nat.add_assoc] using
                    localSupported (halfBitCount + index) (by
                      norm_num [halfBitCount, auxiliaryCount] at bounded ⊢
                      omega))
      · unfold canonicalityConstraint highFlagExpr lowExpr
        apply Expr.VarsSatisfy.mul
        · exact localSupported (bitCount + 1) (by
            norm_num [bitCount, auxiliaryCount])
        · exact weightedExpr_varsSatisfy offset 0 halfBitCount allowed (by
            intro index bounded
            exact localSupported index (by
              norm_num [halfBitCount, auxiliaryCount] at bounded ⊢
              omega))

private theorem bitHints_length (interface : Interface) (offset : Nat) :
    (bitHints interface offset).length = bitCount := by
  simp [bitHints]

private theorem bitHints_readBelow (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    HintsReadBelow offset (bitHints interface offset) := by
  intro hint member
  rcases List.mem_map.mp member with ⟨index, _, rfl⟩
  exact assumptions

private theorem boolean_of_constraint
    (env : Env) (offset index : Nat)
    (zero : (booleanConstraint offset index).eval env = 0) :
    bitValue env offset index ≤ 1 := by
  let value := env (offset + index)
  have product : value * (value - 1) = 0 := by
    simpa [booleanConstraint, bitExpr, value] using zero
  rcases baseFieldNoZeroDivisors value (value - 1) product with
    valueZero | valueMinusOneZero
  · change value.val ≤ 1
    rw [valueZero]
    decide
  · have valueOne : value = 1 := sub_eq_zero.mp valueMinusOneZero
    change value.val ≤ 1
    rw [valueOne]
    decide

private theorem flag_recipe_equation
    (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (operations interface offset)) :
    (highFlagExpr offset).eval env = (flagRecipe offset).eval env := by
  have operationHolds := rows
    (.witness (WitnessBatch.arithmetic (offset + bitCount + 1)
      [flagRecipe offset])) (by simp [operations])
  change ConstraintsHold env
    (recipeConstraints (offset + bitCount + 1) [flagRecipe offset]) at operationHolds
  have row := operationHolds
    (Expr.var (offset + bitCount + 1) - flagRecipe offset) (by
      simp [recipeConstraints])
  simp only [Expr.eval_sub, Expr.eval_var] at row
  change (highFlagExpr offset).eval env - (flagRecipe offset).eval env = 0 at row
  exact sub_eq_zero.mp row

private theorem boolean_rows
    (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (operations interface offset)) :
    ∀ index, index < bitCount → bitValue env offset index ≤ 1 := by
  intro index bounded
  have operationMember :
      .assertZero (booleanConstraint offset index) ∈ operations interface offset := by
    unfold operations
    apply List.mem_append.mpr
    left
    apply List.mem_append.mpr
    right
    exact List.mem_map.mpr
      ⟨index, List.mem_range.mpr bounded, rfl⟩
  have operationHolds := rows (.assertZero (booleanConstraint offset index))
    operationMember
  exact boolean_of_constraint env offset index operationHolds

private theorem recomposition_row
    (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (operations interface offset)) :
    (interface.source offset).eval env = (wordExpr offset).eval env := by
  have operationMember :
      .assertZero (recompositionConstraint interface offset) ∈
        operations interface offset := by
    unfold operations
    apply List.mem_append.mpr
    right
    simp
  have operationHolds := rows
    (.assertZero (recompositionConstraint interface offset)) operationMember
  change (recompositionConstraint interface offset).eval env = 0 at operationHolds
  simp only [recompositionConstraint, Expr.eval_sub] at operationHolds
  exact sub_eq_zero.mp operationHolds

private theorem canonicality_row
    (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (operations interface offset)) :
    (highFlagExpr offset).eval env * (lowExpr offset).eval env = 0 := by
  have operationMember : .assertZero (canonicalityConstraint offset) ∈
      operations interface offset := by
    unfold operations
    apply List.mem_append.mpr
    right
    simp
  have operationHolds := rows
    (.assertZero (canonicalityConstraint offset)) operationMember
  simpa [canonicalityConstraint] using operationHolds

private theorem fieldOfNat_eq_zero_of_lt
    {value : Nat} (less : value < goldilocksModulus)
    (zero : fieldOfNat value = 0) : value = 0 := by
  have values := congrArg Fin.val zero
  simpa [fieldOfNat, Nat.mod_eq_of_lt less] using values

theorem soundness
    (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have binary := boolean_rows interface env offset rows
  have lowBound : lowValue env offset < 2 ^ halfBitCount :=
    weightedValue_lt_twoPow env offset 0 binary halfBitCount (by
      simp [halfBitCount, bitCount])
  have highBound : highValue env offset < 2 ^ halfBitCount :=
    weightedValue_lt_twoPow env offset halfBitCount binary halfBitCount (by
      simp [halfBitCount, bitCount])
  have flagEquation := flag_recipe_equation interface env offset rows
  have canonicality := canonicality_row interface env offset rows
  rcases baseFieldNoZeroDivisors _ _ canonicality with flagZero | lowFieldZero
  · have highNotMax : highValue env offset ≠ highMax := by
      intro highEqual
      have highEval := weightedExpr_eval env offset halfBitCount halfBitCount
      have differenceZero : (highDifferenceExpr offset).eval env = 0 := by
        unfold highDifferenceExpr highExpr
        simp only [Expr.eval_sub, Expr.eval_const]
        rw [highEval]
        change fieldOfNat (highValue env offset) - fieldOfNat highMax = 0
        rw [highEqual]
        exact sub_self _
      have flagOne : (highFlagExpr offset).eval env = 1 := by
        rw [flagEquation, flagRecipe_eval, differenceZero, zero_mul, sub_zero]
      have impossible : (0 : F) = 1 := flagZero.symm.trans flagOne
      exact (by decide : (0 : F) ≠ 1) impossible
    have highStrict : highValue env offset < highMax := by
      have highAtMost : highValue env offset ≤ highMax := by
        simp only [halfBitCount, highMax] at highBound ⊢
        omega
      omega
    have wordBound : wordValue env offset < goldilocksModulus := by
      simp only [wordValue, halfBitCount, highMax, goldilocksModulus] at *
      omega
    have recomposed := recomposition_row interface env offset rows
    rw [wordExpr_eval env offset] at recomposed
    have values := congrArg Fin.val recomposed
    refine ⟨?_, wordBound, ?_⟩
    · simpa [fieldOfNat, Nat.mod_eq_of_lt wordBound] using values
    · intro index bounded
      exact Nat.lt_succ_iff.mpr (binary index bounded)
  · have lowZero : lowValue env offset = 0 := by
      change (weightedExpr offset 0 halfBitCount).eval env = 0 at lowFieldZero
      rw [weightedExpr_eval env offset 0 halfBitCount] at lowFieldZero
      exact fieldOfNat_eq_zero_of_lt
        (lt_trans lowBound (by
          simp [halfBitCount, goldilocksModulus])) lowFieldZero
    have wordBound : wordValue env offset < goldilocksModulus := by
      have highAtMost : highValue env offset ≤ highMax := by
        simp only [halfBitCount, highMax] at highBound ⊢
        omega
      simp only [wordValue, lowZero, Nat.zero_add, halfBitCount,
        highMax, goldilocksModulus] at *
      omega
    have recomposed := recomposition_row interface env offset rows
    rw [wordExpr_eval env offset] at recomposed
    have values := congrArg Fin.val recomposed
    refine ⟨?_, wordBound, ?_⟩
    · simpa [fieldOfNat, Nat.mod_eq_of_lt wordBound] using values
    · intro index bounded
      exact Nat.lt_succ_iff.mpr (binary index bounded)

private theorem weightedValue_split (env : Env) (offset start count : Nat) :
    weightedValue env offset 0 (start + count) =
      weightedValue env offset 0 start +
        2 ^ start * weightedValue env offset start count := by
  induction count with
  | zero => simp [weightedValue]
  | succ count inductionHypothesis =>
      have leftStep := weightedValue_succ env offset 0 (start + count)
      have rightStep := weightedValue_succ env offset start count
      calc
        weightedValue env offset 0 (start + (count + 1)) =
            weightedValue env offset 0 ((start + count) + 1) := by
          apply congrArg (fun current => weightedValue env offset 0 current)
          omega
        _ = weightedValue env offset 0 (start + count) +
            2 ^ (start + count) * bitValue env offset (start + count) :=
          by simpa using leftStep
        _ = (weightedValue env offset 0 start +
              2 ^ start * weightedValue env offset start count) +
            2 ^ (start + count) * bitValue env offset (start + count) := by
          rw [inductionHypothesis]
        _ = weightedValue env offset 0 start +
            2 ^ start *
              (weightedValue env offset start count +
                2 ^ count * bitValue env offset (start + count)) := by
          rw [pow_add]
          ring
        _ = weightedValue env offset 0 start +
            2 ^ start * weightedValue env offset start (count + 1) := by
          rw [rightStep]

private theorem wordValue_eq_fullWeighted (env : Env) (offset : Nat) :
    wordValue env offset = weightedValue env offset 0 bitCount := by
  have split := weightedValue_split env offset halfBitCount halfBitCount
  simpa [wordValue, lowValue, highValue, halfBitCount, bitCount] using split.symm

/-- Every accepted canonical decomposition exposes the exact integer bit
window of its source value. Window weights restart at zero. -/
theorem windowValue_eq (interface : Interface) (env : Env) (offset start count : Nat)
    (refines : Refines interface offset env)
    (within : start + count ≤ bitCount) :
    weightedValue env offset start count =
      ((interface.source offset).eval env).val / 2 ^ start % 2 ^ count := by
  let low := weightedValue env offset 0 start
  let window := weightedValue env offset start count
  let tail := weightedValue env offset (start + count)
    (bitCount - (start + count))
  have binary : ∀ index, index < bitCount → bitValue env offset index ≤ 1 := by
    intro index bounded
    exact Nat.le_pred_of_lt (refines.bit_lt_two index bounded)
  have lowBound : low < 2 ^ start := by
    exact weightedValue_lt_twoPow env offset 0 binary start (by omega)
  have windowBound : window < 2 ^ count := by
    exact weightedValue_lt_twoPow env offset start binary count within
  have firstSplit := weightedValue_split env offset start count
  have secondSplit := weightedValue_split env offset (start + count)
    (bitCount - (start + count))
  have endEq : start + count + (bitCount - (start + count)) = bitCount :=
    Nat.add_sub_of_le within
  have decomposition : ((interface.source offset).eval env).val =
      low + 2 ^ start * (window + 2 ^ count * tail) := by
    rw [refines.source_eq, wordValue_eq_fullWeighted]
    rw [← endEq, secondSplit, firstSplit]
    change low + 2 ^ start * window +
      2 ^ (start + count) * tail =
        low + 2 ^ start * (window + 2 ^ count * tail)
    rw [pow_add]
    ring
  rw [decomposition, Nat.add_mul_div_left _ _ (by positivity),
    Nat.div_eq_of_lt lowBound, Nat.zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt windowBound]

def bitNat (value index : Nat) : Nat :=
  (value / 2 ^ index) % 2

def bitWindowValue (value start count : Nat) : Nat :=
  (List.range count).foldl (fun total index =>
    total + 2 ^ index * bitNat value (start + index)) 0

private theorem bitWindowValue_succ (value start count : Nat) :
    bitWindowValue value start (count + 1) =
      bitWindowValue value start count +
        2 ^ count * bitNat value (start + count) := by
  simp [bitWindowValue, List.range_succ, List.foldl_append]

private theorem bitWindowValue_zero (value count : Nat) :
    bitWindowValue value 0 count = value % 2 ^ count := by
  induction count with
  | zero => simp [bitWindowValue, Nat.mod_one]
  | succ count inductionHypothesis =>
      rw [bitWindowValue_succ, inductionHypothesis, Nat.mod_pow_succ]
      simp [bitNat]

private theorem bitNat_add (value start index : Nat) :
    bitNat value (start + index) = bitNat (value / 2 ^ start) index := by
  simp [bitNat, pow_add, Nat.div_div_eq_div_mul]

private theorem foldl_congr_mem
    {α β : Type} (items : List β) (left right : α → β → α)
    (initial : α)
    (equalStep : ∀ accumulator item, item ∈ items →
      left accumulator item = right accumulator item) :
    items.foldl left initial = items.foldl right initial := by
  induction items generalizing initial with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons,
        equalStep initial item (by simp)]
      apply inductionHypothesis
      intro accumulator current member
      exact equalStep accumulator current (by simp [member])

private theorem bitWindowValue_eq (value start count : Nat) :
    bitWindowValue value start count =
      (value / 2 ^ start) % 2 ^ count := by
  have pointwise : ∀ index ∈ List.range count,
      bitNat value (start + index) = bitNat (value / 2 ^ start) index := by
    intro index _
    exact bitNat_add value start index
  unfold bitWindowValue
  have equalFolds :
      (List.range count).foldl
          (fun total index => total + 2 ^ index * bitNat value (start + index)) 0 =
        (List.range count).foldl
          (fun total index => total + 2 ^ index *
            bitNat (value / 2 ^ start) index) 0 := by
    apply foldl_congr_mem
    intro total index member
    rw [pointwise index member]
  rw [equalFolds]
  simpa [bitWindowValue] using bitWindowValue_zero (value / 2 ^ start) count

private theorem hintBit_value
    (interface : Interface) (env : Env) (offset index : Nat) :
    (Hint.eval env (.bit (interface.source offset) index)).val =
      bitNat ((interface.source offset).eval env).val index := by
  change (((((interface.source offset).eval env).val >>> index) &&& 1) %
      goldilocksModulus) = _
  rw [Nat.and_one_is_mod, Nat.shiftRight_eq_div_pow]
  apply Nat.mod_eq_of_lt
  exact lt_trans (Nat.mod_lt _ (by decide)) (by
    norm_num [goldilocksModulus])

def completeBits (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeHints env offset (bitHints interface offset)

def completeInverse (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeHints (completeBits interface env offset) (offset + bitCount)
    [inverseHint offset]

def completeEnv (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeRecipes (completeInverse interface env offset)
    (offset + bitCount + 1) [flagRecipe offset]

private theorem completeBits_value
    (interface : Interface) (env : Env) (offset index : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : index < bitCount) :
    completeBits interface env offset (offset + index) =
      Hint.eval env (.bit (interface.source offset) index) := by
  unfold completeBits
  have position : index < (bitHints interface offset).length := by
    simpa [bitHints_length] using bounded
  have value := executeHints_value_of_readBelow env offset
    (bitHints interface offset) (bitHints_readBelow interface offset env assumptions)
    index position
  simpa [bitHints] using value

private theorem completeBits_bitValue
    (interface : Interface) (env : Env) (offset index : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : index < bitCount) :
    bitValue (completeBits interface env offset) offset index =
      bitNat ((interface.source offset).eval env).val index := by
  have bit := completeBits_value interface env offset index assumptions bounded
  unfold bitValue
  rw [bit]
  exact hintBit_value interface env offset index

private theorem completeEnv_bitValue
    (interface : Interface) (env : Env) (offset index : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : index < bitCount) :
    bitValue (completeEnv interface env offset) offset index =
      bitNat ((interface.source offset).eval env).val index := by
  have bitIndexBelowInverse : offset + index < offset + bitCount := by omega
  have bitIndexBelowRecipe : offset + index < offset + bitCount + 1 := by omega
  have afterRecipe := executeRecipes_agrees_below
    (completeInverse interface env offset) (offset + bitCount + 1)
    [flagRecipe offset] (offset + index) bitIndexBelowRecipe
  have afterInverse := executeHints_agrees_below
    (completeBits interface env offset) (offset + bitCount)
    [inverseHint offset] (offset + index) bitIndexBelowInverse
  have bit := completeBits_bitValue interface env offset index assumptions bounded
  change (completeEnv interface env offset (offset + index)).val = _
  rw [show completeEnv interface env offset (offset + index) =
      completeInverse interface env offset (offset + index) by
        exact afterRecipe,
    show completeInverse interface env offset (offset + index) =
      completeBits interface env offset (offset + index) by
        exact afterInverse]
  exact bit

private theorem completeBits_weightedValue
    (interface : Interface) (env : Env) (offset start count : Nat)
    (assumptions : Assumptions interface offset env)
    (within : start + count ≤ bitCount) :
    weightedValue (completeBits interface env offset) offset start count =
      bitWindowValue ((interface.source offset).eval env).val start count := by
  unfold weightedValue bitWindowValue
  apply foldl_congr_mem
  intro total index member
  have indexLt : index < count := List.mem_range.mp member
  rw [completeBits_bitValue interface env offset (start + index)
    assumptions (by omega)]

private theorem completeEnv_weightedValue
    (interface : Interface) (env : Env) (offset start count : Nat)
    (assumptions : Assumptions interface offset env)
    (within : start + count ≤ bitCount) :
    weightedValue (completeEnv interface env offset) offset start count =
      bitWindowValue ((interface.source offset).eval env).val start count := by
  unfold weightedValue bitWindowValue
  apply foldl_congr_mem
  intro total index member
  have indexLt : index < count := List.mem_range.mp member
  rw [completeEnv_bitValue interface env offset (start + index)
    assumptions (by omega)]

private theorem completeEnv_wordValue
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    wordValue (completeEnv interface env offset) offset =
      ((interface.source offset).eval env).val := by
  let sourceValue := ((interface.source offset).eval env).val
  have low := completeEnv_weightedValue interface env offset 0 halfBitCount
    assumptions (by simp [halfBitCount, bitCount])
  have high := completeEnv_weightedValue interface env offset halfBitCount
    halfBitCount assumptions (by simp [halfBitCount, bitCount])
  rw [bitWindowValue_eq] at low high
  simp only [pow_zero, Nat.div_one] at low
  change lowValue (completeEnv interface env offset) offset =
    sourceValue % 2 ^ halfBitCount at low
  change highValue (completeEnv interface env offset) offset =
    (sourceValue / 2 ^ halfBitCount) % 2 ^ halfBitCount at high
  have sourceCapacity : sourceValue < 2 ^ bitCount := by
    exact lt_trans ((interface.source offset).eval env).isLt (by
      norm_num [goldilocksModulus, bitCount])
  have highLt : sourceValue / 2 ^ halfBitCount < 2 ^ halfBitCount := by
    rw [Nat.div_lt_iff_lt_mul (by positivity)]
    simpa [bitCount, halfBitCount, pow_add, Nat.mul_comm] using sourceCapacity
  have highMod :
      (sourceValue / 2 ^ halfBitCount) % 2 ^ halfBitCount =
        sourceValue / 2 ^ halfBitCount := Nat.mod_eq_of_lt highLt
  have split := Nat.mod_add_div sourceValue (2 ^ halfBitCount)
  change lowValue (completeEnv interface env offset) offset +
    2 ^ halfBitCount * highValue (completeEnv interface env offset) offset =
      sourceValue
  rw [low, high, highMod]
  exact split

private theorem completeEnv_source
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    (interface.source offset).eval (completeEnv interface env offset) =
      (interface.source offset).eval env := by
  apply Expr.eval_eq_of_agree_below _ offset _ _ assumptions
  intro index below
  unfold completeEnv completeInverse completeBits
  rw [executeRecipes_agrees_below _ (offset + bitCount + 1)
      [flagRecipe offset] index (by omega),
    executeHints_agrees_below _ (offset + bitCount)
      [inverseHint offset] index (by omega),
    executeHints_agrees_below env offset (bitHints interface offset) index below]

private theorem completeEnv_spec
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    SpecHolds interface offset (completeEnv interface env offset) := by
  have word := completeEnv_wordValue interface env offset assumptions
  have source := completeEnv_source interface env offset assumptions
  refine ⟨?_, ?_, ?_⟩
  · rw [source]
    exact word.symm
  · rw [word]
    exact ((interface.source offset).eval env).isLt
  · intro index bounded
    rw [completeEnv_bitValue interface env offset index assumptions bounded]
    unfold bitNat
    exact Nat.mod_lt _ (by decide)

private theorem completeEnv_agreesOutside
    (interface : Interface) (env : Env) (offset : Nat) :
    AgreesOutside env (completeEnv interface env offset) offset auxiliaryCount := by
  have bits := executeHints_agreesOutside env offset (bitHints interface offset)
  have inverse := executeHints_agreesOutside (completeBits interface env offset)
    (offset + bitCount) [inverseHint offset]
  have flag := executeRecipes_agreesOutside (completeInverse interface env offset)
    (offset + bitCount + 1) [flagRecipe offset]
  have first := bits.append inverse
  have all := first.append flag
  simpa [completeEnv, completeInverse, completeBits, bitHints_length,
    auxiliaryCount, bitCount] using all

private theorem completeEnv_holdsFlat
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    holdsFlat (completeEnv interface env offset) (operations interface offset) := by
  have specification := completeEnv_spec interface env offset assumptions
  have binary : ∀ index, index < bitCount →
      (booleanConstraint offset index).eval
        (completeEnv interface env offset) = 0 := by
    intro index bounded
    have bitLt := specification.bit_lt_two index bounded
    have bitZeroOrOne : bitValue (completeEnv interface env offset) offset index = 0 ∨
        bitValue (completeEnv interface env offset) offset index = 1 := by omega
    rcases bitZeroOrOne with bitZero | bitOne
    · have bitFieldZero :
          completeEnv interface env offset (offset + index) = 0 := by
        apply Fin.eq_of_val_eq
        simpa [bitValue] using bitZero
      rw [booleanConstraint_eval, bitFieldZero]
      exact zero_mul _
    · have bitFieldOne :
          completeEnv interface env offset (offset + index) = 1 := by
        apply Fin.eq_of_val_eq
        simpa [bitValue] using bitOne
      rw [booleanConstraint_eval, bitFieldOne, sub_self]
      exact mul_zero _
  have recomposition :
      (recompositionConstraint interface offset).eval
        (completeEnv interface env offset) = 0 := by
    have sourceEq := specification.source_eq
    have wordBound := specification.canonical
    unfold recompositionConstraint
    simp only [Expr.eval_sub]
    rw [wordExpr_eval (completeEnv interface env offset) offset]
    have fieldEq :
        (interface.source offset).eval (completeEnv interface env offset) =
          fieldOfNat (wordValue (completeEnv interface env offset) offset) := by
      apply Fin.eq_of_val_eq
      simpa [fieldOfNat, Nat.mod_eq_of_lt wordBound] using sourceEq
    rw [fieldEq]
    exact sub_self _
  have canonicality :
      (canonicalityConstraint offset).eval
        (completeEnv interface env offset) = 0 := by
    let completed := completeEnv interface env offset
    let sourceValue := ((interface.source offset).eval env).val
    let honestLow := bitWindowValue sourceValue 0 halfBitCount
    let honestHigh := bitWindowValue sourceValue halfBitCount halfBitCount
    have low := completeEnv_weightedValue interface env offset 0 halfBitCount
      assumptions (by simp [halfBitCount, bitCount])
    have high := completeEnv_weightedValue interface env offset halfBitCount
      halfBitCount assumptions (by simp [halfBitCount, bitCount])
    have sourceWord := completeEnv_wordValue interface env offset assumptions
    have flagValue :
        (highFlagExpr offset).eval completed =
          (flagRecipe offset).eval (completeInverse interface env offset) := by
      simp [completed, completeEnv, highFlagExpr, executeRecipes, Env.set]
    by_cases highEqual : honestHigh = highMax
    · have lowZero : honestLow = 0 := by
        have sourceLt := ((interface.source offset).eval env).isLt
        change lowValue completed offset +
            2 ^ halfBitCount * highValue completed offset = sourceValue at sourceWord
        change lowValue completed offset = honestLow at low
        change highValue completed offset = honestHigh at high
        rw [low, high] at sourceWord
        change honestLow + 2 ^ halfBitCount * honestHigh = sourceValue at sourceWord
        simp only [highEqual, halfBitCount, highMax, goldilocksModulus] at sourceWord sourceLt
        omega
      have lowFieldZero : (lowExpr offset).eval completed = 0 := by
        change (weightedExpr offset 0 halfBitCount).eval completed = 0
        rw [weightedExpr_eval completed offset 0 halfBitCount]
        change fieldOfNat (lowValue completed offset) = 0
        change lowValue completed offset = honestLow at low
        rw [low]
        rw [lowZero]
        rfl
      rw [canonicalityConstraint_eval, lowFieldZero]
      exact mul_zero _
    · have highBits := completeBits_weightedValue interface env offset
        halfBitCount halfBitCount assumptions (by
          simp [halfBitCount, bitCount])
      have highDifferenceNonzero :
          (highDifferenceExpr offset).eval
            (completeBits interface env offset) ≠ 0 := by
        intro differenceZero
        have highBound : honestHigh < goldilocksModulus := by
          have bound : weightedValue (completeBits interface env offset) offset
              halfBitCount halfBitCount < 2 ^ halfBitCount :=
            weightedValue_lt_twoPow
              (completeBits interface env offset) offset halfBitCount
              (fun index bounded => by
                have bit := completeBits_bitValue interface env offset index
                  assumptions bounded
                rw [bit]
                have ltTwo :
                    bitNat ((interface.source offset).eval env).val index < 2 := by
                  unfold bitNat
                  exact Nat.mod_lt _ (by decide : 0 < 2)
                omega)
              halfBitCount (by simp [halfBitCount, bitCount])
          rw [highBits] at bound
          exact lt_trans bound (by
            norm_num [halfBitCount, goldilocksModulus])
        have highEval := weightedExpr_eval
          (completeBits interface env offset) offset halfBitCount halfBitCount
        rw [highBits] at highEval
        have differenceEq : fieldOfNat honestHigh = fieldOfNat highMax := by
          apply sub_eq_zero.mp
          simpa [highDifferenceExpr, highExpr, highEval] using differenceZero
        have values := congrArg Fin.val differenceEq
        have highMaxBound : highMax < goldilocksModulus := by
          norm_num [highMax, goldilocksModulus]
        change honestHigh % goldilocksModulus =
          highMax % goldilocksModulus at values
        rw [Nat.mod_eq_of_lt highBound,
          Nat.mod_eq_of_lt highMaxBound] at values
        exact highEqual values
      have inverseValue :
          (inverseExpr offset).eval (completeInverse interface env offset) =
            Hint.inverse
              ((highDifferenceExpr offset).eval
                (completeBits interface env offset)) := by
        have value := executeHints_value_of_readBelow
          (completeBits interface env offset) (offset + bitCount)
          [inverseHint offset] (by
            intro hint member
            simp only [List.mem_singleton] at member
            subst hint
            exact highDifference_varsBelow offset)
          0 (by simp)
        simpa [completeInverse, inverseHint, inverseExpr] using value
      have differencePreserved :
          (highDifferenceExpr offset).eval (completeBits interface env offset) =
            (highDifferenceExpr offset).eval
              (completeInverse interface env offset) := by
        apply Expr.eval_eq_of_agree_below _ (offset + bitCount) _ _
          (highDifference_varsBelow offset)
        intro index below
        exact (executeHints_agrees_below (completeBits interface env offset)
          (offset + bitCount) [inverseHint offset] index below).symm
      have differenceNonzeroAfter :
          (highDifferenceExpr offset).eval
            (completeInverse interface env offset) ≠ 0 := by
        rw [← differencePreserved]
        exact highDifferenceNonzero
      have inverseLaw :
          (highDifferenceExpr offset).eval
              (completeInverse interface env offset) *
            (inverseExpr offset).eval (completeInverse interface env offset) = 1 := by
        let difference := (highDifferenceExpr offset).eval
          (completeInverse interface env offset)
        have inverseValue' :
            (inverseExpr offset).eval (completeInverse interface env offset) =
              Hint.inverse difference := by
          rw [inverseValue, differencePreserved]
        change difference *
          (inverseExpr offset).eval (completeInverse interface env offset) = 1
        rw [inverseValue']
        exact mul_hintInverse_eq_one difference differenceNonzeroAfter
      have flagZero :
          (highFlagExpr offset).eval completed = 0 := by
        rw [flagValue, flagRecipe_eval, inverseLaw]
        exact sub_self _
      rw [canonicalityConstraint_eval, flagZero]
      exact zero_mul _
  unfold holdsFlat
  rw [flatConstraints_operations, constraintsHold_append]
  constructor
  · exact executeRecipes_holds_recipeConstraints
      (completeInverse interface env offset) (offset + bitCount + 1)
      [flagRecipe offset] (by exact ⟨flagRecipe_varsBelow offset, trivial⟩)
  · rw [constraintsHold_append]
    constructor
    · intro expression member
      rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
      exact binary index (List.mem_range.mp indexMember)
    · intro expression member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl
      · exact recomposition
      · exact canonicality

theorem complete
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  refine ⟨completeEnv interface env offset, ?_,
    completeEnv_holdsFlat interface env offset assumptions⟩
  have agrees := completeEnv_agreesOutside interface env offset
  rw [localLength_eq]
  exact agrees

theorem completeness
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  complete interface env offset assumptions

def circuit (interface : Interface) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => auxiliaryCount
    rowCount := fun _ => exactRowCount
    privateCount_eq := by
      intro offset
      exact localLength_eq interface offset
    rowCount_eq := by
      intro offset
      exact flatConstraints_length_eq interface offset
    soundness := by
      intro env offset assumptions rows
      exact soundness interface env offset assumptions rows
    completeness := by
      intro env offset assumptions specification
      exact completeness interface env offset assumptions specification }

end NightstreamFPrime.Gadgets.Range.CanonicalU64
