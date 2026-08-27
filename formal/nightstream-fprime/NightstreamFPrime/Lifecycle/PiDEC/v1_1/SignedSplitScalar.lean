import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Owns the exact signed-binary split of one PiDEC public scalar.

The caller supplies one parent expression and sixteen child-digit expressions.
One non-authoritative bit hint selects the centered sign. Eighteen rows prove
that this bit is Boolean, every digit is zero or the selected common sign, and
the exact production radix recomposition equals the parent.

This module does not own assignment-coordinate enumeration, commitment or
evaluation recomposition, child CE checks, package layout, or Rust execution.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.GoldilocksPrime
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

def signBitIndex : Nat := 63
def exactPrivateCount : Nat := 1
def exactRowCount : Nat := 18

structure Interface where
  parent : Nat → Expr
  digit : Nat → Radix.ChildIndex → Expr

def parentValue (interface : Interface) (offset : Nat) (env : Env) : F :=
  (interface.parent offset).eval env

def digitValues (interface : Interface) (offset : Nat) (env : Env) :
    Radix.ChildIndex → F :=
  fun index => (interface.digit offset index).eval env

def signBitExpr (offset : Nat) : Expr :=
  Expr.var offset

def signExpr (offset : Nat) : Expr :=
  1 - 2 * signBitExpr offset

def signHint (interface : Interface) (offset : Nat) : Hint :=
  .bit (interface.parent offset) signBitIndex

def signConstraint (offset : Nat) : Expr :=
  signBitExpr offset * (signBitExpr offset - 1)

def digitConstraint (interface : Interface) (offset : Nat)
    (index : Radix.ChildIndex) : Expr :=
  interface.digit offset index *
    (interface.digit offset index - signExpr offset)

def digitConstraints (interface : Interface) (offset : Nat) : List Expr :=
  List.ofFn fun index : Radix.ChildIndex =>
    digitConstraint interface offset index

def recomposeExpr (interface : Interface) (offset : Nat) : Expr :=
  ((List.ofFn fun index : Radix.ChildIndex => interface.digit offset index).zip
      (List.ofFn fun index : Radix.ChildIndex =>
        EvaluationHomomorphism.PiDEC.radixWeight index)).foldr
    (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0

def recompositionConstraint (interface : Interface) (offset : Nat) : Expr :=
  recomposeExpr interface offset - interface.parent offset

def constraints (interface : Interface) (offset : Nat) : List Expr :=
  signConstraint offset ::
    digitConstraints interface offset ++
      [recompositionConstraint interface offset]

def operations (interface : Interface) (offset : Nat) : List Op :=
  .witness (WitnessBatch.hinted offset [signHint interface offset]) ::
    (constraints interface offset).map .assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + exactPrivateCount, operations interface offset)

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.parent offset).VarsBelow offset ∧
    ∀ index, (interface.digit offset index).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∃ sign, Radix.UniformSignedDigits.Accepted
    (parentValue interface offset env) sign
    (digitValues interface offset env)

private theorem weightedFold_eval (env : Env) :
    ∀ (values : List Expr) (weights : List F),
      ((values.zip weights).foldr
          (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        ((values.map fun value => value.eval env).zip weights).foldr
          (fun pair suffix => pair.2 * pair.1 + suffix) 0
  | [], _ => by rfl
  | _ :: _, [] => by rfl
  | value :: values, weight :: weights => by
      change weight * value.eval env +
          ((values.zip weights).foldr
            (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        weight * value.eval env +
          (((values.map fun item => item.eval env).zip weights).foldr
            (fun pair suffix => pair.2 * pair.1 + suffix) 0)
      rw [weightedFold_eval env values weights]

theorem recomposeExpr_eval (interface : Interface) (offset : Nat) (env : Env) :
    (recomposeExpr interface offset).eval env =
      Radix.recomposeScalar (digitValues interface offset env) := by
  unfold recomposeExpr
  rw [weightedFold_eval]
  rw [List.map_ofFn]
  exact Radix.recomposeScalarList_eq (digitValues interface offset env)

private theorem flatConstraints_assertions (items : List Expr) :
    flatConstraints (items.map .assertZero) = items := by
  induction items with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change expression :: flatConstraints (rest.map .assertZero) =
        expression :: rest
      rw [inductionHypothesis]

private theorem localLength_assertions (items : List Expr) :
    localLength (items.map .assertZero) = 0 := by
  induction items with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map .assertZero) = 0
      simpa using inductionHypothesis

theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      constraints interface offset := by
  change recipeConstraints offset [] ++
      flatConstraints ((constraints interface offset).map .assertZero) =
    constraints interface offset
  rw [flatConstraints_assertions]
  rfl

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = exactPrivateCount := by
  change 1 + localLength
      ((constraints interface offset).map .assertZero) = exactPrivateCount
  rw [localLength_assertions]
  rfl

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = exactRowCount := by
  rw [flatConstraints_operations]
  simp [constraints, digitConstraints, exactRowCount, Radix.ChildIndex,
    productionGlobalParams]

private theorem weightedFold_varsBelow (bound : Nat) :
    ∀ (values : List Expr) (weights : List F),
      (∀ value ∈ values, value.VarsBelow bound) →
      ((values.zip weights).foldr
        (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).VarsBelow bound
  | [], _, _ => trivial
  | _ :: _, [], _ => trivial
  | value :: values, _ :: weights, below =>
      Expr.VarsBelow.add _ _ bound
        (Expr.VarsBelow.mul _ _ bound trivial (below value (by simp)))
        (weightedFold_varsBelow bound values weights
          (fun item member => below item (by simp [member])))

private theorem signBitExpr_varsBelow (offset : Nat) :
    (signBitExpr offset).VarsBelow (offset + exactPrivateCount) := by
  simp [signBitExpr, Expr.VarsBelow, exactPrivateCount]

private theorem signExpr_varsBelow (offset : Nat) :
    (signExpr offset).VarsBelow (offset + exactPrivateCount) := by
  exact Expr.VarsBelow.sub _ _ _ trivial
    (Expr.VarsBelow.mul _ _ _ trivial (signBitExpr_varsBelow offset))

theorem recomposeExpr_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    (recomposeExpr interface offset).VarsBelow
      (offset + exactPrivateCount) := by
  unfold recomposeExpr
  apply weightedFold_varsBelow
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact Expr.VarsBelow.mono (interface.digit offset index)
    (lower := offset) (upper := offset + exactPrivateCount)
    (assumptions.2 index) (by omega)

theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + exactPrivateCount) := by
  rw [flatConstraints_operations]
  intro expression member
  rw [constraints] at member
  rcases List.mem_cons.mp member with signMember | tailMember
  · subst expression
    exact Expr.VarsBelow.mul _ _ _ (signBitExpr_varsBelow offset)
      (Expr.VarsBelow.sub _ _ _ (signBitExpr_varsBelow offset) trivial)
  · rcases List.mem_append.mp tailMember with digitMember | recompositionMember
    · rcases List.mem_ofFn.mp digitMember with ⟨index, rfl⟩
      have digitBelow := Expr.VarsBelow.mono (interface.digit offset index)
        (lower := offset) (upper := offset + exactPrivateCount)
        (assumptions.2 index) (by omega)
      exact Expr.VarsBelow.mul _ _ _ digitBelow
        (Expr.VarsBelow.sub _ _ _ digitBelow (signExpr_varsBelow offset))
    · have exactRecomposition := List.mem_singleton.mp recompositionMember
      subst expression
      exact Expr.VarsBelow.sub _ _ _
        (recomposeExpr_varsBelow interface offset assumptions)
        (Expr.VarsBelow.mono (interface.parent offset)
          (lower := offset) (upper := offset + exactPrivateCount)
          assumptions.1 (by omega))

private theorem constraintsHold_of_holds
    (interface : Interface) (offset : Nat) (env : Env)
    (rows : holds env (operations interface offset)) :
    ConstraintsHold env (constraints interface offset) := by
  intro expression member
  exact rows (.assertZero expression) (by
    simp [operations, member])

@[simp] private theorem exprOne_eval (env : Env) :
    ((1 : Expr).eval env) = (1 : F) := by
  change Radix.fieldOfNat 1 = (1 : F)
  exact Radix.fieldOfNat_one

@[simp] private theorem exprTwo_eval (env : Env) :
    ((2 : Expr).eval env) = (2 : F) := by
  change Radix.fieldOfNat 2 = (2 : F)
  rfl

private theorem signExpr_eval (env : Env) (offset : Nat) :
    (signExpr offset).eval env = (1 : F) - 2 * env offset := by
  simp only [signExpr, signBitExpr, Expr.eval_sub, Expr.eval_hmul,
    Expr.eval_var, exprOne_eval, exprTwo_eval]

private theorem signConstraint_eval (env : Env) (offset : Nat) :
    (signConstraint offset).eval env =
      env offset * (env offset - 1) := by
  simp only [signConstraint, signBitExpr, Expr.eval_hmul, Expr.eval_sub,
    Expr.eval_var, exprOne_eval]

private theorem digitConstraint_eval
    (interface : Interface) (offset : Nat) (index : Radix.ChildIndex)
    (env : Env) :
    (digitConstraint interface offset index).eval env =
      digitValues interface offset env index *
        (digitValues interface offset env index -
          (signExpr offset).eval env) := by
  simp only [digitConstraint, digitValues, Expr.eval_hmul, Expr.eval_sub]

private theorem recompositionConstraint_eval
    (interface : Interface) (offset : Nat) (env : Env) :
    (recompositionConstraint interface offset).eval env =
      Radix.recomposeScalar (digitValues interface offset env) -
        parentValue interface offset env := by
  simp only [recompositionConstraint, Expr.eval_sub, recomposeExpr_eval,
    parentValue]

@[simp] private theorem one_sub_two_eq_neg_one :
    (1 : F) - 2 = -1 := by
  decide

private theorem sign_root
    (interface : Interface) (offset : Nat) (env : Env)
    (rows : ConstraintsHold env (constraints interface offset)) :
    (signExpr offset).eval env = 1 ∨ (signExpr offset).eval env = -1 := by
  have zero := rows (signConstraint offset) (by simp [constraints])
  have product : env offset * (env offset - 1) = 0 := by
    simpa [signConstraint, signBitExpr] using zero
  rcases baseFieldNoZeroDivisors _ _ product with bitZero | bitOne
  · left
    rw [signExpr_eval, bitZero]
    decide
  · right
    have bitOne' : env offset = 1 := sub_eq_zero.mp bitOne
    rw [signExpr_eval, bitOne']
    exact one_sub_two_eq_neg_one

private theorem digit_root
    (interface : Interface) (offset : Nat) (env : Env)
    (rows : ConstraintsHold env (constraints interface offset))
    (index : Radix.ChildIndex) :
    digitValues interface offset env index = 0 ∨
      digitValues interface offset env index = (signExpr offset).eval env := by
  have member : digitConstraint interface offset index ∈
      digitConstraints interface offset := by
    exact List.mem_ofFn.mpr ⟨index, rfl⟩
  have zero := rows (digitConstraint interface offset index) (by
    rw [constraints]
    exact List.mem_cons_of_mem _ (List.mem_append_left _ member))
  have product :
      digitValues interface offset env index *
        (digitValues interface offset env index -
          (signExpr offset).eval env) = 0 := by
    simpa [digitConstraint, digitValues] using zero
  rcases baseFieldNoZeroDivisors _ _ product with inactive | active
  · exact Or.inl inactive
  · exact Or.inr (sub_eq_zero.mp active)

theorem soundness
    (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have constraintRows := constraintsHold_of_holds interface offset env rows
  refine ⟨(signExpr offset).eval env, ?_⟩
  constructor
  · constructor
    · rcases sign_root interface offset env constraintRows with positive | negative
      · exact Or.inr (Or.inl positive)
      · exact Or.inr (Or.inr negative)
    · exact digit_root interface offset env constraintRows
  · have zero := constraintRows
      (recompositionConstraint interface offset) (by simp [constraints])
    have equation :
        (recomposeExpr interface offset).eval env =
          (interface.parent offset).eval env := by
      apply sub_eq_zero.mp
      simpa [recompositionConstraint] using zero
    simpa [parentValue, digitValues, recomposeExpr_eval] using equation

private theorem hintValue_val
    (interface : Interface) (env : Env) (offset : Nat) :
    (Hint.eval env (signHint interface offset)).val =
      ((parentValue interface offset env).val / 2 ^ signBitIndex) % 2 := by
  change (((((interface.parent offset).eval env).val >>> signBitIndex) &&& 1) %
      goldilocksModulus) = _
  rw [Nat.and_one_is_mod, Nat.shiftRight_eq_div_pow]
  apply Nat.mod_eq_of_lt
  exact lt_trans (Nat.mod_lt _ (by decide : 0 < 2)) (by
    norm_num [goldilocksModulus])

private theorem signHint_eq_branchBit
    (interface : Interface) (env : Env) (offset : Nat)
    (bounded : centeredMagnitude (parentValue interface offset env) <
      Radix.combinedBound) :
    Hint.eval env (signHint interface offset) =
      if Radix.isNonnegative (parentValue interface offset env) then 0 else 1 := by
  let parent := parentValue interface offset env
  by_cases nonnegative : Radix.isNonnegative parent
  · have magnitude : centeredMagnitude parent = parent.val := by
      rw [NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_eq_distance]
      unfold NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered.distance
      rw [if_pos (by simpa [Radix.isNonnegative] using nonnegative)]
    have parentLt : parent.val < 2 ^ signBitIndex := by
      rw [magnitude] at bounded
      norm_num [Radix.combinedBound, productionGlobalParams,
        GlobalParams.bigB, signBitIndex] at bounded ⊢
      omega
    have hintZero : Hint.eval env (signHint interface offset) = 0 := by
      apply Fin.ext
      rw [hintValue_val]
      change parent.val / 2 ^ signBitIndex % 2 = 0
      rw [Nat.div_eq_of_lt parentLt]
    simpa [parent, nonnegative] using hintZero
  · have magnitude :
        centeredMagnitude parent = goldilocksModulus - parent.val := by
      rw [NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_eq_distance]
      unfold NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered.distance
      rw [if_neg (by simpa [Radix.isNonnegative] using nonnegative)]
    have parentLower : 2 ^ signBitIndex ≤ parent.val := by
      rw [magnitude] at bounded
      have parentLtModulus := parent.isLt
      norm_num [Radix.combinedBound, productionGlobalParams,
        GlobalParams.bigB, signBitIndex, goldilocksModulus] at bounded ⊢
      omega
    have parentUpper : parent.val < 2 * 2 ^ signBitIndex := by
      have parentLtModulus := parent.isLt
      norm_num [signBitIndex, goldilocksModulus] at parentLtModulus ⊢
      omega
    have quotient : parent.val / 2 ^ signBitIndex = 1 := by
      norm_num [signBitIndex] at parentLower parentUpper ⊢
      omega
    have hintOne : Hint.eval env (signHint interface offset) = 1 := by
      apply Fin.ext
      rw [hintValue_val]
      change parent.val / 2 ^ signBitIndex % 2 = (1 : F).val
      rw [quotient]
      rfl
    simpa [parent, nonnegative] using hintOne

private theorem signHint_eq_honestSign
    (interface : Interface) (env : Env) (offset : Nat)
    (bounded : centeredMagnitude (parentValue interface offset env) <
      Radix.combinedBound) :
    (1 : F) - 2 * Hint.eval env (signHint interface offset) =
      Radix.UniformSignedDigits.honestSign
        (parentValue interface offset env) := by
  rw [signHint_eq_branchBit interface env offset bounded]
  by_cases nonnegative :
      Radix.isNonnegative (parentValue interface offset env)
  · simp [Radix.UniformSignedDigits.honestSign, nonnegative]
  · simp [Radix.UniformSignedDigits.honestSign, nonnegative]

def completeEnv (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeHints env offset [signHint interface offset]

private theorem hintReadsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    HintsReadBelow offset [signHint interface offset] := by
  intro hint member
  simp only [List.mem_singleton] at member
  subst hint
  exact assumptions.1

private theorem completeEnv_sign
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : centeredMagnitude (parentValue interface offset env) <
      Radix.combinedBound) :
    (signExpr offset).eval (completeEnv interface env offset) =
      Radix.UniformSignedDigits.honestSign
        (parentValue interface offset env) := by
  have value := executeHints_value_of_readBelow env offset
    [signHint interface offset]
    (hintReadsBelow interface offset assumptions) 0 (by simp)
  rw [signExpr_eval]
  have slot : completeEnv interface env offset offset =
      Hint.eval env (signHint interface offset) := by
    simpa [completeEnv] using value
  rw [slot]
  exact signHint_eq_honestSign interface env offset bounded

private theorem completeEnv_parent
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    parentValue interface offset (completeEnv interface env offset) =
      parentValue interface offset env := by
  apply Expr.eval_eq_of_agree_below _ offset _ _ assumptions.1
  intro index below
  change executeHints env offset [signHint interface offset] index = env index
  exact executeHints_agrees_below env offset
    [signHint interface offset] index below

private theorem completeEnv_digit
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (index : Radix.ChildIndex) :
    digitValues interface offset (completeEnv interface env offset) index =
      digitValues interface offset env index := by
  apply Expr.eval_eq_of_agree_below _ offset _ _ (assumptions.2 index)
  intro sourceIndex below
  change executeHints env offset [signHint interface offset] sourceIndex =
    env sourceIndex
  exact executeHints_agrees_below env offset
    [signHint interface offset] sourceIndex below

private theorem completeEnv_holdsFlat
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    holdsFlat (completeEnv interface env offset)
      (operations interface offset) := by
  rcases specification with ⟨_, accepted⟩
  have bounded := accepted.parentBounded
  have exactDigits := accepted.digits_eq_splitScalar
  have honest := Radix.UniformSignedDigits.honest_complete
    (parentValue interface offset env) bounded
  have signValue := completeEnv_sign interface env offset assumptions bounded
  have parentPreserved := completeEnv_parent interface env offset assumptions
  have digitPreserved :
      digitValues interface offset (completeEnv interface env offset) =
        digitValues interface offset env := by
    funext index
    exact completeEnv_digit interface env offset assumptions index
  unfold holdsFlat
  rw [flatConstraints_operations]
  intro expression member
  rw [constraints] at member
  rcases List.mem_cons.mp member with signMember | tailMember
  · subst expression
    rw [signConstraint_eval]
    have bitValue := executeHints_value_of_readBelow env offset
      [signHint interface offset]
      (hintReadsBelow interface offset assumptions) 0 (by simp)
    rw [show completeEnv interface env offset offset =
        Hint.eval env (signHint interface offset) by
          simpa [completeEnv] using bitValue]
    rw [signHint_eq_branchBit interface env offset bounded]
    by_cases nonnegative : Radix.isNonnegative (parentValue interface offset env)
    · simp [nonnegative]
    · simp [nonnegative]
  · rcases List.mem_append.mp tailMember with digitMember | recompositionMember
    · rcases List.mem_ofFn.mp digitMember with ⟨index, rfl⟩
      rw [digitConstraint_eval]
      rw [congrFun digitPreserved index, signValue,
        congrFun exactDigits index]
      rcases honest.constraint.2 index with inactive | active
      · rw [inactive]
        exact zero_mul _
      · rw [active, sub_self]
        exact mul_zero _
    · have exactRecomposition := List.mem_singleton.mp recompositionMember
      subst expression
      rw [recompositionConstraint_eval, digitPreserved, parentPreserved,
        accepted.recomposition, sub_self]

theorem completeness
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  refine ⟨completeEnv interface env offset, ?_,
    completeEnv_holdsFlat interface env offset assumptions specification⟩
  rw [localLength_eq]
  simpa [completeEnv, exactPrivateCount] using
    executeHints_agreesOutside env offset [signHint interface offset]

def circuit (interface : Interface) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => exactPrivateCount
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

theorem spec_parentBounded
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env) :
    centeredMagnitude (parentValue interface offset env) <
      Radix.combinedBound := by
  rcases specification with ⟨_, accepted⟩
  exact accepted.parentBounded

theorem spec_digits_eq_splitScalar
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env) :
    digitValues interface offset env =
      Radix.splitScalar (parentValue interface offset env) := by
  rcases specification with ⟨_, accepted⟩
  exact accepted.digits_eq_splitScalar

theorem spec_uses_bounded_branch
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env) :
    Radix.splitScalar (parentValue interface offset env) =
      Radix.boundedDigit (parentValue interface offset env) := by
  funext index
  simp only [Radix.splitScalar,
    if_pos (spec_parentBounded specification)]

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar
