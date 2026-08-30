import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Layout.R1CS

/-!
Owns generic variable-support preservation for the R1CS lowerer. A lowered
row can use only source variables selected by the caller or variables in the
exact fresh interval allocated by that lowering.

This module does not select a protocol support set or change lowering.
-/

namespace NightstreamFPrime.Layout.R1CS

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec

def LinearCombination.VarsSatisfy (allowed : Nat → Prop)
    (combination : LinearCombination) : Prop :=
  ∀ term ∈ combination.terms, allowed term.1

namespace LinearCombination.VarsSatisfy

theorem mono {allowed larger : Nat → Prop}
    (combination : LinearCombination)
    (scope : combination.VarsSatisfy allowed)
    (includes : ∀ index, allowed index → larger index) :
    combination.VarsSatisfy larger := by
  intro term member
  exact includes term.1 (scope term member)

theorem add (left right : LinearCombination) (allowed : Nat → Prop)
    (leftScope : left.VarsSatisfy allowed)
    (rightScope : right.VarsSatisfy allowed) :
    (LinearCombination.add left right).VarsSatisfy allowed := by
  intro term member
  rcases List.mem_append.mp member with member | member
  · exact leftScope term member
  · exact rightScope term member

theorem ofVar (index : Nat) (allowed : Nat → Prop)
    (scope : allowed index) :
    (LinearCombination.ofVar index).VarsSatisfy allowed := by
  intro term member
  simp only [LinearCombination.ofVar, List.mem_singleton] at member
  simpa [member] using scope

theorem scale (coefficient : F) (combination : LinearCombination)
    (allowed : Nat → Prop) (scope : combination.VarsSatisfy allowed) :
    (LinearCombination.scale coefficient combination).VarsSatisfy allowed := by
  intro term member
  simp only [LinearCombination.scale, List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  exact scope source sourceMember

end LinearCombination.VarsSatisfy

/-- Evaluation depends only on variables admitted by the selected support
predicate. -/
theorem LinearCombination.eval_eq_of_agree
    (combination : LinearCombination) (allowed : Nat → Prop)
    (left right : Env) (scope : combination.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    combination.eval left = combination.eval right := by
  have termsEqual :
      combination.terms.map (fun term => term.2 * left term.1) =
        combination.terms.map (fun term => term.2 * right term.1) := by
    apply List.map_congr_left
    intro term member
    rw [agrees term.1 (scope term member)]
  unfold LinearCombination.eval
  rw [termsEqual]

def Row.VarsSatisfy (allowed : Nat → Prop) (row : Row) : Prop :=
  row.a.VarsSatisfy allowed ∧ row.b.VarsSatisfy allowed ∧
    row.c.VarsSatisfy allowed

namespace Row.VarsSatisfy

theorem mono {allowed larger : Nat → Prop} (row : Row)
    (scope : row.VarsSatisfy allowed)
    (includes : ∀ index, allowed index → larger index) :
    row.VarsSatisfy larger :=
  ⟨scope.1.mono row.a includes, scope.2.1.mono row.b includes,
    scope.2.2.mono row.c includes⟩

end Row.VarsSatisfy

/-- A row remains satisfied when its environment changes only outside the
row's selected support. -/
theorem Row.holds_of_agree (row : Row) (allowed : Nat → Prop)
    (before after : Env) (scope : row.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → after index = before index)
    (holds : row.Holds before) : row.Holds after := by
  unfold Row.Holds at holds ⊢
  rw [row.a.eval_eq_of_agree allowed after before scope.1 agrees,
    row.b.eval_eq_of_agree allowed after before scope.2.1 agrees,
    row.c.eval_eq_of_agree allowed after before scope.2.2 agrees]
  exact holds

/-- A row family remains satisfied when two environments agree on every
selected source column used by that family. -/
theorem rowsHold_of_agree (rows : List Row) (allowed : Nat → Prop)
    (before after : Env)
    (scope : ∀ row ∈ rows, row.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → after index = before index)
    (holds : RowsHold before rows) : RowsHold after rows := by
  intro row member
  exact row.holds_of_agree allowed before after (scope row member) agrees
    (holds row member)

/-- Source support extended by one half-open fresh interval. -/
def SourceOrFresh (allowed : Nat → Prop) (start finish index : Nat) : Prop :=
  allowed index ∨ (start ≤ index ∧ index < finish)

private theorem sourceOrFresh_left {allowed : Nat → Prop}
    {start middle finish index : Nat}
    (scope : SourceOrFresh allowed start middle index)
    (order : middle ≤ finish) : SourceOrFresh allowed start finish index := by
  rcases scope with source | fresh
  · exact Or.inl source
  · exact Or.inr ⟨fresh.1, Nat.lt_of_lt_of_le fresh.2 order⟩

private theorem sourceOrFresh_right {allowed : Nat → Prop}
    {start middle finish index : Nat}
    (scope : SourceOrFresh allowed middle finish index)
    (order : start ≤ middle) : SourceOrFresh allowed start finish index := by
  rcases scope with source | fresh
  · exact Or.inl source
  · exact Or.inr ⟨order.trans fresh.1, fresh.2⟩

private theorem sourceOrFresh_widen {allowed : Nat → Prop}
    {sourceStart sourceFinish targetStart targetFinish index : Nat}
    (scope : SourceOrFresh allowed sourceStart sourceFinish index)
    (lower : targetStart ≤ sourceStart)
    (upper : sourceFinish ≤ targetFinish) :
    SourceOrFresh allowed targetStart targetFinish index := by
  rcases scope with source | fresh
  · exact Or.inl source
  · exact Or.inr
      ⟨lower.trans fresh.1, Nat.lt_of_lt_of_le fresh.2 upper⟩

private theorem lowerAffineCombination_varsSatisfy
    (allowed : Nat → Prop) :
    ∀ (expression : Expr), expression.VarsSatisfy allowed →
      ∀ (combination : LinearCombination),
        Option.map AffineResult.combination (lowerAffine expression) =
            some combination →
          combination.VarsSatisfy allowed
  | .var index, scope, combination, result => by
      simp only [lowerAffine, Option.map_some, Option.some.injEq] at result
      rw [← result]
      exact LinearCombination.VarsSatisfy.ofVar index allowed scope
  | .const value, _scope, combination, result => by
      simp only [lowerAffine, Option.map_some, Option.some.injEq] at result
      rw [← result]
      intro term member
      simp [LinearCombination.const] at member
  | .add left right, scope, combination, result => by
      cases leftResult : lowerAffine left with
      | none => simp [lowerAffine, leftResult] at result
      | some loweredLeft =>
          cases rightResult : lowerAffine right with
          | none => simp [lowerAffine, leftResult, rightResult] at result
          | some loweredRight =>
              simp only [lowerAffine, leftResult, rightResult,
                Option.map_some, Option.some.injEq] at result
              rw [← result]
              exact LinearCombination.VarsSatisfy.add _ _ allowed
                (lowerAffineCombination_varsSatisfy allowed left scope.1
                  loweredLeft.combination (by rw [leftResult]; rfl))
                (lowerAffineCombination_varsSatisfy allowed right scope.2
                  loweredRight.combination (by rw [rightResult]; rfl))
  | .mul (.const coefficient) right, scope, combination, result => by
      cases rightResult : lowerAffine right with
      | none => simp [lowerAffine, rightResult] at result
      | some loweredRight =>
          simp only [lowerAffine, rightResult, Option.map_some,
            Option.some.injEq] at result
          rw [← result]
          exact LinearCombination.VarsSatisfy.scale coefficient _ allowed
            (lowerAffineCombination_varsSatisfy allowed right scope.2
              loweredRight.combination (by rw [rightResult]; rfl))
  | .mul (.var index) (.const coefficient), scope, combination, result => by
      simp only [lowerAffine, Option.map_some, Option.some.injEq] at result
      rw [← result]
      exact LinearCombination.VarsSatisfy.scale coefficient _ allowed
        (LinearCombination.VarsSatisfy.ofVar index allowed scope.1)
  | .mul (.add left right) (.const coefficient), scope, combination,
      result => by
      cases leftResult : lowerAffine left with
      | none => simp [lowerAffine, leftResult] at result
      | some loweredLeft =>
          cases rightResult : lowerAffine right with
          | none => simp [lowerAffine, leftResult, rightResult] at result
          | some loweredRight =>
              simp only [lowerAffine, leftResult, rightResult,
                Option.map_some, Option.some.injEq] at result
              rw [← result]
              apply LinearCombination.VarsSatisfy.scale coefficient _ allowed
              exact LinearCombination.VarsSatisfy.add _ _ allowed
                (lowerAffineCombination_varsSatisfy allowed left scope.1.1
                  loweredLeft.combination (by rw [leftResult]; rfl))
                (lowerAffineCombination_varsSatisfy allowed right scope.1.2
                  loweredRight.combination (by rw [rightResult]; rfl))
  | .mul (.mul left right) (.const coefficient), scope, combination,
      result => by
      cases leftResult : lowerAffine (Expr.mul left right) with
      | none => simp [lowerAffine, leftResult] at result
      | some loweredLeft =>
          simp only [lowerAffine, leftResult, Option.map_some,
            Option.some.injEq] at result
          rw [← result]
          exact LinearCombination.VarsSatisfy.scale coefficient _ allowed
            (lowerAffineCombination_varsSatisfy allowed (Expr.mul left right)
              scope.1 loweredLeft.combination (by rw [leftResult]; rfl))
  | .mul (.var _) (.var _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.var _) (.add _ _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.var _) (.mul _ _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.add _ _) (.var _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.add _ _) (.add _ _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.add _ _) (.mul _ _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.mul _ _) (.var _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.mul _ _) (.add _ _), _scope, _combination, result => by
      simp [lowerAffine] at result
  | .mul (.mul _ _) (.mul _ _), _scope, _combination, result => by
      simp [lowerAffine] at result

theorem lowerAffine_varsSatisfy (expression : Expr) (allowed : Nat → Prop)
    (scope : expression.VarsSatisfy allowed)
    (lowered : AffineResult expression)
    (result : lowerAffine expression = some lowered) :
    lowered.combination.VarsSatisfy allowed := by
  apply lowerAffineCombination_varsSatisfy allowed expression scope
    lowered.combination
  rw [result]
  rfl

theorem lowerExpression_value_varsSatisfy (expression : Expr) (start : Nat)
    (allowed : Nat → Prop) (scope : expression.VarsSatisfy allowed) :
    (lowerExpression expression start).value.VarsSatisfy
      (SourceOrFresh allowed start (start + mulCount expression)) := by
  induction expression generalizing start with
  | var index =>
      exact LinearCombination.VarsSatisfy.ofVar index _ (Or.inl scope)
  | const value =>
      intro term member
      simp [lowerExpression, LinearCombination.const] at member
  | add left right leftIH rightIH =>
      let middle := start + mulCount left
      let finish := middle + mulCount right
      have leftScope := leftIH start scope.1
      have rightScope := rightIH middle scope.2
      have combined := LinearCombination.VarsSatisfy.add _ _
        (SourceOrFresh allowed start finish)
        (leftScope.mono _ fun index support =>
          sourceOrFresh_left (allowed := allowed) (start := start)
            (middle := start + mulCount left) (finish := finish) support (by
              dsimp [finish, middle]
              omega))
        (rightScope.mono _ fun index support =>
          sourceOrFresh_right (allowed := allowed) (start := start)
            (middle := middle) (finish := finish) support (by
              dsimp [middle]
              omega))
      simpa [lowerExpression, mulCount, middle, finish, Nat.add_assoc] using
        combined
  | mul left right leftIH rightIH =>
      let output := start + mulCount left + mulCount right
      have result := LinearCombination.VarsSatisfy.ofVar output
        (SourceOrFresh allowed start
          (start + mulCount (Expr.mul left right)))
        (Or.inr ⟨by unfold output; omega, by
          unfold output
          simp [mulCount]
          omega⟩)
      simpa [lowerExpression, lowerExpression_next, output] using result

theorem lowerExpression_rows_varsSatisfy (expression : Expr) (start : Nat)
    (allowed : Nat → Prop) (scope : expression.VarsSatisfy allowed) :
    ∀ row ∈ (lowerExpression expression start).rows,
      row.VarsSatisfy
        (SourceOrFresh allowed start (start + mulCount expression)) := by
  induction expression generalizing start with
  | var index =>
      intro row member
      simp [lowerExpression] at member
  | const value =>
      intro row member
      simp [lowerExpression] at member
  | add left right leftIH rightIH =>
      let middle := start + mulCount left
      let finish := middle + mulCount right
      intro row member
      have member' : row ∈
          (lowerExpression left start).rows ++
            (lowerExpression right middle).rows := by
        simpa [lowerExpression, middle] using member
      rcases List.mem_append.mp member' with leftMember | rightMember
      · have leftScope := leftIH start scope.1 row leftMember
        have widened : row.VarsSatisfy
            (SourceOrFresh allowed start finish) :=
          leftScope.mono row fun index support =>
            sourceOrFresh_left (allowed := allowed) (start := start)
              (middle := start + mulCount left) (finish := finish) support (by
                dsimp [finish, middle]
                omega)
        simpa [mulCount, finish, middle, Nat.add_assoc] using widened
      · have rightScope := rightIH middle scope.2 row rightMember
        have widened : row.VarsSatisfy
            (SourceOrFresh allowed start finish) :=
          rightScope.mono row fun index support =>
            sourceOrFresh_right (allowed := allowed) (start := start)
              (middle := middle) (finish := finish) support (by
                dsimp [middle]
                omega)
        simpa [mulCount, finish, middle, Nat.add_assoc] using widened
  | mul left right leftIH rightIH =>
      let middle := start + mulCount left
      let output := middle + mulCount right
      let finish := output + 1
      intro row member
      have member' : row ∈
          (lowerExpression left start).rows ++
            (lowerExpression right middle).rows ++
              [⟨(lowerExpression left start).value,
                (lowerExpression right middle).value,
                LinearCombination.ofVar output⟩] := by
        simpa [lowerExpression, middle, output] using member
      rcases List.mem_append.mp member' with priorMember | productMember
      · rcases List.mem_append.mp priorMember with leftMember | rightMember
        · have leftScope := leftIH start scope.1 row leftMember
          have widened : row.VarsSatisfy
              (SourceOrFresh allowed start finish) :=
            leftScope.mono row fun index support =>
              sourceOrFresh_left (allowed := allowed) (start := start)
                (middle := start + mulCount left) (finish := finish) support (by
                  dsimp [finish, output, middle]
                  omega)
          simpa [mulCount, finish, output, middle, Nat.add_assoc] using widened
        · have rightScope := rightIH middle scope.2 row rightMember
          have widened : row.VarsSatisfy
              (SourceOrFresh allowed start finish) :=
            rightScope.mono row fun index support =>
              sourceOrFresh_widen (allowed := allowed)
                (sourceStart := middle)
                (sourceFinish := middle + mulCount right)
                (targetStart := start) (targetFinish := finish) support (by
                  dsimp [middle]
                  omega) (by
                    dsimp [finish, output]
                    omega)
          simpa [mulCount, finish, output, middle, Nat.add_assoc] using widened
      · simp only [List.mem_singleton] at productMember
        subst row
        refine ⟨?_, ?_, ?_⟩
        · have leftValue := lowerExpression_value_varsSatisfy left start
            allowed scope.1
          have widened :
              (lowerExpression left start).value.VarsSatisfy
                (SourceOrFresh allowed start finish) :=
            leftValue.mono _ fun index support =>
              sourceOrFresh_left (allowed := allowed) (start := start)
                (middle := start + mulCount left) (finish := finish) support (by
                dsimp [middle, output, finish]
                omega)
          simpa [mulCount, finish, output, middle, Nat.add_assoc] using widened
        · have rightValue := lowerExpression_value_varsSatisfy right middle
            allowed scope.2
          have widened :
              (lowerExpression right middle).value.VarsSatisfy
                (SourceOrFresh allowed start finish) :=
            rightValue.mono _ fun index support =>
              sourceOrFresh_widen (allowed := allowed)
                (sourceStart := middle)
                (sourceFinish := middle + mulCount right)
                (targetStart := start) (targetFinish := finish) support (by
                  dsimp [middle]
                  omega) (by
                    dsimp [finish, output]
                    omega)
          simpa [mulCount, finish, output, middle, Nat.add_assoc] using widened
        · exact LinearCombination.VarsSatisfy.ofVar output _
            (Or.inr ⟨by unfold output middle; omega, by
              unfold output middle
              simp [mulCount]
              omega⟩)

theorem lowerGenericConstraint_rows_varsSatisfy (expression : Expr)
    (start : Nat) (allowed : Nat → Prop)
    (scope : expression.VarsSatisfy allowed) :
    ∀ row ∈ (lowerGenericConstraint expression start).rows,
      row.VarsSatisfy
        (SourceOrFresh allowed start (start + mulCount expression)) := by
  let lowered := lowerExpression expression start
  let assertion : Row :=
    ⟨lowered.value, LinearCombination.one, LinearCombination.zero⟩
  intro row member
  have member' : row ∈ lowered.rows ++ [assertion] := by
    simpa [lowerGenericConstraint, lowered, assertion] using member
  rcases List.mem_append.mp member' with expressionMember | assertionMember
  · exact lowerExpression_rows_varsSatisfy expression start allowed scope
      row expressionMember
  · simp only [List.mem_singleton] at assertionMember
    subst row
    refine ⟨?_, ?_, ?_⟩
    · simpa [lowered] using
        lowerExpression_value_varsSatisfy expression start allowed scope
    · intro term termMember
      simp [assertion, LinearCombination.one] at termMember
    · intro term termMember
      simp [assertion, LinearCombination.zero] at termMember

private theorem affineConstraint_row_varsSatisfy (expression : Expr)
    (allowed : Nat → Prop) (scope : expression.VarsSatisfy allowed)
    (result : DirectConstraintResult expression)
    (found : affineConstraint expression = some result) :
    result.row.VarsSatisfy allowed := by
  unfold affineConstraint at found
  cases loweredEq : lowerAffine expression with
  | none => simp [loweredEq] at found
  | some lowered =>
      simp only [loweredEq, Option.some.injEq] at found
      subst result
      refine ⟨lowerAffine_varsSatisfy expression allowed scope lowered
          loweredEq, ?_, ?_⟩
      · intro term member
        change term ∈ ([] : List (Nat × F)) at member
        simp at member
      · intro term member
        change term ∈ ([] : List (Nat × F)) at member
        simp at member

private theorem directRecipeRow_row_varsSatisfy (output : Nat)
    (allowed : Nat → Prop) (recipe : Expr) (outputAllowed : allowed output)
    (scope : recipe.VarsSatisfy allowed)
    (result : RecipeRowResult output recipe)
    (found : directRecipeRow output recipe = some result) :
    result.row.VarsSatisfy allowed := by
  cases affineEq : lowerAffine recipe with
  | some lowered =>
      simp only [directRecipeRow, affineEq, Option.some.injEq] at found
      subst result
      refine ⟨lowerAffine_varsSatisfy recipe allowed scope lowered affineEq,
        ?_, LinearCombination.VarsSatisfy.ofVar output allowed outputAllowed⟩
      intro term member
      change term ∈ ([] : List (Nat × F)) at member
      simp at member
  | none =>
      cases recipe with
      | var index => simp [directRecipeRow, affineEq] at found
      | const value => simp [directRecipeRow, affineEq] at found
      | add left right => simp [directRecipeRow, affineEq] at found
      | mul left right =>
          cases leftEq : lowerAffine left with
          | none => simp [directRecipeRow, affineEq, leftEq] at found
          | some loweredLeft =>
              cases rightEq : lowerAffine right with
              | none =>
                  simp [directRecipeRow, affineEq, leftEq, rightEq] at found
              | some loweredRight =>
                  simp only [directRecipeRow, affineEq, leftEq, rightEq,
                    Option.some.injEq] at found
                  subst result
                  exact ⟨lowerAffine_varsSatisfy left allowed scope.1
                      loweredLeft leftEq,
                    lowerAffine_varsSatisfy right allowed scope.2
                      loweredRight rightEq,
                    LinearCombination.VarsSatisfy.ofVar output allowed
                      outputAllowed⟩

private theorem directConstraint_row_varsSatisfy (expression : Expr)
    (allowed : Nat → Prop) (scope : expression.VarsSatisfy allowed)
    (result : DirectConstraintResult expression)
    (found : directConstraint expression = some result) :
    result.row.VarsSatisfy allowed := by
  cases expression with
  | var index =>
      exact affineConstraint_row_varsSatisfy (.var index) allowed scope result
        found
  | const value =>
      exact affineConstraint_row_varsSatisfy (.const value) allowed scope result
        found
  | mul left right =>
      exact affineConstraint_row_varsSatisfy (.mul left right) allowed scope
        result found
  | add left right =>
      cases left with
      | var output =>
          cases right with
          | mul coefficientExpr recipe =>
              cases coefficientExpr with
              | const coefficient =>
                  by_cases coefficientEquals : coefficient = -1
                  · rw [directConstraint, dif_pos coefficientEquals] at found
                    cases recipeEq : directRecipeRow output recipe with
                    | none =>
                        rw [recipeEq] at found
                        exact affineConstraint_row_varsSatisfy
                          (.add (.var output)
                            (.mul (.const coefficient) recipe)) allowed scope
                          result found
                    | some recipeResult =>
                        rw [recipeEq] at found
                        simp only [Option.some.injEq] at found
                        subst result
                        exact directRecipeRow_row_varsSatisfy output allowed
                          recipe scope.1 scope.2.2 recipeResult recipeEq
                  · rw [directConstraint, dif_neg coefficientEquals] at found
                    exact affineConstraint_row_varsSatisfy
                      (.add (.var output) (.mul (.const coefficient) recipe))
                      allowed scope result found
              | var index =>
                  exact affineConstraint_row_varsSatisfy
                    (.add (.var output) (.mul (.var index) recipe)) allowed
                    scope result found
              | add first second =>
                  exact affineConstraint_row_varsSatisfy
                    (.add (.var output) (.mul (.add first second) recipe))
                    allowed scope result found
              | mul first second =>
                  exact affineConstraint_row_varsSatisfy
                    (.add (.var output) (.mul (.mul first second) recipe))
                    allowed scope result found
          | var index =>
              exact affineConstraint_row_varsSatisfy
                (.add (.var output) (.var index)) allowed scope result found
          | const value =>
              exact affineConstraint_row_varsSatisfy
                (.add (.var output) (.const value)) allowed scope result found
          | add first second =>
              exact affineConstraint_row_varsSatisfy
                (.add (.var output) (.add first second)) allowed scope result
                found
      | const value =>
          exact affineConstraint_row_varsSatisfy
            (.add (.const value) right) allowed scope result found
      | add first second =>
          exact affineConstraint_row_varsSatisfy
            (.add (.add first second) right) allowed scope result found
      | mul first second =>
          exact affineConstraint_row_varsSatisfy
            (.add (.mul first second) right) allowed scope result found

theorem lowerConstraint_rows_varsSatisfy (expression : Expr) (start : Nat)
    (allowed : Nat → Prop) (scope : expression.VarsSatisfy allowed) :
    ∀ row ∈ (lowerConstraint expression start).rows,
      row.VarsSatisfy (SourceOrFresh allowed start
        (start + constraintFreshCount expression)) := by
  cases resultEq : directConstraint expression with
  | none =>
      simpa [lowerConstraint, constraintFreshCount, resultEq] using
        lowerGenericConstraint_rows_varsSatisfy expression start allowed scope
  | some result =>
      intro row member
      simp only [lowerConstraint, resultEq, List.mem_singleton] at member
      subst row
      have directScope := directConstraint_row_varsSatisfy expression allowed
        scope result resultEq
      exact directScope.mono _ fun index support => Or.inl support

theorem lowerConstraints_rows_varsSatisfy (constraints : List Expr)
    (start : Nat) (allowed : Nat → Prop)
    (scope : ∀ expression ∈ constraints,
      expression.VarsSatisfy allowed) :
    ∀ row ∈ (lowerConstraints constraints start).rows,
      row.VarsSatisfy
        (SourceOrFresh allowed start (start + totalFreshCount constraints)) := by
  induction constraints generalizing start with
  | nil =>
      intro row member
      simp [lowerConstraints] at member
  | cons expression rest inductionHypothesis =>
      let middle := start + constraintFreshCount expression
      let finish := middle + totalFreshCount rest
      have firstScope := lowerConstraint_rows_varsSatisfy expression start
        allowed (scope expression (by simp))
      have tailScope := inductionHypothesis middle
        (fun current member => scope current (by simp [member]))
      intro row member
      simp only [lowerConstraints, List.mem_append] at member
      have nextEq : (lowerConstraint expression start).next = middle := by
        simp [middle]
      rw [nextEq] at member
      rcases member with firstMember | tailMember
      · have first := firstScope row firstMember
        have widened : row.VarsSatisfy
            (SourceOrFresh allowed start finish) :=
          first.mono row fun index support =>
            sourceOrFresh_left (allowed := allowed) (start := start)
              (middle := middle) (finish := finish) support (by
                dsimp [middle, finish]
                omega)
        simpa [totalFreshCount, finish, middle, Nat.add_assoc] using widened
      · have tail := tailScope row tailMember
        have widened : row.VarsSatisfy
            (SourceOrFresh allowed start finish) :=
          tail.mono row fun index support =>
            sourceOrFresh_right (allowed := allowed) (start := start)
              (middle := middle) (finish := finish) support (by
                dsimp [middle]
                omega)
        simpa [totalFreshCount, finish, middle, Nat.add_assoc] using widened

namespace LoweringPlan

/-- The opaque lowering-plan boundary preserves generic source support
without exposing the concrete constraint list to parent proofs. -/
theorem rows_varsSatisfy (plan : LoweringPlan) (allowed : Nat → Prop)
    (scope : ∀ expression ∈ plan.constraints,
      expression.VarsSatisfy allowed) :
    ∀ row ∈ plan.rows,
      row.VarsSatisfy (SourceOrFresh allowed plan.firstFresh plan.next) := by
  rw [plan.next_eq]
  change ∀ row ∈
      (lowerConstraints plan.constraints plan.firstFresh).rows,
    row.VarsSatisfy (SourceOrFresh allowed plan.firstFresh
      (plan.firstFresh + totalFreshCount plan.constraints))
  exact lowerConstraints_rows_varsSatisfy plan.constraints plan.firstFresh
    allowed scope

end LoweringPlan

end NightstreamFPrime.Layout.R1CS
