import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Circuit.StraightLine

/-!
Owns constructive physical witnesses for the generic expression lowerer.
The executor writes only the fresh multiplication interval and proves every
generated rank-one row. It does not own logical witness recipes or any
protocol relation.
-/

namespace NightstreamFPrime.Layout.R1CS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def LinearCombination.VarsBelow
    (combination : LinearCombination) (bound : Nat) : Prop :=
  ∀ term ∈ combination.terms, term.1 < bound

namespace LinearCombination.VarsBelow

theorem mono (combination : LinearCombination) {lower upper : Nat}
    (scope : combination.VarsBelow lower) (le : lower ≤ upper) :
    combination.VarsBelow upper := by
  intro term member
  exact Nat.lt_of_lt_of_le (scope term member) le

theorem add (left right : LinearCombination) (bound : Nat)
    (leftScope : left.VarsBelow bound)
    (rightScope : right.VarsBelow bound) :
    (LinearCombination.add left right).VarsBelow bound := by
  intro term member
  rcases List.mem_append.mp member with member | member
  · exact leftScope term member
  · exact rightScope term member

theorem ofVar (index bound : Nat) (below : index < bound) :
    (LinearCombination.ofVar index).VarsBelow bound := by
  intro term member
  simp only [LinearCombination.ofVar, List.mem_singleton] at member
  simpa [member] using below

end LinearCombination.VarsBelow

theorem LinearCombination.eval_eq_of_agree_below
    (combination : LinearCombination) (bound : Nat) (left right : Env)
    (scope : combination.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    combination.eval left = combination.eval right := by
  have termsEqual :
      combination.terms.map (fun term => term.2 * left term.1) =
        combination.terms.map (fun term => term.2 * right term.1) := by
    apply List.map_congr_left
    intro term member
    rw [agrees term.1 (scope term member)]
  unfold LinearCombination.eval
  rw [termsEqual]

def Row.VarsBelow (row : Row) (bound : Nat) : Prop :=
  row.a.VarsBelow bound ∧ row.b.VarsBelow bound ∧ row.c.VarsBelow bound

namespace Row.VarsBelow

theorem mono (row : Row) {lower upper : Nat}
    (scope : row.VarsBelow lower) (le : lower ≤ upper) :
    row.VarsBelow upper :=
  ⟨scope.1.mono row.a le,
    scope.2.1.mono row.b le,
    scope.2.2.mono row.c le⟩

end Row.VarsBelow

theorem Row.holds_of_agree_below (row : Row) (bound : Nat)
    (before after : Env) (scope : row.VarsBelow bound)
    (agrees : ∀ index, index < bound → after index = before index)
    (holds : row.Holds before) : row.Holds after := by
  unfold Row.Holds at holds ⊢
  rw [row.a.eval_eq_of_agree_below bound after before scope.1 agrees,
    row.b.eval_eq_of_agree_below bound after before scope.2.1 agrees,
    row.c.eval_eq_of_agree_below bound after before scope.2.2 agrees]
  exact holds

theorem rowsHold_of_agree_below (rows : List Row) (bound : Nat)
    (before after : Env)
    (scope : ∀ row ∈ rows, row.VarsBelow bound)
    (agrees : ∀ index, index < bound → after index = before index)
    (holds : RowsHold before rows) : RowsHold after rows := by
  intro row member
  exact row.holds_of_agree_below bound before after (scope row member) agrees
    (holds row member)

theorem lowerExpression_value_varsBelow (expression : Expr) (start : Nat)
    (scope : expression.VarsBelow start) :
    (lowerExpression expression start).value.VarsBelow
      (start + mulCount expression) := by
  induction expression generalizing start with
  | var index =>
      exact LinearCombination.VarsBelow.ofVar index start scope
  | const value =>
      intro term member
      simp [lowerExpression, LinearCombination.const] at member
  | add left right leftIH rightIH =>
      let rightStart := start + mulCount left
      have leftScope := leftIH start scope.1
      have rightInputScope : right.VarsBelow rightStart :=
        Expr.VarsBelow.mono right scope.2 (by unfold rightStart; omega)
      have rightScope := rightIH rightStart rightInputScope
      have result : (LinearCombination.add
          (lowerExpression left start).value
          (lowerExpression right rightStart).value).VarsBelow
            (start + (mulCount left + mulCount right)) :=
        LinearCombination.VarsBelow.add _ _ _
          (leftScope.mono _ (by omega)) (by
            simpa [rightStart, Nat.add_assoc] using rightScope)
      simpa [lowerExpression, rightStart, mulCount, Nat.add_assoc] using result
  | mul left right leftIH rightIH =>
      have result : (LinearCombination.ofVar
          (start + mulCount left + mulCount right)).VarsBelow
            (start + (mulCount left + mulCount right + 1)) :=
        LinearCombination.VarsBelow.ofVar _ _ (by omega)
      simpa [lowerExpression, mulCount, Nat.add_assoc] using result

theorem lowerExpression_rows_varsBelow (expression : Expr) (start : Nat)
    (scope : expression.VarsBelow start) :
    ∀ row ∈ (lowerExpression expression start).rows,
      row.VarsBelow (start + mulCount expression) := by
  induction expression generalizing start with
  | var index =>
      intro row member
      simp [lowerExpression] at member
  | const value =>
      intro row member
      simp [lowerExpression] at member
  | add left right leftIH rightIH =>
      let rightStart := start + mulCount left
      have rightInputScope : right.VarsBelow rightStart :=
        Expr.VarsBelow.mono right scope.2 (by unfold rightStart; omega)
      intro row member
      have member' : row ∈
          (lowerExpression left start).rows ++
            (lowerExpression right rightStart).rows := by
        simpa [lowerExpression, rightStart] using member
      rcases List.mem_append.mp member' with leftMember | rightMember
      · have below := (leftIH start scope.1 row leftMember).mono row
            (show start + mulCount left ≤
              start + (mulCount left + mulCount right) by omega)
        simpa [mulCount] using below
      · have below := rightIH rightStart rightInputScope row rightMember
        simpa [rightStart, mulCount, Nat.add_assoc] using below
  | mul left right leftIH rightIH =>
      let rightStart := start + mulCount left
      let output := rightStart + mulCount right
      have rightInputScope : right.VarsBelow rightStart :=
        Expr.VarsBelow.mono right scope.2 (by unfold rightStart; omega)
      intro row member
      have member' : row ∈
          (lowerExpression left start).rows ++
            (lowerExpression right rightStart).rows ++
              [⟨(lowerExpression left start).value,
                (lowerExpression right rightStart).value,
                LinearCombination.ofVar output⟩] := by
        simpa [lowerExpression, rightStart, output] using member
      rcases List.mem_append.mp member' with priorMember | productMember
      · rcases List.mem_append.mp priorMember with leftMember | rightMember
        · have below := (leftIH start scope.1 row leftMember).mono row
              (show start + mulCount left ≤
                start + (mulCount left + mulCount right + 1) by omega)
          simpa [mulCount] using below
        · have below := rightIH rightStart rightInputScope row rightMember
          have widened := below.mono row
            (show rightStart + mulCount right ≤
              start + (mulCount left + mulCount right + 1) by
                unfold rightStart
                omega)
          simpa [mulCount] using widened
      · simp only [List.mem_singleton] at productMember
        subst row
        refine ⟨?_, ?_, ?_⟩
        · have below :=
            (lowerExpression_value_varsBelow left start scope.1).mono _
              (show start + mulCount left ≤
                start + (mulCount left + mulCount right + 1) by omega)
          simpa [mulCount] using below
        · have below := lowerExpression_value_varsBelow right rightStart
            rightInputScope
          have widened := below.mono _
            (show rightStart + mulCount right ≤
              start + (mulCount left + mulCount right + 1) by
                unfold rightStart
                omega)
          simpa [mulCount] using widened
        · have below := LinearCombination.VarsBelow.ofVar output
            (start + (mulCount left + mulCount right + 1)) (by
              unfold output rightStart
              omega)
          simpa [mulCount] using below

private theorem envSet_agreesOutside (env : Env) (index : Nat) (value : F) :
    AgreesOutside env (Env.set env index value) index 1 := by
  intro current outside
  apply Env.set_of_ne
  omega

/-- Execute the multiplication intermediates in the same order as
`lowerExpression`. -/
def executeExpression : Env → Expr → Nat → Env
  | env, .var _, _ => env
  | env, .const _, _ => env
  | env, .add left right, start =>
      let afterLeft := executeExpression env left start
      executeExpression afterLeft right (start + mulCount left)
  | env, .mul left right, start =>
      let rightStart := start + mulCount left
      let afterLeft := executeExpression env left start
      let afterRight := executeExpression afterLeft right rightStart
      let output := rightStart + mulCount right
      let value :=
        (lowerExpression left start).value.eval afterRight *
          (lowerExpression right rightStart).value.eval afterRight
      Env.set afterRight output value

theorem executeExpression_agreesOutside (env : Env) (expression : Expr)
    (start : Nat) :
    AgreesOutside env (executeExpression env expression start) start
      (mulCount expression) := by
  induction expression generalizing env start with
  | var index =>
      intro current _
      rfl
  | const value =>
      intro current _
      rfl
  | add left right leftIH rightIH =>
      let afterLeft := executeExpression env left start
      have first := leftIH env start
      have second := rightIH afterLeft (start + mulCount left)
      simpa [executeExpression, afterLeft, mulCount, Nat.add_assoc] using
        first.append second
  | mul left right leftIH rightIH =>
      let rightStart := start + mulCount left
      let afterLeft := executeExpression env left start
      let afterRight := executeExpression afterLeft right rightStart
      let output := rightStart + mulCount right
      let value :=
        (lowerExpression left start).value.eval afterRight *
          (lowerExpression right rightStart).value.eval afterRight
      have first := leftIH env start
      have second := rightIH afterLeft rightStart
      have prefixAgrees : AgreesOutside env afterRight start
          (mulCount left + mulCount right) := by
        simpa [afterLeft, afterRight, rightStart] using first.append second
      have final := envSet_agreesOutside afterRight output value
      have finalNormalized : AgreesOutside afterRight
          (Env.set afterRight output value)
          (start + (mulCount left + mulCount right)) 1 := by
        simpa [output, rightStart, Nat.add_assoc] using final
      simpa [executeExpression, afterLeft, afterRight, rightStart, output,
        value, mulCount, Nat.add_assoc] using
          prefixAgrees.append finalNormalized

theorem executeExpression_holds_rows (env : Env) (expression : Expr)
    (start : Nat) (scope : expression.VarsBelow start) :
    RowsHold (executeExpression env expression start)
      (lowerExpression expression start).rows := by
  induction expression generalizing env start with
  | var index =>
      intro row member
      simp [lowerExpression] at member
  | const constant =>
      intro row member
      simp [lowerExpression] at member
  | add left right leftIH rightIH =>
      let rightStart := start + mulCount left
      let afterLeft := executeExpression env left start
      let afterRight := executeExpression afterLeft right rightStart
      have rightInputScope : right.VarsBelow rightStart :=
        Expr.VarsBelow.mono right scope.2 (by unfold rightStart; omega)
      have leftRows : RowsHold afterLeft (lowerExpression left start).rows :=
        leftIH env start scope.1
      have rightRows : RowsHold afterRight
          (lowerExpression right rightStart).rows :=
        rightIH afterLeft rightStart rightInputScope
      have rightAgrees := executeExpression_agreesOutside afterLeft right
        rightStart
      have leftRowsAfter : RowsHold afterRight
          (lowerExpression left start).rows :=
        rowsHold_of_agree_below _ rightStart afterLeft afterRight
          (lowerExpression_rows_varsBelow left start scope.1)
          (fun index below => rightAgrees index (Or.inl below)) leftRows
      simpa [executeExpression, afterLeft, afterRight, rightStart,
        lowerExpression] using
          (rowsHold_append afterRight _ _).mpr ⟨leftRowsAfter, rightRows⟩
  | mul left right leftIH rightIH =>
      let rightStart := start + mulCount left
      let afterLeft := executeExpression env left start
      let afterRight := executeExpression afterLeft right rightStart
      let output := rightStart + mulCount right
      let value :=
        (lowerExpression left start).value.eval afterRight *
          (lowerExpression right rightStart).value.eval afterRight
      let completed := Env.set afterRight output value
      let productRow : Row :=
        ⟨(lowerExpression left start).value,
          (lowerExpression right rightStart).value,
          LinearCombination.ofVar output⟩
      have rightInputScope : right.VarsBelow rightStart :=
        Expr.VarsBelow.mono right scope.2 (by unfold rightStart; omega)
      have leftRows : RowsHold afterLeft (lowerExpression left start).rows :=
        leftIH env start scope.1
      have rightRows : RowsHold afterRight
          (lowerExpression right rightStart).rows :=
        rightIH afterLeft rightStart rightInputScope
      have rightAgrees := executeExpression_agreesOutside afterLeft right
        rightStart
      have leftRowsAfter : RowsHold afterRight
          (lowerExpression left start).rows :=
        rowsHold_of_agree_below _ rightStart afterLeft afterRight
          (lowerExpression_rows_varsBelow left start scope.1)
          (fun index below => rightAgrees index (Or.inl below)) leftRows
      have finalAgrees := envSet_agreesOutside afterRight output value
      have leftRowsCompleted : RowsHold completed
          (lowerExpression left start).rows :=
        rowsHold_of_agree_below _ output afterRight completed
          (fun row member =>
            (lowerExpression_rows_varsBelow left start scope.1 row member).mono
              row (by unfold output rightStart; omega))
          (fun index below => finalAgrees index (Or.inl below)) leftRowsAfter
      have rightRowsCompleted : RowsHold completed
          (lowerExpression right rightStart).rows :=
        rowsHold_of_agree_below _ output afterRight completed
          (fun row member => by
            simpa [output] using
              lowerExpression_rows_varsBelow right rightStart rightInputScope
                row member)
          (fun index below => finalAgrees index (Or.inl below)) rightRows
      have leftValueScope :
          (lowerExpression left start).value.VarsBelow output :=
        (lowerExpression_value_varsBelow left start scope.1).mono _ (by
          unfold output rightStart
          omega)
      have rightValueScope :
          (lowerExpression right rightStart).value.VarsBelow output := by
        simpa [output] using
          lowerExpression_value_varsBelow right rightStart rightInputScope
      have agreesBelow : ∀ index, index < output →
          completed index = afterRight index :=
        fun index below => finalAgrees index (Or.inl below)
      have leftEval :=
        (lowerExpression left start).value.eval_eq_of_agree_below output
          completed afterRight leftValueScope agreesBelow
      have rightEval :=
        (lowerExpression right rightStart).value.eval_eq_of_agree_below output
          completed afterRight rightValueScope agreesBelow
      have productHolds : productRow.Holds completed := by
        unfold productRow Row.Holds
        rw [leftEval, rightEval]
        simp [completed, value]
      have priorRows : RowsHold completed
          ((lowerExpression left start).rows ++
            (lowerExpression right rightStart).rows) :=
        (rowsHold_append completed _ _).mpr
          ⟨leftRowsCompleted, rightRowsCompleted⟩
      have allRows : RowsHold completed
          ((lowerExpression left start).rows ++
            (lowerExpression right rightStart).rows ++ [productRow]) :=
        (rowsHold_append completed _ _).mpr
          ⟨priorRows, fun row member => by
            simp only [List.mem_singleton] at member
            simpa [member] using productHolds⟩
      simpa [executeExpression, afterLeft, afterRight, rightStart, output,
        value, completed, productRow, lowerExpression] using allRows

theorem lowerGenericConstraint_rows_varsBelow (expression : Expr)
    (start : Nat) (scope : expression.VarsBelow start) :
    ∀ row ∈ (lowerGenericConstraint expression start).rows,
      row.VarsBelow (start + mulCount expression) := by
  let lowered := lowerExpression expression start
  let assertion : Row :=
    ⟨lowered.value, LinearCombination.one, LinearCombination.zero⟩
  intro row member
  have member' : row ∈ lowered.rows ++ [assertion] := by
    simpa [lowerGenericConstraint, lowered, assertion] using member
  rcases List.mem_append.mp member' with expressionMember | assertionMember
  · exact lowerExpression_rows_varsBelow expression start scope row
      expressionMember
  · simp only [List.mem_singleton] at assertionMember
    subst row
    refine ⟨?_, ?_, ?_⟩
    · simpa [lowered] using
        lowerExpression_value_varsBelow expression start scope
    · intro term termMember
      simp [assertion, LinearCombination.VarsBelow, LinearCombination.one]
        at termMember
    · intro term termMember
      simp [assertion, LinearCombination.VarsBelow, LinearCombination.zero]
        at termMember

/-- Execute one optimized logical constraint. Direct rows need no fresh
value; generic rows execute every multiplication in the expression. -/
def executeConstraint (env : Env) (expression : Expr) (start : Nat) : Env :=
  match directConstraint expression with
  | some _ => env
  | none => executeExpression env expression start

theorem executeConstraint_agreesOutside (env : Env) (expression : Expr)
    (start : Nat) :
    AgreesOutside env (executeConstraint env expression start) start
      (constraintFreshCount expression) := by
  cases result : directConstraint expression with
  | none =>
      simpa [executeConstraint, constraintFreshCount, result] using
        executeExpression_agreesOutside env expression start
  | some direct =>
      intro index _
      simp [executeConstraint, result]

theorem executeConstraint_holds_rows (env : Env) (expression : Expr)
    (start : Nat) (scope : expression.VarsBelow start)
    (logical : expression.eval env = 0) :
    RowsHold (executeConstraint env expression start)
      (lowerConstraint expression start).rows := by
  cases result : directConstraint expression with
  | some direct =>
      have fresh : constraintFreshCount expression = 0 := by
        simp [constraintFreshCount, result]
      simpa [executeConstraint, result] using
        lowerConstraint_complete_of_freshCount_zero env expression start fresh
          logical
  | none =>
      let completed := executeExpression env expression start
      let lowered := lowerExpression expression start
      let assertion : Row :=
        ⟨lowered.value, LinearCombination.one, LinearCombination.zero⟩
      have expressionRows : RowsHold completed lowered.rows := by
        simpa [completed, lowered] using
          executeExpression_holds_rows env expression start scope
      have agrees := executeExpression_agreesOutside env expression start
      have expressionEval : expression.eval completed = 0 := by
        rw [expression.eval_eq_of_agree_below start completed env scope
          (fun index below => agrees index (Or.inl below))]
        exact logical
      have loweredEval : lowered.value.eval completed = expression.eval completed :=
        lowerExpression_sound completed expression start (by
          simpa [lowered] using expressionRows)
      have assertionHolds : assertion.Holds completed := by
        simp [assertion, Row.Holds, loweredEval, expressionEval]
      have allRows : RowsHold completed (lowered.rows ++ [assertion]) :=
        (rowsHold_append completed _ _).mpr
          ⟨expressionRows, fun row member => by
            simp only [List.mem_singleton] at member
            simpa [member] using assertionHolds⟩
      simpa [executeConstraint, lowerConstraint, lowerGenericConstraint,
        result, completed, lowered, assertion] using allRows

/-- Execute physical intermediates for an ordered logical constraint list. -/
def executeConstraints : Env → List Expr → Nat → Env
  | env, [], _ => env
  | env, expression :: rest, start =>
      let afterFirst := executeConstraint env expression start
      executeConstraints afterFirst rest
        (start + constraintFreshCount expression)

theorem executeConstraints_agreesOutside (env : Env)
    (constraints : List Expr) (start : Nat) :
    AgreesOutside env (executeConstraints env constraints start) start
      (totalFreshCount constraints) := by
  induction constraints generalizing env start with
  | nil =>
      intro index _
      rfl
  | cons expression rest inductionHypothesis =>
      let afterFirst := executeConstraint env expression start
      have first := executeConstraint_agreesOutside env expression start
      have tail := inductionHypothesis afterFirst
        (start + constraintFreshCount expression)
      simpa [executeConstraints, afterFirst, totalFreshCount, Nat.add_assoc] using
        first.append tail

theorem executeConstraints_holds_rows (env : Env)
    (constraints : List Expr) (start : Nat)
    (scope : ∀ expression ∈ constraints, expression.VarsBelow start)
    (logical : ConstraintsHold env constraints) :
    RowsHold (executeConstraints env constraints start)
      (lowerConstraints constraints start).rows := by
  induction constraints generalizing env start with
  | nil =>
      intro row member
      simp [lowerConstraints] at member
  | cons expression rest inductionHypothesis =>
      let firstFresh := constraintFreshCount expression
      let next := start + firstFresh
      let afterFirst := executeConstraint env expression start
      let completed := executeConstraints afterFirst rest next
      have expressionScope : expression.VarsBelow start :=
        scope expression (by simp)
      have restScope : ∀ current ∈ rest, current.VarsBelow start := by
        intro current member
        exact scope current (by simp [member])
      have restScopeAtNext : ∀ current ∈ rest,
          current.VarsBelow next := by
        intro current member
        exact Expr.VarsBelow.mono current (restScope current member) (by
          unfold next
          omega)
      have expressionLogical : expression.eval env = 0 :=
        logical expression (by simp)
      have firstAgrees := executeConstraint_agreesOutside env expression start
      have restLogical : ConstraintsHold afterFirst rest := by
        apply constraintsHold_of_agree_below env afterFirst rest start restScope
        · intro index below
          exact firstAgrees index (Or.inl below)
        · intro current member
          exact logical current (by simp [member])
      have restRows : RowsHold completed
          (lowerConstraints rest next).rows :=
        inductionHypothesis afterFirst next restScopeAtNext restLogical
      have tailAgrees := executeConstraints_agreesOutside afterFirst rest next
      have allAgrees := executeConstraints_agreesOutside env
        (expression :: rest) start
      have expressionLogicalCompleted : expression.eval completed = 0 := by
        rw [expression.eval_eq_of_agree_below start completed env
          expressionScope (fun index below => allAgrees index (Or.inl below))]
        exact expressionLogical
      have firstRows : RowsHold completed
          (lowerConstraint expression start).rows := by
        cases result : directConstraint expression with
        | some direct =>
            have fresh : constraintFreshCount expression = 0 := by
              simp [constraintFreshCount, result]
            exact lowerConstraint_complete_of_freshCount_zero completed
              expression start fresh expressionLogicalCompleted
        | none =>
            have immediate := executeConstraint_holds_rows env expression start
              expressionScope expressionLogical
            have genericScope := lowerGenericConstraint_rows_varsBelow
              expression start expressionScope
            have preserved := rowsHold_of_agree_below
              (lowerGenericConstraint expression start).rows next afterFirst
              completed (by
                intro row member
                have below := genericScope row member
                simpa [next, firstFresh, constraintFreshCount, result] using below)
              (fun index below => tailAgrees index (Or.inl below)) (by
                simpa [afterFirst, executeConstraint, lowerConstraint, result]
                  using immediate)
            simpa [lowerConstraint, result] using preserved
      have combined :=
        (rowsHold_append completed
          (lowerConstraint expression start).rows
          (lowerConstraints rest next).rows).mpr ⟨firstRows, restRows⟩
      simpa [executeConstraints, lowerConstraints, afterFirst, completed,
        firstFresh, next] using combined

theorem lowerConstraints_complete (env : Env) (constraints : List Expr)
    (start : Nat)
    (scope : ∀ expression ∈ constraints, expression.VarsBelow start)
    (logical : ConstraintsHold env constraints) :
    ∃ completed,
      AgreesOutside env completed start (totalFreshCount constraints) ∧
      RowsHold completed (lowerConstraints constraints start).rows := by
  exact ⟨executeConstraints env constraints start,
    executeConstraints_agreesOutside env constraints start,
    executeConstraints_holds_rows env constraints start scope logical⟩

theorem LoweringPlan.complete (plan : LoweringPlan) (env : Env)
    (scope : ∀ expression ∈ plan.constraints,
      expression.VarsBelow plan.firstFresh)
    (logical : ConstraintsHold env plan.constraints) :
    ∃ completed,
      AgreesOutside env completed plan.firstFresh plan.freshColumnCount ∧
      RowsHold completed plan.rows := by
  exact lowerConstraints_complete env plan.constraints plan.firstFresh scope
    logical

end NightstreamFPrime.Layout.R1CS
