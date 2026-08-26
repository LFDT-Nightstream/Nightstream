import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns the generic compact-row template compiler and its lossless expansion
theorems.

Templates contain Lean-built expressions and R1CS rows. An invocation only
renames input columns and shifts the contiguous R1CS-fresh interval. The proof
is structural in one expression and does not inspect an emitted artifact.
-/

namespace NightstreamFPrime.Export.Stage1.CompactRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

def renameExpr (column : Nat → Nat) : Expr → Expr
  | .var index => .var (column index)
  | .const value => .const value
  | .add left right => .add (renameExpr column left) (renameExpr column right)
  | .mul left right => .mul (renameExpr column left) (renameExpr column right)

theorem renameExpr_comp (outer inner : Nat → Nat) (expression : Expr) :
    renameExpr outer (renameExpr inner expression) =
      renameExpr (outer ∘ inner) expression := by
  induction expression with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp [renameExpr, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp [renameExpr, leftIH, rightIH]

theorem renameExpr_congr {bound : Nat} (left right : Nat → Nat)
    (expression : Expr) (scope : expression.VarsBelow bound)
    (agree : ∀ index, index < bound → left index = right index) :
    renameExpr left expression = renameExpr right expression := by
  induction expression with
  | var index =>
      simpa [renameExpr] using agree index scope
  | const value => rfl
  | add first second firstIH secondIH =>
      simp only [renameExpr]
      rw [firstIH scope.1, secondIH scope.2]
  | mul first second firstIH secondIH =>
      simp only [renameExpr]
      rw [firstIH scope.1, secondIH scope.2]

@[simp] theorem renameExpr_sub (column : Nat → Nat) (left right : Expr) :
    renameExpr column (left - right) =
      renameExpr column left - renameExpr column right := by
  rfl

@[simp] theorem renameExpr_mulCount (column : Nat → Nat)
    (expression : Expr) :
    R1CS.mulCount (renameExpr column expression) = R1CS.mulCount expression := by
  induction expression with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp [renameExpr, R1CS.mulCount, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp [renameExpr, R1CS.mulCount, leftIH, rightIH]

def renameCombination (column : Nat → Nat)
    (combination : R1CS.LinearCombination) : R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (column term.1, term.2)⟩

def renameRow (column : Nat → Nat) (row : R1CS.Row) : R1CS.Row :=
  ⟨renameCombination column row.a, renameCombination column row.b,
    renameCombination column row.c⟩

theorem renameCombination_comp (outer inner : Nat → Nat)
    (combination : R1CS.LinearCombination) :
    renameCombination outer (renameCombination inner combination) =
      renameCombination (outer ∘ inner) combination := by
  cases combination
  simp [renameCombination, List.map_map, Function.comp_def]

theorem renameRow_comp (outer inner : Nat → Nat) (row : R1CS.Row) :
    renameRow outer (renameRow inner row) =
      renameRow (outer ∘ inner) row := by
  cases row
  simp [renameRow, renameCombination_comp]

def relocate (inputCount shift : Nat) (inputColumn : Nat → Nat)
    (column : Nat) : Nat :=
  if column < inputCount then inputColumn column else column + shift

@[simp] theorem relocate_input (inputCount shift : Nat)
    (inputColumn : Nat → Nat) (column : Nat) (bound : column < inputCount) :
    relocate inputCount shift inputColumn column = inputColumn column := by
  simp [relocate, bound]

@[simp] theorem relocate_local (inputCount shift : Nat)
    (inputColumn : Nat → Nat) (column : Nat) (bound : inputCount ≤ column) :
    relocate inputCount shift inputColumn column = column + shift := by
  simp [relocate, Nat.not_lt.mpr bound]

@[simp] theorem renameCombination_zero (column : Nat → Nat) :
    renameCombination column R1CS.LinearCombination.zero =
      R1CS.LinearCombination.zero := by
  rfl

@[simp] theorem renameCombination_one (column : Nat → Nat) :
    renameCombination column R1CS.LinearCombination.one =
      R1CS.LinearCombination.one := by
  rfl

@[simp] theorem renameCombination_const (column : Nat → Nat) (value : F) :
    renameCombination column (R1CS.LinearCombination.const value) =
      R1CS.LinearCombination.const value := by
  rfl

@[simp] theorem renameCombination_ofVar (column : Nat → Nat) (index : Nat) :
    renameCombination column (R1CS.LinearCombination.ofVar index) =
      R1CS.LinearCombination.ofVar (column index) := by
  rfl

@[simp] theorem renameCombination_add (column : Nat → Nat)
    (left right : R1CS.LinearCombination) :
    renameCombination column (R1CS.LinearCombination.add left right) =
      R1CS.LinearCombination.add (renameCombination column left)
        (renameCombination column right) := by
  cases left
  cases right
  simp [renameCombination, R1CS.LinearCombination.add, List.map_append]

private theorem varsBelow_left {left right : Expr} {bound : Nat}
    (scope : (left + right).VarsBelow bound) : left.VarsBelow bound := by
  exact scope.1

private theorem varsBelow_right {left right : Expr} {bound : Nat}
    (scope : (left + right).VarsBelow bound) : right.VarsBelow bound := by
  exact scope.2

private theorem varsBelow_mul_left {left right : Expr} {bound : Nat}
    (scope : (left * right).VarsBelow bound) : left.VarsBelow bound := by
  exact scope.1

private theorem varsBelow_mul_right {left right : Expr} {bound : Nat}
    (scope : (left * right).VarsBelow bound) : right.VarsBelow bound := by
  exact scope.2

/-- Renaming inputs does not change allocation; shifting the first fresh
column shifts the final fresh boundary by the same amount. -/
theorem lowerExpression_next_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (expression : Expr) :
    (R1CS.lowerExpression (renameExpr inputColumn expression)
        (start + shift)).next =
      (R1CS.lowerExpression expression start).next + shift := by
  rw [R1CS.lowerExpression_next, R1CS.lowerExpression_next,
    renameExpr_mulCount]
  omega

theorem lowerExpression_value_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (expression : Expr)
    (startBound : inputCount ≤ start)
    (scope : expression.VarsBelow inputCount) :
    (R1CS.lowerExpression (renameExpr inputColumn expression)
        (start + shift)).value =
      renameCombination (relocate inputCount shift inputColumn)
        (R1CS.lowerExpression expression start).value := by
  induction expression generalizing start with
  | var index =>
      have indexBound : index < inputCount := scope
      simp only [renameExpr, R1CS.lowerExpression]
      rw [renameCombination_ofVar,
        relocate_input inputCount shift inputColumn index indexBound]
  | const value =>
      simp only [renameExpr, R1CS.lowerExpression]
      rw [renameCombination_const]
  | add left right leftIH rightIH =>
      have leftScope := varsBelow_left scope
      have rightScope := varsBelow_right scope
      have nextBound : inputCount ≤ (R1CS.lowerExpression left start).next := by
        rw [R1CS.lowerExpression_next]
        omega
      simp only [renameExpr, R1CS.lowerExpression]
      rw [lowerExpression_next_rename inputCount start shift inputColumn left,
        leftIH start startBound leftScope,
        rightIH (R1CS.lowerExpression left start).next nextBound rightScope,
        renameCombination_add]
  | mul left right leftIH rightIH =>
      have nextBound : inputCount ≤ (R1CS.lowerExpression left start).next := by
        rw [R1CS.lowerExpression_next]
        omega
      have outputBound : inputCount ≤
          (R1CS.lowerExpression right
            (R1CS.lowerExpression left start).next).next := by
        rw [R1CS.lowerExpression_next, R1CS.lowerExpression_next]
        omega
      simp only [renameExpr, R1CS.lowerExpression]
      rw [lowerExpression_next_rename inputCount start shift inputColumn left,
        lowerExpression_next_rename inputCount
          (R1CS.lowerExpression left start).next shift inputColumn right,
        renameCombination_ofVar,
        relocate_local inputCount shift inputColumn _ outputBound]

theorem lowerExpression_rows_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (expression : Expr)
    (startBound : inputCount ≤ start)
    (scope : expression.VarsBelow inputCount) :
    (R1CS.lowerExpression (renameExpr inputColumn expression)
        (start + shift)).rows =
      (R1CS.lowerExpression expression start).rows.map
        (renameRow (relocate inputCount shift inputColumn)) := by
  induction expression generalizing start with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      have leftScope := varsBelow_left scope
      have rightScope := varsBelow_right scope
      have nextBound : inputCount ≤ (R1CS.lowerExpression left start).next := by
        rw [R1CS.lowerExpression_next]
        omega
      simp only [renameExpr, R1CS.lowerExpression]
      rw [lowerExpression_next_rename inputCount start shift inputColumn left,
        leftIH start startBound leftScope,
        rightIH (R1CS.lowerExpression left start).next nextBound rightScope,
        List.map_append]
  | mul left right leftIH rightIH =>
      have leftScope := varsBelow_mul_left scope
      have rightScope := varsBelow_mul_right scope
      have nextBound : inputCount ≤ (R1CS.lowerExpression left start).next := by
        rw [R1CS.lowerExpression_next]
        omega
      have outputBound : inputCount ≤
          (R1CS.lowerExpression right
            (R1CS.lowerExpression left start).next).next := by
        rw [R1CS.lowerExpression_next, R1CS.lowerExpression_next]
        omega
      simp only [renameExpr, R1CS.lowerExpression]
      rw [lowerExpression_next_rename inputCount start shift inputColumn left,
        leftIH start startBound leftScope,
        rightIH (R1CS.lowerExpression left start).next nextBound rightScope,
        lowerExpression_value_rename inputCount start shift inputColumn left
          startBound leftScope,
        lowerExpression_value_rename inputCount
          (R1CS.lowerExpression left start).next shift inputColumn right
          nextBound rightScope,
        lowerExpression_next_rename inputCount
          (R1CS.lowerExpression left start).next shift inputColumn right,
        List.map_append, List.map_singleton]
      simp [renameRow, renameCombination_ofVar]
      rw [relocate_local inputCount shift inputColumn
        (start + R1CS.mulCount left + R1CS.mulCount right) (by omega)]

/-- Generic one-constraint lowering, including its final assertion row, is
equivariant under the same renaming. -/
theorem lowerGenericConstraint_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (expression : Expr)
    (startBound : inputCount ≤ start)
    (scope : expression.VarsBelow inputCount) :
    (R1CS.lowerGenericConstraint (renameExpr inputColumn expression)
        (start + shift)).rows =
      (R1CS.lowerGenericConstraint expression start).rows.map
        (renameRow (relocate inputCount shift inputColumn)) := by
  let actual := R1CS.lowerExpression (renameExpr inputColumn expression)
    (start + shift)
  let normalized := R1CS.lowerExpression expression start
  have rowsEq : actual.rows = normalized.rows.map
      (renameRow (relocate inputCount shift inputColumn)) := by
    exact lowerExpression_rows_rename inputCount start shift inputColumn
      expression startBound scope
  have valueEq : actual.value = renameCombination
      (relocate inputCount shift inputColumn) normalized.value := by
    exact lowerExpression_value_rename inputCount start shift inputColumn
      expression startBound scope
  change actual.rows ++
      [({ a := actual.value
          b := R1CS.LinearCombination.one
          c := R1CS.LinearCombination.zero } : R1CS.Row)] =
    (normalized.rows ++
      [({ a := normalized.value
          b := R1CS.LinearCombination.one
          c := R1CS.LinearCombination.zero } : R1CS.Row)]).map
      (renameRow (relocate inputCount shift inputColumn))
  rw [rowsEq, valueEq, List.map_append, List.map_singleton]
  simp [renameRow]

def abstractColumn (inputCount : Nat) (column : Nat) : ColumnRef :=
  if column < inputCount then .input column else .local (column - inputCount)

def abstractTerm (inputCount : Nat) (term : Nat × F) : TemplateTerm :=
  ⟨abstractColumn inputCount term.1, term.2.val⟩

def abstractCombination (inputCount : Nat)
    (combination : R1CS.LinearCombination) : TemplateCombination :=
  ⟨combination.constant.val, combination.terms.map (abstractTerm inputCount)⟩

def outputLocal? (inputCount : Nat)
    (combination : R1CS.LinearCombination) : Option Nat :=
  match Rows.target? combination with
  | some target =>
      if inputCount ≤ target then some (target - inputCount) else none
  | none => none

def abstractRow (inputCount : Nat) (row : R1CS.Row) : CompactTemplateRow :=
  ⟨outputLocal? inputCount row.c,
    abstractCombination inputCount row.a,
    abstractCombination inputCount row.b,
    abstractCombination inputCount row.c⟩

abbrev inputColumnOfRanges := compactInputColumn

def instantiateColumn (inputColumn : Nat → Nat) (localStart : Nat) :
    ColumnRef → Nat
  | .input index => inputColumn index
  | .local index => localStart + index

def instantiateCombination (inputColumn : Nat → Nat) (localStart : Nat)
    (combination : TemplateCombination) : R1CS.LinearCombination :=
  ⟨fieldValue combination.constant,
    combination.terms.map fun term =>
      (instantiateColumn inputColumn localStart term.column,
        fieldValue term.coefficient)⟩

def instantiateRow (inputColumn : Nat → Nat) (localStart : Nat)
    (row : CompactTemplateRow) : R1CS.Row :=
  ⟨instantiateCombination inputColumn localStart row.a,
    instantiateCombination inputColumn localStart row.b,
    instantiateCombination inputColumn localStart row.c⟩

def instantiateRows (inputColumn : Nat → Nat) (localStart : Nat)
    (template : CompactRowTemplate) : List R1CS.Row :=
  template.rows.map (instantiateRow inputColumn localStart)

/-- The compact compiler and the generic package semantics instantiate the
same final R1CS rows. -/
theorem instantiateRows_eq_package (invocation : CompactRowInvocation)
    (template : CompactRowTemplate) :
    instantiateRows (inputColumnOfRanges invocation.inputRanges)
        invocation.localStart template =
      template.rows.map (instantiateCompactRow invocation) := by
  rfl

def compactTemplate (inputCount outputInput : Nat)
    (outputRecipe : Expr) : CompactRowTemplate :=
  let constraint := Expr.var outputInput - outputRecipe
  let lowered := R1CS.lowerGenericConstraint constraint inputCount
  { inputCount := inputCount
    localColumnCount := R1CS.mulCount constraint
    outputInput := outputInput
    outputRecipe := outputRecipe
    rows := lowered.rows.map (abstractRow inputCount) }

def compactConstraintTemplate (inputCount outputInput : Nat)
    (outputRecipe : Expr) : CompactRowTemplate :=
  let constraint := Expr.var outputInput - outputRecipe
  let lowered := R1CS.lowerConstraint constraint inputCount
  { inputCount := inputCount
    localColumnCount := R1CS.constraintFreshCount constraint
    outputInput := outputInput
    outputRecipe := outputRecipe
    rows := lowered.rows.map (abstractRow inputCount) }

@[simp] theorem compactConstraintTemplate_rows_length
    (inputCount outputInput : Nat)
    (outputRecipe : Expr) :
    (compactConstraintTemplate inputCount outputInput outputRecipe).rows.length =
      R1CS.constraintRowCount (Expr.var outputInput - outputRecipe) := by
  simp [compactConstraintTemplate]

private theorem instantiate_abstractColumn (inputCount shift : Nat)
    (inputColumn : Nat → Nat) (column : Nat) :
    instantiateColumn inputColumn (inputCount + shift)
        (abstractColumn inputCount column) =
      relocate inputCount shift inputColumn column := by
  by_cases input : column < inputCount
  · simp [abstractColumn, instantiateColumn, relocate, input]
  · have localBound : inputCount ≤ column := Nat.le_of_not_gt input
    simp [abstractColumn, instantiateColumn, relocate, input]
    omega

theorem instantiate_abstractCombination (inputCount shift : Nat)
    (inputColumn : Nat → Nat) (combination : R1CS.LinearCombination) :
    instantiateCombination inputColumn (inputCount + shift)
        (abstractCombination inputCount combination) =
      renameCombination (relocate inputCount shift inputColumn)
        combination := by
  cases combination with
  | mk constant terms =>
      unfold instantiateCombination abstractCombination renameCombination
      simp only [Rows.fieldValue_val, List.map_map]
      congr 1
      apply List.map_congr_left
      intro term member
      cases term with
      | mk column coefficient =>
          simp [abstractTerm, instantiate_abstractColumn, Rows.fieldValue_val]

theorem instantiate_abstractRow (inputCount shift : Nat)
    (inputColumn : Nat → Nat) (row : R1CS.Row) :
    instantiateRow inputColumn (inputCount + shift)
        (abstractRow inputCount row) =
      renameRow (relocate inputCount shift inputColumn) row := by
  cases row
  simp [instantiateRow, abstractRow, renameRow,
    instantiate_abstractCombination]

private theorem instantiate_abstractColumn_congr (inputCount localStart : Nat)
    (leftInput rightInput : Nat → Nat)
    (agree : ∀ input, input < inputCount →
      leftInput input = rightInput input)
    (column : Nat) :
    instantiateColumn leftInput localStart (abstractColumn inputCount column) =
      instantiateColumn rightInput localStart
        (abstractColumn inputCount column) := by
  by_cases input : column < inputCount
  · simp [abstractColumn, instantiateColumn, input, agree column input]
  · simp [abstractColumn, instantiateColumn, input]

private theorem instantiate_abstractCombination_congr
    (inputCount localStart : Nat) (leftInput rightInput : Nat → Nat)
    (agree : ∀ input, input < inputCount →
      leftInput input = rightInput input)
    (combination : R1CS.LinearCombination) :
    instantiateCombination leftInput localStart
        (abstractCombination inputCount combination) =
      instantiateCombination rightInput localStart
        (abstractCombination inputCount combination) := by
  cases combination with
  | mk constant terms =>
      unfold instantiateCombination abstractCombination
      simp only [Rows.fieldValue_val, List.map_map]
      congr 1
      apply List.map_congr_left
      intro term member
      cases term with
      | mk column coefficient =>
          simp [abstractTerm,
            instantiate_abstractColumn_congr inputCount localStart leftInput
              rightInput agree column, Rows.fieldValue_val]

private theorem instantiate_abstractRow_congr (inputCount localStart : Nat)
    (leftInput rightInput : Nat → Nat)
    (agree : ∀ input, input < inputCount →
      leftInput input = rightInput input)
    (row : R1CS.Row) :
    instantiateRow leftInput localStart (abstractRow inputCount row) =
      instantiateRow rightInput localStart (abstractRow inputCount row) := by
  cases row
  simp [instantiateRow, abstractRow,
    instantiate_abstractCombination_congr inputCount localStart leftInput
      rightInput agree]

/-- Expanding a compact template gives exactly the renamed generic Lean
lowering, entry for entry and in the same row order. -/
theorem instantiate_compactTemplate (inputCount outputInput shift : Nat)
    (inputColumn : Nat → Nat) (outputRecipe : Expr) :
    instantiateRows inputColumn (inputCount + shift)
        (compactTemplate inputCount outputInput outputRecipe) =
      (R1CS.lowerGenericConstraint
        (Expr.var outputInput - outputRecipe) inputCount).rows.map
          (renameRow (relocate inputCount shift inputColumn)) := by
  unfold instantiateRows compactTemplate
  rw [List.map_map]
  apply List.map_congr_left
  intro row member
  exact instantiate_abstractRow inputCount shift inputColumn row

/-- The optimized compact constructor expands to the exact optimized Lean
constraint lowering under the declared input and local-column maps. -/
theorem instantiate_compactConstraintTemplate
    (inputCount outputInput shift : Nat) (inputColumn : Nat → Nat)
    (outputRecipe : Expr) :
    instantiateRows inputColumn (inputCount + shift)
        (compactConstraintTemplate inputCount outputInput outputRecipe) =
      (R1CS.lowerConstraint
        (Expr.var outputInput - outputRecipe) inputCount).rows.map
          (renameRow (relocate inputCount shift inputColumn)) := by
  unfold instantiateRows compactConstraintTemplate
  rw [List.map_map]
  apply List.map_congr_left
  intro row _member
  exact instantiate_abstractRow inputCount shift inputColumn row

theorem renameCombination_eval (column : Nat → Nat)
    (combination : R1CS.LinearCombination) (env : Env) :
    (renameCombination column combination).eval env =
      combination.eval (fun index => env (column index)) := by
  cases combination
  simp [renameCombination, R1CS.LinearCombination.eval, List.map_map,
    Function.comp_def]

theorem renameRow_holds (column : Nat → Nat) (row : R1CS.Row)
    (env : Env) :
    (renameRow column row).Holds env ↔
      row.Holds (fun index => env (column index)) := by
  cases row
  simp [renameRow, R1CS.Row.Holds, renameCombination_eval]

theorem rowsHold_map_renameRow (column : Nat → Nat)
    (rows : List R1CS.Row) (env : Env)
    (holds : R1CS.RowsHold env (rows.map (renameRow column))) :
    R1CS.RowsHold (fun index => env (column index)) rows := by
  intro row member
  apply (renameRow_holds column row env).mp
  exact holds (renameRow column row) (List.mem_map.mpr ⟨row, member, rfl⟩)

/-- Holding every instantiated row of an optimized compact template implies
the exact normalized constraint. -/
theorem compactConstraintTemplate_rows_imply_eval_zero
    (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
    (outputRecipe : Expr) (env : Env)
    (localBound : inputCount ≤ localStart)
    (holds : R1CS.RowsHold env
      (instantiateRows inputColumn localStart
        (compactConstraintTemplate inputCount outputInput outputRecipe))) :
    (Expr.var outputInput - outputRecipe).eval
        (fun index => env
          (relocate inputCount (localStart - inputCount) inputColumn index)) =
      0 := by
  have startEq : inputCount + (localStart - inputCount) = localStart := by
    omega
  have expanded := instantiate_compactConstraintTemplate inputCount
    outputInput (localStart - inputCount) inputColumn outputRecipe
  rw [startEq] at expanded
  rw [expanded] at holds
  apply R1CS.lowerConstraint_sound _ _ inputCount
  exact rowsHold_map_renameRow
    (relocate inputCount (localStart - inputCount) inputColumn)
    (R1CS.lowerConstraint
      (Expr.var outputInput - outputRecipe) inputCount).rows env holds

private def compactInputEnv (inputCount : Nat) (inputColumn : Nat → Nat)
    (env : Env) : Env :=
  fun index => if index < inputCount then env (inputColumn index) else 0

private def copyCompactLocals (inputCount localStart count : Nat)
    (base source : Env) : Env :=
  fun index =>
    if localStart ≤ index ∧ index < localStart + count then
      source (inputCount + (index - localStart))
    else
      base index

private theorem copyCompactLocals_agreesOutside
    (inputCount localStart count : Nat) (base source : Env) :
    AgreesOutside base
      (copyCompactLocals inputCount localStart count base source)
      localStart count := by
  intro index outside
  unfold copyCompactLocals
  rw [if_neg]
  intro inside
  rcases outside with before | after <;> omega

/-- Constructive completeness for one optimized compact constraint template.
Only the declared invocation-local fresh interval changes. Exact input columns
must lie outside that interval, as enforced by the strict package loader. -/
theorem compactConstraintTemplate_complete
    (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
    (outputRecipe : Expr) (env : Env)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + R1CS.constraintFreshCount
          (Expr.var outputInput - outputRecipe) ≤ inputColumn input)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (logical : (Expr.var outputInput - outputRecipe).eval
      (fun index => env
        (relocate inputCount (localStart - inputCount) inputColumn index)) = 0) :
    ∃ completed,
      AgreesOutside env completed localStart
          (R1CS.constraintFreshCount
            (Expr.var outputInput - outputRecipe)) ∧
        R1CS.RowsHold completed
          (instantiateRows inputColumn localStart
            (compactConstraintTemplate inputCount outputInput outputRecipe)) := by
  let expression := Expr.var outputInput - outputRecipe
  let abstract := compactInputEnv inputCount inputColumn env
  let source := R1CS.executeConstraint abstract expression inputCount
  let count := R1CS.constraintFreshCount expression
  let completed := copyCompactLocals inputCount localStart count env source
  have abstractLogical : expression.eval abstract = 0 := by
    rw [expression.eval_eq_of_agree_below inputCount abstract
      (fun index => env
        (relocate inputCount (localStart - inputCount) inputColumn index))
      scope]
    · exact logical
    · intro index below
      unfold abstract compactInputEnv
      rw [if_pos below, relocate_input inputCount (localStart - inputCount)
        inputColumn index below]
  have sourceAgrees : AgreesOutside abstract source inputCount count := by
    exact R1CS.executeConstraint_agreesOutside abstract expression inputCount
  have sourceRows : R1CS.RowsHold source
      (R1CS.lowerConstraint expression inputCount).rows := by
    exact R1CS.executeConstraint_holds_rows abstract expression inputCount scope
      abstractLogical
  have mappedAgrees : ∀ index, index < inputCount + count →
      completed
          (relocate inputCount (localStart - inputCount) inputColumn index) =
        source index := by
    intro index below
    by_cases input : index < inputCount
    · rw [relocate_input inputCount (localStart - inputCount) inputColumn
        index input]
      unfold completed copyCompactLocals
      rw [if_neg]
      · unfold abstract compactInputEnv at sourceAgrees
        have stable := sourceAgrees index (Or.inl input)
        simpa only [if_pos input] using stable.symm
      · intro inside
        rcases inputsOutside index input with before | after
        · omega
        · change localStart + count ≤ inputColumn index at after
          omega
    · have inputLe : inputCount ≤ index := Nat.le_of_not_gt input
      rw [relocate_local inputCount (localStart - inputCount) inputColumn
        index inputLe]
      have startEq : inputCount + (localStart - inputCount) = localStart := by
        omega
      have inside : localStart ≤ index + (localStart - inputCount) ∧
          index + (localStart - inputCount) < localStart + count := by
        constructor <;> omega
      unfold completed copyCompactLocals
      rw [if_pos inside]
      congr 1
      omega
  refine ⟨completed,
    copyCompactLocals_agreesOutside inputCount localStart count env source, ?_⟩
  have startEq : inputCount + (localStart - inputCount) = localStart := by
    omega
  have expanded := instantiate_compactConstraintTemplate inputCount outputInput
    (localStart - inputCount) inputColumn outputRecipe
  rw [startEq] at expanded
  rw [expanded]
  intro row member
  rcases List.mem_map.mp member with ⟨sourceRow, sourceMember, rfl⟩
  apply (renameRow_holds
    (relocate inputCount (localStart - inputCount) inputColumn)
    sourceRow completed).mpr
  exact sourceRow.holds_of_agree_below (inputCount + count) source
    (fun index => completed
      (relocate inputCount (localStart - inputCount) inputColumn index))
    (R1CS.lowerConstraint_rows_varsBelow expression inputCount scope sourceRow
      sourceMember)
    mappedAgrees (sourceRows sourceRow sourceMember)

/-- Constructive completeness for one generic compact constraint template.
This is the corresponding package-completeness primitive for combination
rows, whose template uses the generic expression lowerer. -/
theorem compactTemplate_complete
    (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
    (outputRecipe : Expr) (env : Env)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + R1CS.mulCount
          (Expr.var outputInput - outputRecipe) ≤ inputColumn input)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (logical : (Expr.var outputInput - outputRecipe).eval
      (fun index => env
        (relocate inputCount (localStart - inputCount) inputColumn index)) = 0) :
    ∃ completed,
      AgreesOutside env completed localStart
          (R1CS.mulCount (Expr.var outputInput - outputRecipe)) ∧
        R1CS.RowsHold completed
          (instantiateRows inputColumn localStart
            (compactTemplate inputCount outputInput outputRecipe)) := by
  let expression := Expr.var outputInput - outputRecipe
  let abstract := compactInputEnv inputCount inputColumn env
  let source := R1CS.executeExpression abstract expression inputCount
  let lowered := R1CS.lowerExpression expression inputCount
  let assertion : R1CS.Row :=
    ⟨lowered.value, R1CS.LinearCombination.one,
      R1CS.LinearCombination.zero⟩
  let count := R1CS.mulCount expression
  let completed := copyCompactLocals inputCount localStart count env source
  have abstractLogical : expression.eval abstract = 0 := by
    rw [expression.eval_eq_of_agree_below inputCount abstract
      (fun index => env
        (relocate inputCount (localStart - inputCount) inputColumn index))
      scope]
    · exact logical
    · intro index below
      unfold abstract compactInputEnv
      rw [if_pos below, relocate_input inputCount (localStart - inputCount)
        inputColumn index below]
  have sourceAgrees : AgreesOutside abstract source inputCount count := by
    exact R1CS.executeExpression_agreesOutside abstract expression inputCount
  have expressionRows : R1CS.RowsHold source lowered.rows := by
    exact R1CS.executeExpression_holds_rows abstract expression inputCount scope
  have expressionEval : expression.eval source = 0 := by
    rw [expression.eval_eq_of_agree_below inputCount source abstract scope]
    · exact abstractLogical
    · intro index below
      exact sourceAgrees index (Or.inl below)
  have loweredEval : lowered.value.eval source = expression.eval source := by
    exact R1CS.lowerExpression_sound source expression inputCount expressionRows
  have assertionHolds : assertion.Holds source := by
    simp [assertion, R1CS.Row.Holds, loweredEval, expressionEval]
  have sourceRows : R1CS.RowsHold source
      (R1CS.lowerGenericConstraint expression inputCount).rows := by
    apply (R1CS.rowsHold_append source lowered.rows [assertion]).mpr
    refine ⟨expressionRows, ?_⟩
    intro row member
    simp only [List.mem_singleton] at member
    simpa [R1CS.lowerGenericConstraint, lowered, assertion, member] using
      assertionHolds
  have mappedAgrees : ∀ index, index < inputCount + count →
      completed
          (relocate inputCount (localStart - inputCount) inputColumn index) =
        source index := by
    intro index below
    by_cases input : index < inputCount
    · rw [relocate_input inputCount (localStart - inputCount) inputColumn
        index input]
      unfold completed copyCompactLocals
      rw [if_neg]
      · unfold abstract compactInputEnv at sourceAgrees
        have stable := sourceAgrees index (Or.inl input)
        simpa only [if_pos input] using stable.symm
      · intro inside
        rcases inputsOutside index input with before | after
        · omega
        · change localStart + count ≤ inputColumn index at after
          omega
    · have inputLe : inputCount ≤ index := Nat.le_of_not_gt input
      rw [relocate_local inputCount (localStart - inputCount) inputColumn
        index inputLe]
      have inside : localStart ≤ index + (localStart - inputCount) ∧
          index + (localStart - inputCount) < localStart + count := by
        constructor <;> omega
      unfold completed copyCompactLocals
      rw [if_pos inside]
      congr 1
      omega
  refine ⟨completed,
    copyCompactLocals_agreesOutside inputCount localStart count env source, ?_⟩
  have startEq : inputCount + (localStart - inputCount) = localStart := by
    omega
  have expanded := instantiate_compactTemplate inputCount outputInput
    (localStart - inputCount) inputColumn outputRecipe
  rw [startEq] at expanded
  rw [expanded]
  intro row member
  rcases List.mem_map.mp member with ⟨sourceRow, sourceMember, rfl⟩
  apply (renameRow_holds
    (relocate inputCount (localStart - inputCount) inputColumn)
    sourceRow completed).mpr
  exact sourceRow.holds_of_agree_below (inputCount + count) source
    (fun index => completed
      (relocate inputCount (localStart - inputCount) inputColumn index))
    (R1CS.lowerGenericConstraint_rows_varsBelow expression inputCount scope
      sourceRow sourceMember)
    mappedAgrees (sourceRows sourceRow sourceMember)

/-- Compact-template expansion observes only declared input slots. Values of
the input maps outside `inputCount` cannot affect an expanded row. -/
theorem instantiate_compactTemplate_congr_inputs
    (inputCount outputInput localStart : Nat)
    (leftInput rightInput : Nat → Nat) (outputRecipe : Expr)
    (agree : ∀ input, input < inputCount →
      leftInput input = rightInput input) :
    instantiateRows leftInput localStart
        (compactTemplate inputCount outputInput outputRecipe) =
      instantiateRows rightInput localStart
        (compactTemplate inputCount outputInput outputRecipe) := by
  unfold instantiateRows compactTemplate
  simp only [List.map_map]
  apply List.map_congr_left
  intro row member
  exact instantiate_abstractRow_congr inputCount localStart leftInput rightInput
    agree row

/-- Expanding after an affine final column map gives exactly the final-column
image of the Lean-lowered source rows. -/
theorem instantiate_compactTemplate_remap
    (inputCount outputInput sourceFresh finalFresh : Nat)
    (sourceInput remap : Nat → Nat) (outputRecipe : Expr)
    (sourceBound : inputCount ≤ sourceFresh)
    (finalBound : inputCount ≤ finalFresh)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (remapFresh : ∀ offset,
      remap (sourceFresh + offset) = finalFresh + offset) :
    instantiateRows (fun input => remap (sourceInput input)) finalFresh
        (compactTemplate inputCount outputInput outputRecipe) =
      (R1CS.lowerGenericConstraint
        (renameExpr sourceInput (Expr.var outputInput - outputRecipe))
        sourceFresh).rows.map (renameRow remap) := by
  have sourceStart : inputCount + (sourceFresh - inputCount) = sourceFresh := by
    omega
  have finalStart : inputCount + (finalFresh - inputCount) = finalFresh := by
    omega
  have instantiated := instantiate_compactTemplate inputCount outputInput
    (finalFresh - inputCount) (fun input => remap (sourceInput input))
    outputRecipe
  rw [finalStart] at instantiated
  have lowered := lowerGenericConstraint_rename inputCount inputCount
    (sourceFresh - inputCount) sourceInput
    (Expr.var outputInput - outputRecipe) (Nat.le_refl _) scope
  rw [sourceStart] at lowered
  rw [lowered, List.map_map]
  rw [instantiated]
  apply List.map_congr_left
  intro row member
  change renameRow
      (relocate inputCount (finalFresh - inputCount)
        (fun input => remap (sourceInput input))) row =
    renameRow remap
      (renameRow (relocate inputCount (sourceFresh - inputCount) sourceInput)
        row)
  rw [renameRow_comp]
  congr 1
  funext column
  unfold relocate Function.comp
  by_cases input : column < inputCount
  · simp [input]
  · have inputBound : inputCount ≤ column := Nat.le_of_not_gt input
    simp only [if_neg input]
    have sourceColumn : column + (sourceFresh - inputCount) =
        sourceFresh + (column - inputCount) := by omega
    have finalColumn : column + (finalFresh - inputCount) =
        finalFresh + (column - inputCount) := by omega
    rw [sourceColumn, remapFresh, finalColumn]

theorem renameExpr_eval (column : Nat → Nat) (expression : Expr)
    (env : Env) :
    (renameExpr column expression).eval env =
      expression.eval (fun index => env (column index)) := by
  induction expression with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp [renameExpr, Expr.eval, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp [renameExpr, Expr.eval, leftIH, rightIH]

end NightstreamFPrime.Export.Stage1.CompactRows
