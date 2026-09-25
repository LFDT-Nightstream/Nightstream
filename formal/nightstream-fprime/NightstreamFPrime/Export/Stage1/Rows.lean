import NightstreamFPrime.Export.RowSemantics

/-!
Owns the compact encoding of ordinary Stage 1 R1CS rows.

One row becomes:
- a `WitnessInstruction` when its `C` side is exactly one target variable;
- a `SparseRow` otherwise.

The conversion is lossless. It does not classify Poseidon2 template rows;
those use `PermutationInvocation` and are assembled separately.
-/

namespace NightstreamFPrime.Export.Stage1.Rows

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

def sparseTerm (term : Nat × F) : SparseTerm :=
  ⟨term.1, term.2.val⟩

def sparseCombination (combination : R1CS.LinearCombination) :
    SparseCombination :=
  ⟨combination.constant.val, combination.terms.map sparseTerm⟩

theorem fieldValue_val (value : F) : fieldValue value.val = value := by
  apply Fin.ext
  simp [fieldValue, Spec.Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

theorem sparseCombination_toR1CS (combination : R1CS.LinearCombination) :
    (sparseCombination combination).toR1CS = combination := by
  cases combination with
  | mk constant terms =>
      simp only [sparseCombination, SparseCombination.toR1CS]
      apply congrArg₂ R1CS.LinearCombination.mk
      · exact fieldValue_val constant
      · rw [List.map_map]
        calc
          List.map
              ((fun term =>
                (term.column, fieldValue term.coefficient)) ∘ sparseTerm)
              terms = List.map id terms := by
            apply List.map_congr_left
            intro term member
            cases term with
            | mk column coefficient =>
                simp [sparseTerm, fieldValue_val]
          _ = terms := by simp

/-- Recognize exactly `0 + 1·target`. -/
def target? (combination : R1CS.LinearCombination) : Option Nat :=
  if combination.constant = 0 then
    match combination.terms with
    | [(target, coefficient)] =>
        if coefficient = 1 then some target else none
    | _ => none
  else
    none

theorem target?_eq_some {combination : R1CS.LinearCombination}
    {target : Nat} (found : target? combination = some target) :
    combination = R1CS.LinearCombination.ofVar target := by
  rcases combination with ⟨constant, terms⟩
  by_cases constantZero : constant = 0
  · subst constant
    cases terms with
    | nil => simp [target?] at found
    | cons term rest =>
        cases rest with
        | nil =>
            rcases term with ⟨column, coefficient⟩
            by_cases coefficientOne : coefficient = 1
            · subst coefficient
              simp [target?] at found
              subst column
              rfl
            · simp [target?, coefficientOne] at found
        | cons next tail => simp [target?] at found
  · simp [target?, constantZero] at found

inductive CompiledRow where
  | witness (instruction : WitnessInstruction)
  | assertion (row : SparseRow)
deriving Repr

def CompiledRow.rowIndex : CompiledRow → Nat
  | .witness instruction => instruction.rowIndex
  | .assertion row => row.rowIndex

def CompiledRow.toR1CS : CompiledRow → R1CS.Row
  | .witness instruction => instruction.toR1CS
  | .assertion row => row.toR1CS

def compileRow (freshStart rowIndex : Nat) (row : R1CS.Row) : CompiledRow :=
  match target? row.c with
  | some target =>
      if freshStart ≤ target then
        .witness
          ⟨rowIndex, target, sparseCombination row.a, sparseCombination row.b⟩
      else
        .assertion
          ⟨rowIndex, sparseCombination row.a, sparseCombination row.b,
            sparseCombination row.c⟩
  | none => .assertion
      ⟨rowIndex, sparseCombination row.a, sparseCombination row.b,
        sparseCombination row.c⟩

theorem compileRow_toR1CS (freshStart rowIndex : Nat) (row : R1CS.Row) :
    (compileRow freshStart rowIndex row).toR1CS = row := by
  unfold compileRow
  cases found : target? row.c with
  | none =>
      simp [CompiledRow.toR1CS, SparseRow.toR1CS,
        sparseCombination_toR1CS]
  | some target =>
      by_cases fresh : freshStart ≤ target
      · simp only [fresh, if_pos, CompiledRow.toR1CS,
          WitnessInstruction.toR1CS, sparseCombination_toR1CS]
        have cEquals : row.c = R1CS.LinearCombination.ofVar target :=
          target?_eq_some found
        cases row
        simp_all
      · simp [fresh, CompiledRow.toR1CS, SparseRow.toR1CS,
          sparseCombination_toR1CS]

@[simp] theorem compileRow_rowIndex (freshStart rowIndex : Nat)
    (row : R1CS.Row) :
    (compileRow freshStart rowIndex row).rowIndex = rowIndex := by
  unfold compileRow
  split
  · split <;> rfl
  · rfl

def compileRowsFrom (freshStart : Nat) : Nat → List R1CS.Row → List CompiledRow
  | _, [] => []
  | rowIndex, row :: rows =>
      compileRow freshStart rowIndex row ::
        compileRowsFrom freshStart (rowIndex + 1) rows

def compileRows (freshStart rowStart : Nat) (rows : List R1CS.Row) :
    List CompiledRow :=
  compileRowsFrom freshStart rowStart rows

private def lowerConstraintsRev :
    List Expr → Nat → List R1CS.Row → R1CS.LoweredConstraints
  | [], next, rowsRev => ⟨next, rowsRev.reverse⟩
  | expression :: rest, next, rowsRev =>
      let first := R1CS.lowerConstraint expression next
      lowerConstraintsRev rest first.next (first.rows.reverse ++ rowsRev)

private theorem lowerConstraintsRev_eq (constraints : List Expr)
    (next : Nat) (rowsRev : List R1CS.Row) :
    lowerConstraintsRev constraints next rowsRev =
      let lowered := R1CS.lowerConstraints constraints next
      ⟨lowered.next, rowsRev.reverse ++ lowered.rows⟩ := by
  induction constraints generalizing next rowsRev with
  | nil => simp [lowerConstraintsRev, R1CS.lowerConstraints]
  | cons expression rest inductionHypothesis =>
      simp only [lowerConstraintsRev, R1CS.lowerConstraints]
      let first := R1CS.lowerConstraint expression next
      rw [inductionHypothesis]
      apply congrArg₂ R1CS.LoweredConstraints.mk
      · rfl
      · simp [List.reverse_append, List.append_assoc]

/-- Stack-safe executable form of the proved physical lowering. -/
def lowerConstraintsTR (constraints : List Expr) (start : Nat) :
    R1CS.LoweredConstraints :=
  lowerConstraintsRev constraints start []

theorem lowerConstraintsTR_eq (constraints : List Expr) (start : Nat) :
    lowerConstraintsTR constraints start =
      R1CS.lowerConstraints constraints start := by
  rw [lowerConstraintsTR, lowerConstraintsRev_eq]
  generalize loweredEquation :
    R1CS.lowerConstraints constraints start = lowered
  cases lowered with
  | mk next rows =>
      have nextEquation : next = start + R1CS.totalFreshCount constraints := by
        have authoritative := R1CS.lowerConstraints_next constraints start
        rw [loweredEquation] at authoritative
        exact authoritative
      subst next
      rfl

private def compileRowsFromRev :
    Nat → Nat → List R1CS.Row → List CompiledRow → List CompiledRow
  | _, _, [], rowsRev => rowsRev.reverse
  | freshStart, rowIndex, row :: rows, rowsRev =>
      compileRowsFromRev freshStart (rowIndex + 1) rows
        (compileRow freshStart rowIndex row :: rowsRev)

private theorem compileRowsFromRev_eq (freshStart rowStart : Nat)
    (rows : List R1CS.Row) (rowsRev : List CompiledRow) :
    compileRowsFromRev freshStart rowStart rows rowsRev =
      rowsRev.reverse ++ compileRowsFrom freshStart rowStart rows := by
  induction rows generalizing rowStart rowsRev with
  | nil => simp [compileRowsFromRev, compileRowsFrom]
  | cons row rows inductionHypothesis =>
      simp only [compileRowsFromRev, compileRowsFrom]
      rw [inductionHypothesis]
      simp [List.reverse_cons, List.append_assoc]

/-- Stack-safe executable form of the lossless row compiler. -/
def compileRowsTR (freshStart rowStart : Nat) (rows : List R1CS.Row) :
    List CompiledRow :=
  compileRowsFromRev freshStart rowStart rows []

theorem compileRowsTR_eq (freshStart rowStart : Nat) (rows : List R1CS.Row) :
    compileRowsTR freshStart rowStart rows =
      compileRows freshStart rowStart rows := by
  rw [compileRowsTR, compileRowsFromRev_eq]
  rfl

def witnessInstructions : List CompiledRow → List WitnessInstruction
  | [] => []
  | .witness instruction :: rows =>
      instruction :: witnessInstructions rows
  | .assertion _ :: rows => witnessInstructions rows

def assertionRows : List CompiledRow → List SparseRow
  | [] => []
  | .witness _ :: rows => assertionRows rows
  | .assertion assertion :: rows => assertion :: assertionRows rows

private def witnessInstructionsRev :
    List CompiledRow → List WitnessInstruction → List WitnessInstruction
  | [], instructionsRev => instructionsRev.reverse
  | .witness instruction :: rows, instructionsRev =>
      witnessInstructionsRev rows (instruction :: instructionsRev)
  | .assertion _ :: rows, instructionsRev =>
      witnessInstructionsRev rows instructionsRev

private theorem witnessInstructionsRev_eq (rows : List CompiledRow)
    (instructionsRev : List WitnessInstruction) :
    witnessInstructionsRev rows instructionsRev =
      instructionsRev.reverse ++ witnessInstructions rows := by
  induction rows generalizing instructionsRev with
  | nil => simp [witnessInstructionsRev, witnessInstructions]
  | cons row rows inductionHypothesis =>
      cases row with
      | witness instruction =>
          simp only [witnessInstructionsRev, witnessInstructions]
          rw [inductionHypothesis]
          simp [List.reverse_cons, List.append_assoc]
      | assertion assertion =>
          simp only [witnessInstructionsRev, witnessInstructions]
          exact inductionHypothesis instructionsRev

/-- Stack-safe executable witness-instruction classifier. -/
def witnessInstructionsTR (rows : List CompiledRow) :
    List WitnessInstruction :=
  witnessInstructionsRev rows []

theorem witnessInstructionsTR_eq (rows : List CompiledRow) :
    witnessInstructionsTR rows = witnessInstructions rows := by
  rw [witnessInstructionsTR, witnessInstructionsRev_eq]
  rfl

private def assertionRowsRev :
    List CompiledRow → List SparseRow → List SparseRow
  | [], assertionsRev => assertionsRev.reverse
  | .witness _ :: rows, assertionsRev =>
      assertionRowsRev rows assertionsRev
  | .assertion assertion :: rows, assertionsRev =>
      assertionRowsRev rows (assertion :: assertionsRev)

private theorem assertionRowsRev_eq (rows : List CompiledRow)
    (assertionsRev : List SparseRow) :
    assertionRowsRev rows assertionsRev =
      assertionsRev.reverse ++ assertionRows rows := by
  induction rows generalizing assertionsRev with
  | nil => simp [assertionRowsRev, assertionRows]
  | cons row rows inductionHypothesis =>
      cases row with
      | witness instruction =>
          simp only [assertionRowsRev, assertionRows]
          exact inductionHypothesis assertionsRev
      | assertion assertion =>
          simp only [assertionRowsRev, assertionRows]
          rw [inductionHypothesis]
          simp [List.reverse_cons, List.append_assoc]

/-- Stack-safe executable assertion-row classifier. -/
def assertionRowsTR (rows : List CompiledRow) : List SparseRow :=
  assertionRowsRev rows []

theorem assertionRowsTR_eq (rows : List CompiledRow) :
    assertionRowsTR rows = assertionRows rows := by
  rw [assertionRowsTR, assertionRowsRev_eq]
  rfl

theorem witnessInstructions_append (left right : List CompiledRow) :
    witnessInstructions (left ++ right) =
      witnessInstructions left ++ witnessInstructions right := by
  induction left with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      cases row <;> simp [witnessInstructions, inductionHypothesis]

theorem assertionRows_append (left right : List CompiledRow) :
    assertionRows (left ++ right) =
      assertionRows left ++ assertionRows right := by
  induction left with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      cases row <;> simp [assertionRows, inductionHypothesis]

theorem witnessInstructionsTR_append (left right : List CompiledRow) :
    witnessInstructionsTR (left ++ right) =
      witnessInstructionsTR left ++ witnessInstructionsTR right := by
  rw [witnessInstructionsTR_eq, witnessInstructions_append,
    witnessInstructionsTR_eq, witnessInstructionsTR_eq]

theorem assertionRowsTR_append (left right : List CompiledRow) :
    assertionRowsTR (left ++ right) =
      assertionRowsTR left ++ assertionRowsTR right := by
  rw [assertionRowsTR_eq, assertionRows_append,
    assertionRowsTR_eq, assertionRowsTR_eq]

private def classifyRowsRev :
    List CompiledRow → List WitnessInstruction → List SparseRow →
      List WitnessInstruction × List SparseRow
  | [], instructionsRev, assertionsRev =>
      (instructionsRev.reverse, assertionsRev.reverse)
  | .witness instruction :: rows, instructionsRev, assertionsRev =>
      classifyRowsRev rows (instruction :: instructionsRev) assertionsRev
  | .assertion assertion :: rows, instructionsRev, assertionsRev =>
      classifyRowsRev rows instructionsRev (assertion :: assertionsRev)

private theorem classifyRowsRev_eq (rows : List CompiledRow)
    (instructionsRev : List WitnessInstruction)
    (assertionsRev : List SparseRow) :
    classifyRowsRev rows instructionsRev assertionsRev =
      (instructionsRev.reverse ++ witnessInstructions rows,
        assertionsRev.reverse ++ assertionRows rows) := by
  induction rows generalizing instructionsRev assertionsRev with
  | nil => simp [classifyRowsRev, witnessInstructions, assertionRows]
  | cons row rows inductionHypothesis =>
      cases row with
      | witness instruction =>
          simp only [classifyRowsRev, witnessInstructions, assertionRows]
          rw [inductionHypothesis]
          simp [List.reverse_cons, List.append_assoc]
      | assertion assertion =>
          simp only [classifyRowsRev, witnessInstructions, assertionRows]
          rw [inductionHypothesis]
          simp [List.reverse_cons, List.append_assoc]

/-- Classify one compiled-row block in one stack-safe pass. -/
def classifyRowsTR (rows : List CompiledRow) :
    List WitnessInstruction × List SparseRow :=
  classifyRowsRev rows [] []

theorem classifyRowsTR_eq (rows : List CompiledRow) :
    classifyRowsTR rows =
      (witnessInstructionsTR rows, assertionRowsTR rows) := by
  rw [classifyRowsTR, classifyRowsRev_eq, witnessInstructionsTR_eq,
    assertionRowsTR_eq]
  rfl

/-- Compact classification preserves every ordinary physical row exactly
once. -/
theorem witnessInstructions_length_add_assertionRows_length
    (rows : List CompiledRow) :
    (witnessInstructions rows).length + (assertionRows rows).length =
      rows.length := by
  induction rows with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      cases row <;>
        simp only [witnessInstructions, assertionRows, List.length_cons]
      all_goals omega

/-- The stack-safe classifiers also partition every compiled row exactly
once. This form lets large executable packets keep their row lists opaque. -/
theorem witnessInstructionsTR_length_add_assertionRowsTR_length
    (rows : List CompiledRow) :
    (witnessInstructionsTR rows).length +
      (assertionRowsTR rows).length = rows.length := by
  rw [witnessInstructionsTR_eq, assertionRowsTR_eq,
    witnessInstructions_length_add_assertionRows_length]

theorem witnessInstructions_member
    (rows : List CompiledRow) (instruction : WitnessInstruction) :
    instruction ∈ witnessInstructions rows ↔
      CompiledRow.witness instruction ∈ rows := by
  induction rows with
  | nil => simp [witnessInstructions]
  | cons row rows inductionHypothesis =>
      cases row <;>
        simp [witnessInstructions, inductionHypothesis]

theorem assertionRows_member (rows : List CompiledRow) (row : SparseRow) :
    row ∈ assertionRows rows ↔ CompiledRow.assertion row ∈ rows := by
  induction rows with
  | nil => simp [assertionRows]
  | cons current rows inductionHypothesis =>
      cases current <;>
        simp [assertionRows, inductionHypothesis]

private theorem rowsHold_cons (env : Env) (head : R1CS.Row)
    (tail : List R1CS.Row) :
    R1CS.RowsHold env (head :: tail) ↔
      head.Holds env ∧ R1CS.RowsHold env tail := by
  constructor
  · intro holds
    exact ⟨holds head (by simp), fun row member => holds row (by simp [member])⟩
  · rintro ⟨headHolds, tailHolds⟩ row member
    rcases List.mem_cons.mp member with rfl | member
    · exact headHolds
    · exact tailHolds row member

/-- Splitting ordinary rows into witness instructions and assertions preserves
their complete R1CS satisfaction predicate. -/
theorem compiledRows_hold_iff (rows : List CompiledRow) (env : Env) :
    R1CS.RowsHold env (rows.map CompiledRow.toR1CS) ↔
      (∀ instruction ∈ witnessInstructions rows, instruction.Holds env) ∧
      ∀ assertion ∈ assertionRows rows, assertion.Holds env := by
  induction rows with
  | nil => simp [R1CS.RowsHold, witnessInstructions, assertionRows]
  | cons row rows inductionHypothesis =>
      cases row with
      | witness instruction =>
          simp only [List.map_cons, witnessInstructions, assertionRows,
            CompiledRow.toR1CS]
          rw [rowsHold_cons, witnessInstruction_toR1CS_holds,
            inductionHypothesis]
          simp only [List.forall_mem_cons]
          tauto
      | assertion assertion =>
          simp only [List.map_cons, witnessInstructions, assertionRows,
            CompiledRow.toR1CS]
          rw [rowsHold_cons, sparseRow_holds, inductionHypothesis]
          simp only [List.forall_mem_cons]
          tauto

theorem compileRowsFrom_toR1CS (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRowsFrom freshStart rowStart rows).map CompiledRow.toR1CS = rows := by
  induction rows generalizing rowStart with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      simp [compileRowsFrom, compileRow_toR1CS,
        inductionHypothesis (rowStart := rowStart + 1)]

theorem compileRows_toR1CS (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRows freshStart rowStart rows).map CompiledRow.toR1CS = rows := by
  exact compileRowsFrom_toR1CS freshStart rowStart rows

theorem compileRowsTR_toR1CS (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRowsTR freshStart rowStart rows).map CompiledRow.toR1CS = rows := by
  rw [compileRowsTR_eq]
  exact compileRows_toR1CS freshStart rowStart rows

@[simp] theorem compileRows_length (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRows freshStart rowStart rows).length = rows.length := by
  unfold compileRows
  induction rows generalizing rowStart with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      simp [compileRowsFrom, inductionHypothesis]

@[simp] theorem compileRowsTR_length (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRowsTR freshStart rowStart rows).length = rows.length := by
  rw [compileRowsTR_eq, compileRows_length]

/-- Compiled rows own one exact contiguous physical row interval. -/
theorem compileRows_rowIndices (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRows freshStart rowStart rows).map CompiledRow.rowIndex =
      List.range' rowStart rows.length := by
  unfold compileRows
  induction rows generalizing rowStart with
  | nil => rfl
  | cons row rows inductionHypothesis =>
      simp [compileRowsFrom, inductionHypothesis, List.range'_succ]

/-- The stack-safe executable compiler preserves the same exact interval. -/
theorem compileRowsTR_rowIndices (freshStart rowStart : Nat)
    (rows : List R1CS.Row) :
    (compileRowsTR freshStart rowStart rows).map CompiledRow.rowIndex =
      List.range' rowStart rows.length := by
  rw [compileRowsTR_eq]
  exact compileRows_rowIndices freshStart rowStart rows

end NightstreamFPrime.Export.Stage1.Rows
