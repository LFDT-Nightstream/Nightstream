import Nightstream.Implementation.R1CS.Program

/-!
Contract: certifying execution semantics for production R1CS programs that
mix deterministic SSA definitions with verifier assertions.

`Program` proves deterministic definition blocks. Real verifier circuits also
contain rows that reject bad inputs: bit checks, public pins, transcript
equalities, and algebraic verifier equations. `Instruction` preserves the
exact Rust row order while classifying each row as either:

- `define`: a fresh SSA output computed from already-known columns; or
- `check`: an assertion over input or derived columns.

The two main theorems are deliberately symmetric:

- exact-row satisfaction fixes every derived column and makes every extracted
  check true under the executable interpreter;
- if those checks are true, the interpreter constructs a satisfying witness
  for the exact rows.

No assertion is converted into a definition. In particular, the exporter may
not hide a prover-controlled proof value by solving an accepting row for it.
-/

namespace Nightstream.Implementation.R1CS.CheckedProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

inductive Instruction where
  | define (definition : Definition)
  | check (row : Row)
deriving DecidableEq, Repr

def Instruction.row : Instruction → Row
  | .define definition => definition.builderRow
  | .check row => row

/-- Definition projection. Lean's compiler replaces `filterMap` with its
tail-recursive implementation, which is essential for 10k+ row artifacts. -/
def definitions (instructions : List Instruction) : List Definition :=
  instructions.filterMap fun instruction =>
    match instruction with
    | .define definition => some definition
    | .check _ => none

/-- Assertion projection, likewise compiled through `filterMapTR`. -/
def checks (instructions : List Instruction) : List Row :=
  instructions.filterMap fun instruction =>
    match instruction with
    | .define _ => none
    | .check row => some row

def rows (instructions : List Instruction) : List Row :=
  instructions.map Instruction.row

def interpret (state : Nat → Nat) (instructions : List Instruction) : Nat → Nat :=
  run state (definitions instructions)

def rowRefs (row : Row) : List Nat :=
  row.a.map Prod.fst ++ row.b.map Prod.fst ++ row.c.map Prod.fst

def ChecksReference (known : List Nat) (instructions : List Instruction) : Prop :=
  ∀ row ∈ checks instructions, ∀ column ∈ rowRefs row, column ∈ known

instance (known : List Nat) (instructions : List Instruction) :
    Decidable (ChecksReference known instructions) := by
  unfold ChecksReference
  infer_instance

def ChecksHold (state : Nat → Nat) (instructions : List Instruction) : Prop :=
  Satisfies (checks instructions) (interpret state instructions)

instance (state : Nat → Nat) (instructions : List Instruction) :
    Decidable (ChecksHold state instructions) := by
  unfold ChecksHold
  infer_instance

/-- Executable compiler/verifier result.  Definitions are interpreted first,
then every retained assertion is checked on that output.  A successful result
therefore packages the exact operational premise used by witness synthesis,
without exposing `ChecksHold` as a caller-supplied proof field. -/
def execute? (state : Nat → Nat)
    (instructions : List Instruction) : Option (Nat → Nat) :=
  if ChecksHold state instructions then
    some (interpret state instructions)
  else
    none

theorem execute?_eq_some_iff
    (state assignment : Nat → Nat) (instructions : List Instruction) :
    execute? state instructions = some assignment ↔
      ChecksHold state instructions ∧
        interpret state instructions = assignment := by
  unfold execute?
  by_cases checks : ChecksHold state instructions
  · simp [checks]
  · simp [checks]

private theorem definition_row_mem_rows
    {definition : Definition} {instructions : List Instruction}
    (member : definition ∈ definitions instructions) :
    definition.builderRow ∈ rows instructions := by
  rcases List.mem_filterMap.mp member with ⟨instruction, inProgram, mapped⟩
  cases instruction with
  | define current =>
      simp only at mapped
      cases mapped
      exact List.mem_map.mpr ⟨.define definition, inProgram, rfl⟩
  | check row => simp at mapped

private theorem check_row_mem_rows
    {row : Row} {instructions : List Instruction}
    (member : row ∈ checks instructions) : row ∈ rows instructions := by
  rcases List.mem_filterMap.mp member with ⟨instruction, inProgram, mapped⟩
  cases instruction with
  | define definition => simp at mapped
  | check current =>
      simp only at mapped
      cases mapped
      exact List.mem_map.mpr ⟨.check row, inProgram, rfl⟩

private theorem define_mem_definitions
    {definition : Definition} {instructions : List Instruction}
    (member : Instruction.define definition ∈ instructions) :
    definition ∈ definitions instructions := by
  apply List.mem_filterMap.mpr
  exact ⟨.define definition, member, rfl⟩

private theorem check_mem_checks
    {row : Row} {instructions : List Instruction}
    (member : Instruction.check row ∈ instructions) :
    row ∈ checks instructions := by
  apply List.mem_filterMap.mpr
  exact ⟨.check row, member, rfl⟩

private theorem lcEval_agree {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (terms : List (Nat × Nat))
    (references : ∀ term ∈ terms, term.1 ∈ known) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgree : ∀ initial,
      terms.foldl (fun acc term => acc + term.2 * left term.1) initial =
        terms.foldl (fun acc term => acc + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head.1 (references head (by simp))]
        apply inductionHypothesis
        intro term member
        exact references term (by simp [member])
  rw [foldAgree 0]

theorem rowHolds_agree {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (row : Row)
    (references : ∀ column ∈ rowRefs row, column ∈ known) :
    RowHolds left row ↔ RowHolds right row := by
  have aAgree : lcEval left row.a = lcEval right row.a := by
    apply lcEval_agree agreement
    intro term member
    apply references term.1
    unfold rowRefs
    apply List.mem_append_left
    apply List.mem_append_left
    exact List.mem_map.mpr ⟨term, member, rfl⟩
  have bAgree : lcEval left row.b = lcEval right row.b := by
    apply lcEval_agree agreement
    intro term member
    apply references term.1
    unfold rowRefs
    apply List.mem_append_left
    apply List.mem_append_right
    exact List.mem_map.mpr ⟨term, member, rfl⟩
  have cAgree : lcEval left row.c = lcEval right row.c := by
    apply lcEval_agree agreement
    intro term member
    apply references term.1
    unfold rowRefs
    apply List.mem_append_right
    exact List.mem_map.mpr ⟨term, member, rfl⟩
  simp only [RowHolds, aAgree, bAgree, cAgree]

structure SoundResult
    (inputColumns : List Nat)
    (instructions : List Instruction)
    (state assignment : Nat → Nat) : Prop where
  agreement :
    AgreeOn (interpret state instructions) assignment
      (knownAfter inputColumns (definitions instructions))
  checksHold : ChecksHold state instructions

/-- Exact mixed-program row satisfaction forces every deterministic
definition equation.  This is the semantic bridge used by artifact-specific
proofs that interpret a generated trace directly. -/
theorem definitionsHold_of_satisfies
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies (rows instructions) assignment) :
    ∀ definition ∈ definitions instructions,
      definition.Holds assignment := by
  apply builderDefinitions_sound assignmentCanonical constantOne
    canonicalDefinitions
  intro row member
  rcases List.mem_map.mp member with
    ⟨definition, definitionMember, rowEqual⟩
  subst row
  exact satisfies definition.builderRow
    (definition_row_mem_rows definitionMember)

/-- Every classified assertion is literally one of the exact mixed-program
rows, so exact program satisfaction also satisfies the assertion projection. -/
theorem checksSatisfy_of_satisfies
    {instructions : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows instructions) assignment) :
    Satisfies (checks instructions) assignment := by
  intro row member
  exact satisfies row (check_row_mem_rows member)

/-- `CIR-SOUND` compiler rule for a mixed exact-row program.

Every premise except canonical field representation and constant-one is a
structural certificate over the emitted instruction list. The conclusion is
computed from exact-row satisfaction; it is never carried in a certificate. -/
theorem sound
    {inputColumns : List Nat}
    {instructions : List Instruction}
    {state assignment : Nat → Nat}
    (wellFormed : WellFormed inputColumns (definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (checksReference :
      ChecksReference
        (knownAfter inputColumns (definitions instructions)) instructions)
    (initialAgreement : AgreeOn state assignment inputColumns)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies (rows instructions) assignment) :
    SoundResult inputColumns instructions state assignment := by
  have definitionSatisfaction :
      Satisfies ((definitions instructions).map Definition.builderRow)
        assignment := by
    intro row member
    rcases List.mem_map.mp member with ⟨definition, definitionMember, rowEqual⟩
    subst row
    exact satisfies definition.builderRow
      (definition_row_mem_rows definitionMember)
  have agreement := run_agrees_of_builder_satisfies wellFormed
    initialAgreement assignmentCanonical constantOne canonicalDefinitions
    definitionSatisfaction
  refine ⟨agreement, ?_⟩
  intro row member
  apply (rowHolds_agree agreement row
    (checksReference row member)).mpr
  exact satisfies row (check_row_mem_rows member)

/-- `CIR-COMPLETE` compiler rule for a mixed exact-row program.

The semantic validity premise is precisely `ChecksHold`: all verifier
assertions evaluate to true after deterministic execution. The theorem then
constructs a satisfying assignment for every exact emitted row. -/
theorem complete
    {inputColumns : List Nat}
    {instructions : List Instruction}
    {state : Nat → Nat}
    (wellFormed : WellFormed inputColumns (definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (constantOneColumn : 0 ∈ inputColumns)
    (constantOne : state 0 = 1)
    (checksHold : ChecksHold state instructions) :
    Satisfies (rows instructions) (interpret state instructions) := by
  have definitionsHold := run_satisfies_builder_rows wellFormed stateCanonical
    constantOneColumn constantOne canonicalDefinitions
  intro row member
  rcases List.mem_map.mp member with ⟨instruction, instructionMember, rowEqual⟩
  subst row
  cases instruction with
  | define definition =>
      apply definitionsHold definition.builderRow
      apply List.mem_map.mpr
      exact ⟨definition, define_mem_definitions instructionMember, rfl⟩
  | check checkRow =>
      exact checksHold checkRow (check_mem_checks instructionMember)

/-- A successful executable compiler run constructs a satisfying assignment
for the exact checked-program rows. -/
theorem complete_of_execute
    {inputColumns : List Nat}
    {instructions : List Instruction}
    {state assignment : Nat → Nat}
    (wellFormed : WellFormed inputColumns (definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (constantOneColumn : 0 ∈ inputColumns)
    (constantOne : state 0 = 1)
    (executed : execute? state instructions = some assignment) :
    Satisfies (rows instructions) assignment := by
  rcases (execute?_eq_some_iff state assignment instructions).mp executed with
    ⟨checks, output⟩
  have compiled := complete wellFormed canonicalDefinitions stateCanonical
    constantOneColumn constantOne checks
  rw [output] at compiled
  exact compiled

/-- Source-level witness returned by one successful checked-program run. -/
structure ExecutionWitness
    (instructions : List Instruction) (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  executed : execute? source instructions = some assignment

theorem ExecutionWitness.compiles
    {inputColumns : List Nat}
    {instructions : List Instruction}
    {assignment : Nat → Nat}
    (witness : ExecutionWitness instructions assignment)
    (wellFormed : WellFormed inputColumns (definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (constantOneColumn : 0 ∈ inputColumns) :
    Satisfies (rows instructions) assignment :=
  complete_of_execute wellFormed canonicalDefinitions witness.sourceCanonical
    constantOneColumn witness.sourceOne witness.executed

/-- Same-assignment semantics for an exact mixed program.  This is useful for
owners assembled from several ordinary instruction segments and compact
compiler blocks: it records the SSA equations and verifier assertions that
the assignment itself satisfies, without hiding either class as the other. -/
structure AssignmentHolds
    (instructions : List Instruction) (assignment : Nat → Nat) : Prop where
  definitions : ∀ definition ∈ definitions instructions,
    Definition.Holds assignment definition
  checks : Satisfies (checks instructions) assignment

/-- Executable independent semantics for a normalized mixed program.  This
checks the extracted definitions and assertions directly; it does not call the
R1CS owner-satisfaction predicate. -/
def assignmentCheck
    (instructions : List Instruction) (assignment : Nat → Nat) : Bool :=
  ((definitions instructions).all fun definition =>
      decide (Definition.Holds assignment definition)) &&
    ((checks instructions).all fun row => decide (RowHolds assignment row))

theorem assignmentCheck_eq_true_iff
    (instructions : List Instruction) (assignment : Nat → Nat) :
    assignmentCheck instructions assignment = true ↔
      AssignmentHolds instructions assignment := by
  simp only [assignmentCheck, Bool.and_eq_true, List.all_eq_true,
    decide_eq_true_eq]
  constructor
  · rintro ⟨definitionHolds, checksHold⟩
    exact ⟨definitionHolds, checksHold⟩
  · intro holds
    exact ⟨holds.definitions, holds.checks⟩

/-- Exact rows derive their same-assignment executable semantics. -/
theorem assignmentHolds_sound
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies (rows instructions) assignment) :
    AssignmentHolds instructions assignment where
  definitions := definitionsHold_of_satisfies canonicalDefinitions
    assignmentCanonical constantOne satisfies
  checks := checksSatisfy_of_satisfies satisfies

/-- The same-assignment semantics are sufficient for every exact emitted
row.  Unlike `complete`, this theorem does not construct a fresh interpreter
state; it validates a witness already produced by the compiler. -/
theorem assignmentHolds_complete
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : AssignmentHolds instructions assignment) :
    Satisfies (rows instructions) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with
    ⟨instruction, instructionMember, rowEqual⟩
  subst row
  cases instruction with
  | define definition =>
      exact builderDefinition_complete assignmentCanonical constantOne
        definition
        (canonicalDefinitions definition
          (define_mem_definitions instructionMember))
        (holds.definitions definition
          (define_mem_definitions instructionMember))
  | check checkRow =>
      exact holds.checks checkRow (check_mem_checks instructionMember)

theorem satisfies_iff_assignmentHolds
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonicalDefinitions :
      ∀ definition ∈ definitions instructions, definition.Canonical)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1) :
    Satisfies (rows instructions) assignment ↔
      AssignmentHolds instructions assignment :=
  ⟨assignmentHolds_sound canonicalDefinitions assignmentCanonical constantOne,
    assignmentHolds_complete canonicalDefinitions assignmentCanonical constantOne⟩

end Nightstream.Implementation.R1CS.CheckedProgram
