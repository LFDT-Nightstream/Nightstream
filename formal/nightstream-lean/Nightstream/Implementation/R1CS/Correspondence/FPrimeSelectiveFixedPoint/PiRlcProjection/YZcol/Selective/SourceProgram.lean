import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

/-!
Deterministic source-program reconstruction for the bounded selective
fixed-point `y_zcol` slice.

Owns: the exact source-definition program, its independently checked
SSA input boundary, and the canonical source assignment obtained by executing
those definitions after source-column decoding.

Does not own: compact-row satisfaction, rewrite-family semantics, final-check
satisfaction, producer authority, the projection security event, production
conformance, or permission to remove rows.

Emits constraints: no.

The seed may contain candidate values for source output columns. They are not
trusted: every output is excluded from `sourceKnownColumns` and overwritten by
`Program.run`. Only columns outside the exact source-definition output set are
preserved.

| Source-program leaf | Mathematical obligation | Authority class |
|---|---|---|
| SSA schedule | every definition reads known or earlier columns | checked |
| execution | recomputed assignment satisfies every definition | derived |
| boundary | non-output source columns are preserved exactly | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

private abbrev certificate :=
  Selective.Materialized.Checked.sourceArtifact.certificate

/-- Exact handwritten source-trace definitions, in physical source-row
execution order. -/
def sourceDefinitions : List Program.Definition :=
  certificate.definitions

def sourceOutputColumns : List Nat :=
  sourceDefinitions.map Program.Definition.output

/-- Complete focused source closure minus every column assigned by the source
program. This is an executable boundary check, not a stage-label census. -/
def sourceKnownColumns : List Nat :=
  Selective.Materialized.Checked.sourceColumns.filter fun column =>
    !(sourceOutputColumns.contains column)

set_option maxRecDepth 100000 in
theorem sourceProgramWellFormed :
    Program.WellFormed sourceKnownColumns sourceDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
theorem constantColumnKnown : 0 ∈ sourceKnownColumns := by
  native_decide

set_option maxRecDepth 100000 in
theorem exactDefinitionCount : sourceDefinitions.length = 5720 := by
  native_decide

/-- Source semantics reconstructed independently of the selectively emitted
rows. Candidate terminal values in `compilerAssignment` are overwritten. -/
def sourceAssignment (assignment : Nat → Nat) : Nat → Nat :=
  Program.run (SourceDecode.compilerAssignment assignment) sourceDefinitions

theorem sourceAssignmentCanonical (assignment : Nat → Nat) :
    ∀ column, sourceAssignment assignment column < goldilocksP := by
  exact Program.run_canonical (SourceDecode.compilerAssignmentCanonical assignment)

theorem sourceAssignmentDefinitionsHold (assignment : Nat → Nat) :
    ProjectionProgram.DefinitionsHold (sourceAssignment assignment)
      sourceDefinitions := by
  exact Program.run_definitions_hold sourceProgramWellFormed
    (SourceDecode.compilerAssignment assignment)

theorem sourceAssignmentPreservesKnown (assignment : Nat → Nat) :
    Program.AgreeOn (sourceAssignment assignment)
      (SourceDecode.compilerAssignment assignment) sourceKnownColumns := by
  exact Program.run_preserves_known sourceProgramWellFormed
    (SourceDecode.compilerAssignment assignment)

theorem sourceAssignmentConstantOne
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    sourceAssignment assignment 0 = 1 := by
  rw [sourceAssignmentPreservesKnown assignment 0 constantColumnKnown]
  exact SourceDecode.compilerAssignmentConstantOne constantOne

/-! ## Eliminated source-linear base definitions -/

private def baseC0Definition : Program.Definition :=
  { output := 8683829, rhs := .linear [(0, 1)] }

private def baseC1Definition : Program.Definition :=
  { output := 8683830, rhs := .linear [] }

/-- The only source-program outputs also reconstructed by the compiler's
linear-definition program are the two ladder-base coordinates. This exact
intersection prevents an eliminated source definition from silently escaping
the semantic bridge. -/
theorem compilerSourceOutputDefinitions_exact :
    SourceDecode.compilerDefinitions.filter (fun definition =>
      sourceOutputColumns.contains definition.output) =
      [baseC0Definition, baseC1Definition] := by
  set_option maxRecDepth 100000 in
    native_decide

set_option maxRecDepth 100000 in
private theorem baseC0_mem_compiler :
    baseC0Definition ∈ SourceDecode.compilerDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
private theorem baseC1_mem_compiler :
    baseC1Definition ∈ SourceDecode.compilerDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
private theorem baseC0_mem_source : baseC0Definition ∈ sourceDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
private theorem baseC1_mem_source : baseC1Definition ∈ sourceDefinitions := by
  native_decide

theorem compilerBaseC0
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    SourceDecode.compilerAssignment assignment 8683829 = 1 := by
  have holds := SourceDecode.compilerAssignmentDefinitionsHold assignment
    baseC0Definition baseC0_mem_compiler
  simpa [baseC0Definition, Program.Definition.Holds, Program.Rhs.eval,
    lcEval, SourceDecode.compilerAssignmentConstantOne constantOne] using holds

theorem compilerBaseC1 (assignment : Nat → Nat) :
    SourceDecode.compilerAssignment assignment 8683830 = 0 := by
  have holds := SourceDecode.compilerAssignmentDefinitionsHold assignment
    baseC1Definition baseC1_mem_compiler
  simpa [baseC1Definition, Program.Definition.Holds, Program.Rhs.eval,
    lcEval] using holds

theorem sourceBaseC0
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    sourceAssignment assignment 8683829 = 1 := by
  have holds := sourceAssignmentDefinitionsHold assignment
    baseC0Definition baseC0_mem_source
  simpa [baseC0Definition, Program.Definition.Holds, Program.Rhs.eval,
    lcEval, sourceAssignmentConstantOne constantOne] using holds

theorem sourceBaseC1 (assignment : Nat → Nat) :
    sourceAssignment assignment 8683830 = 0 := by
  have holds := sourceAssignmentDefinitionsHold assignment
    baseC1Definition baseC1_mem_source
  simpa [baseC1Definition, Program.Definition.Holds, Program.Rhs.eval,
    lcEval] using holds

/-- The eliminated ladder-base definitions agree without assuming decoded
equality or source-row satisfaction. -/
theorem ladderBaseAgrees
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    (sourceAssignment assignment 8683829 =
        SourceDecode.compilerAssignment assignment 8683829) ∧
      (sourceAssignment assignment 8683830 =
        SourceDecode.compilerAssignment assignment 8683830) := by
  rw [sourceBaseC0 constantOne, compilerBaseC0 constantOne,
    sourceBaseC1 assignment, compilerBaseC1 assignment]
  constructor <;> rfl

/-! ## Compiler/source execution agreement -/

private def compilerBoundaryColumns : List Nat :=
  [8683829, 8683830] ++ sourceKnownColumns

set_option maxRecDepth 100000 in
/-- Every input and output of the checked compiler-linear program is either a
source-program boundary column or one of the two independently reconciled
ladder bases. This is an exact column check, not a stage-label claim. -/
theorem compilerDefinitionColumnsKnown :
    ∀ definition ∈ SourceDecode.compilerDefinitions,
      definition.output ∈ compilerBoundaryColumns ∧
        ∀ column ∈ definition.rhs.refs,
          column ∈ compilerBoundaryColumns := by
  native_decide

private theorem sourceCompilerBoundaryAgrees
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    Program.AgreeOn (sourceAssignment assignment)
      (SourceDecode.compilerAssignment assignment)
      compilerBoundaryColumns := by
  intro column member
  simp only [compilerBoundaryColumns, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with base | known
  · rcases base with rfl | rfl
    · exact (ladderBaseAgrees constantOne).1
    · exact (ladderBaseAgrees constantOne).2
  · exact sourceAssignmentPreservesKnown assignment column known

private theorem lcEval_eq_of_agree
    {left right : Nat → Nat} {known : List Nat}
    (agreement : Program.AgreeOn left right known) :
    ∀ terms : List (Nat × Nat),
      (∀ term ∈ terms, term.1 ∈ known) →
        lcEval left terms = lcEval right terms := by
  intro terms references
  unfold lcEval
  have foldAgree : ∀ initial,
      terms.foldl (fun total term => total + term.2 * left term.1) initial =
        terms.foldl (fun total term => total + term.2 * right term.1)
          initial := by
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

private theorem rhsEval_eq_of_agree
    {left right : Nat → Nat} {known : List Nat}
    (agreement : Program.AgreeOn left right known)
    (rhs : Program.Rhs)
    (references : ∀ column ∈ rhs.refs, column ∈ known) :
    rhs.eval left = rhs.eval right := by
  cases rhs with
  | linear terms =>
      apply lcEval_eq_of_agree agreement terms
      intro term member
      apply references term.1
      exact List.mem_map.mpr ⟨term, member, rfl⟩
  | product lhs rhs =>
      simp only [Program.Rhs.eval]
      rw [lcEval_eq_of_agree agreement lhs (by
        intro term member
        apply references term.1
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]
      rw [lcEval_eq_of_agree agreement rhs (by
        intro term member
        apply references term.1
        apply List.mem_append_right
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]

/-- Executing the source program also satisfies every compiler-linear
definition used to decode compact ports. The proof derives this from exact
column provenance and both deterministic executions; it assumes no source row
or decoded equality. -/
theorem sourceAssignmentCompilerDefinitionsHold
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1) :
    ∀ definition ∈ SourceDecode.compilerDefinitions,
      definition.Holds (sourceAssignment assignment) := by
  intro definition member
  have columns := compilerDefinitionColumnsKnown definition member
  have agreement := sourceCompilerBoundaryAgrees constantOne
  have holds := SourceDecode.compilerAssignmentDefinitionsHold assignment
    definition member
  unfold Program.Definition.Holds at holds ⊢
  rw [agreement definition.output columns.1,
    rhsEval_eq_of_agree agreement definition.rhs columns.2]
  exact holds

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceProgram
