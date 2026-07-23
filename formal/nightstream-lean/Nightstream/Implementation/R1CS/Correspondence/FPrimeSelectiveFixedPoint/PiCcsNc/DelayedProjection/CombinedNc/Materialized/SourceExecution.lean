import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Provenance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.StageProgram

/-!
Executable source-program boundary for the fixed production combined-NC
relation.

Owns: the exact input-column boundary obtained by removing every materialized
definition output from the generated source-column registry, bounded
well-formedness certificates for the exact `StageProgram` definition stream,
and deterministic execution consequences of that well-formedness.

Does not own: source-row satisfaction, retained-check truth, selective-row
refinement, assignment authority, transcript scheduling, commitment binding,
costs, or permission to remove rows.

The executable certificates inspect proof-free `ExecutionStepShape` records.
Each record contains one output column and three Booleans: output membership in
the generated source registry, membership of all RHS references in that
registry, and strict precedence of all RHS references.  The 7,969 definitions
are split into 31 chunks of 250 records and one complete 219-record remainder.
No certificate evaluates `Program.WellFormed` or a proof-carrying value.

Assurance tier: artifact-checked for the fixed generated source registry and
the independently reconstructed `StageProgram` instruction stream once this
leaf validates.
-/

/-!
Emits constraints: none; this module proves execution of the existing source program.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_execution` | Derive the typed source-program trace from its materialized assignment. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Exact definition and input boundaries -/

private theorem definitions_append
    (left right : List Instruction) :
    definitions (left ++ right) = definitions left ++ definitions right := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head <;> simp [definitions, inductionHypothesis]

private theorem definitions_defines (values : List Definition) :
    definitions (values.map .define) = values := by
  simp [definitions, Function.comp_def]

private theorem definitions_checks (values : List Row) :
    definitions (values.map .check) = [] := by
  simp [definitions, Function.comp_def]

/-- Definition-only form of the four exact `StageProgram` phases.  Keeping
the already-oriented definition lists avoids repeatedly filtering the full
8,021-instruction stream inside bounded certificates. -/
def sourceDefinitions : List Definition :=
  StageProgram.paddingDefinitions ++ InitialProgram.definitions ++
    definitions StageProgram.roundInstructions ++ TerminalProgram.definitions

/-- The direct stage decomposition is exactly the definition projection of
the production mixed instruction stream. -/
theorem sourceDefinitions_eq_stageProjection :
    sourceDefinitions = definitions StageProgram.instructions := by
  simp only [sourceDefinitions, StageProgram.instructions,
    definitions_append, StageProgram.paddingInstructions,
    StageProgram.initialInstructions, StageProgram.terminalInstructions,
    definitions_defines, definitions_checks, List.append_nil]

def definitionOutputs : List Nat :=
  sourceDefinitions.map Definition.output

/-- Exact source-program seed boundary: every generated source column except
the outputs assigned by the oriented production definitions. -/
def inputColumns : List Nat :=
  Provenance.sourceColumns.filter fun column =>
    decide (column ∉ definitionOutputs)

theorem mem_inputColumns_iff (column : Nat) :
    column ∈ inputColumns ↔
      column ∈ Provenance.sourceColumns ∧ column ∉ definitionOutputs := by
  simp [inputColumns]

theorem sourceDefinition_count : sourceDefinitions.length = 7969 := by
  rw [sourceDefinitions_eq_stageProjection]
  exact StageProgram.definition_count

/-! ## Compact executable well-formedness checker -/

def PreviousOutputLt (previous : Option Nat) (output : Nat) : Prop :=
  match previous with
  | none => True
  | some column => column < output

private instance (previous : Option Nat) (output : Nat) :
    Decidable (PreviousOutputLt previous output) := by
  cases previous <;> unfold PreviousOutputLt <;> infer_instance

/-- Kernel relation recovered from the compact checker.  Besides adjacent
strict output order, it records exactly the source-registry and earlier-read
facts needed by the generic `Program.WellFormed` proof below. -/
inductive ExecutionValid : Option Nat → List Definition → Prop where
  | nil (previous) : ExecutionValid previous []
  | cons {previous : Option Nat} {head : Definition}
      {tail : List Definition}
      (previousLt : PreviousOutputLt previous head.output)
      (outputInSource : head.output ∈ Provenance.sourceColumns)
      (referencesInSource :
        ∀ column ∈ head.rhs.refs, column ∈ Provenance.sourceColumns)
      (referencesEarlier :
        ∀ column ∈ head.rhs.refs, column < head.output)
      (rest : ExecutionValid (some head.output) tail) :
      ExecutionValid previous (head :: tail)

/-- Four-word proof-free projection.  The three potentially large predicates
are reduced to Booleans before the native certificate is checked. -/
structure ExecutionStepShape where
  output : Nat
  outputInSource : Bool
  referencesInSource : Bool
  referencesEarlier : Bool
deriving DecidableEq, Repr

def executionStepShape (definition : Definition) : ExecutionStepShape :=
  { output := definition.output
    outputInSource := decide
      (definition.output ∈ Provenance.sourceColumns)
    referencesInSource := decide
      (∀ column ∈ definition.rhs.refs,
        column ∈ Provenance.sourceColumns)
    referencesEarlier := decide
      (∀ column ∈ definition.rhs.refs,
        column < definition.output) }

def executionShapeCheck :
    Option Nat → List ExecutionStepShape → Bool
  | _, [] => true
  | previous, shape :: rest =>
      decide (PreviousOutputLt previous shape.output) &&
        (shape.outputInSource &&
          (shape.referencesInSource &&
            (shape.referencesEarlier &&
              executionShapeCheck (some shape.output) rest)))

/-- Generic kernel bridge from compact Booleans to the original typed
definition stream. -/
theorem executionValid_of_shapeCheck_true :
    ∀ previous values,
      executionShapeCheck previous (values.map executionStepShape) = true →
        ExecutionValid previous values := by
  intro previous values
  induction values generalizing previous with
  | nil =>
      intro _
      exact .nil previous
  | cons head tail inductionHypothesis =>
      intro checked
      rw [List.map_cons, executionShapeCheck] at checked
      simp only [Bool.and_eq_true] at checked
      refine .cons ?_ ?_ ?_ ?_ ?_
      · exact of_decide_eq_true (by
          simpa only [executionStepShape] using checked.1)
      · exact of_decide_eq_true (by
          simpa only [executionStepShape] using checked.2.1)
      · exact of_decide_eq_true (by
          simpa only [executionStepShape] using checked.2.2.1)
      · exact of_decide_eq_true (by
          simpa only [executionStepShape] using checked.2.2.2.1)
      · exact inductionHypothesis _ checked.2.2.2.2

def previousAfter :
    Option Nat → List Definition → Option Nat
  | previous, [] => previous
  | _, head :: tail => previousAfter (some head.output) tail

theorem executionValid_append
    {previous : Option Nat} {left right : List Definition}
    (leftValid : ExecutionValid previous left)
    (rightValid : ExecutionValid (previousAfter previous left) right) :
    ExecutionValid previous (left ++ right) := by
  induction leftValid generalizing right with
  | nil =>
      simpa [previousAfter] using rightValid
  | cons previousLt outputInSource referencesInSource referencesEarlier
      rest inductionHypothesis =>
      apply ExecutionValid.cons previousLt outputInSource referencesInSource
        referencesEarlier
      apply inductionHypothesis
      simpa [previousAfter] using rightValid

/-! ## Bounded definition certificates -/

private abbrev definitionChunkSize : Nat := 250
private abbrev definitionChunkCount : Nat := 32

theorem definitionChunkSize_le_certificateLimit :
    definitionChunkSize ≤ 256 := by decide

def definitionChunk (index : Nat) : List Definition :=
  (sourceDefinitions.drop (index * definitionChunkSize)).take
    definitionChunkSize

def definitionShapeChunk (index : Nat) : List ExecutionStepShape :=
  (definitionChunk index).map executionStepShape

def definitionChunks : List (List Definition) :=
  List.ofFn fun index : Fin definitionChunkCount =>
    definitionChunk index.val

def definitionChunkRanges : List (Nat × Nat) :=
  List.ofFn fun index : Fin definitionChunkCount =>
    (index.val * definitionChunkSize,
      min ((index.val + 1) * definitionChunkSize)
        sourceDefinitions.length)

/-- Thirty-two proof-free interval pairs.  This certificate establishes that
the half-open chunk ranges are ordered and therefore non-overlapping. -/
theorem definitionChunkRanges_ordered :
    definitionChunkRanges.Pairwise (fun left right => left.2 ≤ right.1) := by
  native_decide

theorem definitionChunk_length_le (index : Nat) :
    (definitionChunk index).length ≤ definitionChunkSize := by
  exact List.length_take_le _ _

theorem nonfinalDefinitionChunk_length (index : Nat)
    (bound : index < 31) :
    (definitionChunk index).length = definitionChunkSize := by
  simp only [definitionChunk, List.length_take, List.length_drop,
    sourceDefinition_count, definitionChunkSize]
  omega

theorem finalDefinitionChunk_length :
    (definitionChunk 31).length = 219 := by
  simp [definitionChunk, sourceDefinition_count, definitionChunkSize]

theorem definitionShapeChunk_length (index : Nat) :
    (definitionShapeChunk index).length = (definitionChunk index).length := by
  simp [definitionShapeChunk]

private theorem fixedChunks_flatten
    {Alpha : Type} (values : List Alpha) (size : Nat) :
    ∀ count,
      values.length ≤ count * size →
        (List.ofFn fun index : Fin count =>
          (values.drop (index.val * size)).take size).flatten = values := by
  intro count
  induction count generalizing values with
  | zero =>
      intro covered
      have empty : values = [] := by
        apply List.eq_nil_of_length_eq_zero
        omega
      subst values
      rfl
  | succ count inductionHypothesis =>
      intro covered
      simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
        Nat.zero_mul, List.drop_zero]
      have tailCovered :
          (values.drop size).length ≤ count * size := by
        rw [Nat.succ_mul] at covered
        simp only [List.length_drop]
        omega
      have tail := inductionHypothesis (values.drop size) tailCovered
      have tail' :
          (List.ofFn fun index : Fin count =>
            (values.drop ((Fin.succ index).val * size)).take size).flatten =
              values.drop size := by
        simpa [Fin.val_succ, Nat.succ_mul, List.drop_drop,
          Nat.add_comm] using tail
      rw [tail']
      exact List.take_append_drop size values

/-- Kernel coverage of all 7,969 definitions.  The final chunk is the actual
remainder, so an oversized or truncated tail cannot be hidden. -/
theorem definitionChunks_flatten :
    definitionChunks.flatten = sourceDefinitions := by
  apply fixedChunks_flatten sourceDefinitions definitionChunkSize
    definitionChunkCount
  rw [sourceDefinition_count]
  decide

def previousForChunk : Nat → Option Nat
  | 0 => some 0
  | index + 1 =>
      previousAfter (previousForChunk index) (definitionChunk index)

theorem previousForChunk_succ (index : Nat) :
    previousForChunk (index + 1) =
      previousAfter (previousForChunk index) (definitionChunk index) := rfl

/-! Every theorem below evaluates at most 250 proof-free
`ExecutionStepShape` records.  Chunks zero through thirty contain exactly 250
records; chunk thirty-one is the complete 219-record remainder. -/

set_option maxRecDepth 100000 in
private theorem definitionChunk0Check :
    executionShapeCheck (previousForChunk 0) (definitionShapeChunk 0) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk1Check :
    executionShapeCheck (previousForChunk 1) (definitionShapeChunk 1) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk2Check :
    executionShapeCheck (previousForChunk 2) (definitionShapeChunk 2) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk3Check :
    executionShapeCheck (previousForChunk 3) (definitionShapeChunk 3) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk4Check :
    executionShapeCheck (previousForChunk 4) (definitionShapeChunk 4) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk5Check :
    executionShapeCheck (previousForChunk 5) (definitionShapeChunk 5) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk6Check :
    executionShapeCheck (previousForChunk 6) (definitionShapeChunk 6) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk7Check :
    executionShapeCheck (previousForChunk 7) (definitionShapeChunk 7) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk8Check :
    executionShapeCheck (previousForChunk 8) (definitionShapeChunk 8) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk9Check :
    executionShapeCheck (previousForChunk 9) (definitionShapeChunk 9) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk10Check :
    executionShapeCheck (previousForChunk 10) (definitionShapeChunk 10) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk11Check :
    executionShapeCheck (previousForChunk 11) (definitionShapeChunk 11) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk12Check :
    executionShapeCheck (previousForChunk 12) (definitionShapeChunk 12) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk13Check :
    executionShapeCheck (previousForChunk 13) (definitionShapeChunk 13) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk14Check :
    executionShapeCheck (previousForChunk 14) (definitionShapeChunk 14) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk15Check :
    executionShapeCheck (previousForChunk 15) (definitionShapeChunk 15) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk16Check :
    executionShapeCheck (previousForChunk 16) (definitionShapeChunk 16) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk17Check :
    executionShapeCheck (previousForChunk 17) (definitionShapeChunk 17) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk18Check :
    executionShapeCheck (previousForChunk 18) (definitionShapeChunk 18) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk19Check :
    executionShapeCheck (previousForChunk 19) (definitionShapeChunk 19) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk20Check :
    executionShapeCheck (previousForChunk 20) (definitionShapeChunk 20) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk21Check :
    executionShapeCheck (previousForChunk 21) (definitionShapeChunk 21) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk22Check :
    executionShapeCheck (previousForChunk 22) (definitionShapeChunk 22) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk23Check :
    executionShapeCheck (previousForChunk 23) (definitionShapeChunk 23) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk24Check :
    executionShapeCheck (previousForChunk 24) (definitionShapeChunk 24) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk25Check :
    executionShapeCheck (previousForChunk 25) (definitionShapeChunk 25) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk26Check :
    executionShapeCheck (previousForChunk 26) (definitionShapeChunk 26) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk27Check :
    executionShapeCheck (previousForChunk 27) (definitionShapeChunk 27) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk28Check :
    executionShapeCheck (previousForChunk 28) (definitionShapeChunk 28) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk29Check :
    executionShapeCheck (previousForChunk 29) (definitionShapeChunk 29) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk30Check :
    executionShapeCheck (previousForChunk 30) (definitionShapeChunk 30) = true := by native_decide
set_option maxRecDepth 100000 in
private theorem definitionChunk31Check :
    executionShapeCheck (previousForChunk 31) (definitionShapeChunk 31) = true := by native_decide

private theorem definitionChunkCheckAt (index : Nat)
    (bound : index < definitionChunkCount) :
    executionShapeCheck (previousForChunk index)
      (definitionShapeChunk index) = true := by
  change index < 32 at bound
  cases index with
  | zero => exact definitionChunk0Check
  | succ index =>
      cases index with
      | zero => exact definitionChunk1Check
      | succ index =>
          cases index with
          | zero => exact definitionChunk2Check
          | succ index =>
              cases index with
              | zero => exact definitionChunk3Check
              | succ index =>
                  cases index with
                  | zero => exact definitionChunk4Check
                  | succ index =>
                      cases index with
                      | zero => exact definitionChunk5Check
                      | succ index =>
                          cases index with
                          | zero => exact definitionChunk6Check
                          | succ index =>
                              cases index with
                              | zero => exact definitionChunk7Check
                              | succ index =>
                                  cases index with
                                  | zero => exact definitionChunk8Check
                                  | succ index =>
                                      cases index with
                                      | zero => exact definitionChunk9Check
                                      | succ index =>
                                          cases index with
                                          | zero => exact definitionChunk10Check
                                          | succ index =>
                                              cases index with
                                              | zero => exact definitionChunk11Check
                                              | succ index =>
                                                  cases index with
                                                  | zero => exact definitionChunk12Check
                                                  | succ index =>
                                                      cases index with
                                                      | zero => exact definitionChunk13Check
                                                      | succ index =>
                                                          cases index with
                                                          | zero => exact definitionChunk14Check
                                                          | succ index =>
                                                              cases index with
                                                              | zero => exact definitionChunk15Check
                                                              | succ index =>
                                                                  cases index with
                                                                  | zero => exact definitionChunk16Check
                                                                  | succ index =>
                                                                      cases index with
                                                                      | zero => exact definitionChunk17Check
                                                                      | succ index =>
                                                                          cases index with
                                                                          | zero => exact definitionChunk18Check
                                                                          | succ index =>
                                                                              cases index with
                                                                              | zero => exact definitionChunk19Check
                                                                              | succ index =>
                                                                                  cases index with
                                                                                  | zero => exact definitionChunk20Check
                                                                                  | succ index =>
                                                                                      cases index with
                                                                                      | zero => exact definitionChunk21Check
                                                                                      | succ index =>
                                                                                          cases index with
                                                                                          | zero => exact definitionChunk22Check
                                                                                          | succ index =>
                                                                                              cases index with
                                                                                              | zero => exact definitionChunk23Check
                                                                                              | succ index =>
                                                                                                  cases index with
                                                                                                  | zero => exact definitionChunk24Check
                                                                                                  | succ index =>
                                                                                                      cases index with
                                                                                                      | zero => exact definitionChunk25Check
                                                                                                      | succ index =>
                                                                                                          cases index with
                                                                                                          | zero => exact definitionChunk26Check
                                                                                                          | succ index =>
                                                                                                              cases index with
                                                                                                              | zero => exact definitionChunk27Check
                                                                                                              | succ index =>
                                                                                                                  cases index with
                                                                                                                  | zero => exact definitionChunk28Check
                                                                                                                  | succ index =>
                                                                                                                      cases index with
                                                                                                                      | zero => exact definitionChunk29Check
                                                                                                                      | succ index =>
                                                                                                                          cases index with
                                                                                                                          | zero => exact definitionChunk30Check
                                                                                                                          | succ index =>
                                                                                                                              cases index with
                                                                                                                              | zero => exact definitionChunk31Check
                                                                                                                              | succ index =>
                                                                                                                                  omega

private theorem definitionChunkValidAt (index : Nat)
    (bound : index < definitionChunkCount) :
    ExecutionValid (previousForChunk index) (definitionChunk index) := by
  apply executionValid_of_shapeCheck_true
  simpa only [definitionShapeChunk] using definitionChunkCheckAt index bound

private theorem prependDefinitionChunk
    (index : Nat) {tail : List Definition}
    (headValid :
      ExecutionValid (previousForChunk index) (definitionChunk index))
    (tailValid :
      ExecutionValid (previousForChunk (index + 1)) tail) :
    ExecutionValid (previousForChunk index) (definitionChunk index ++ tail) := by
  apply executionValid_append headValid
  simpa only [previousForChunk_succ] using tailValid

def joinedDefinitionChunks : List Definition :=
  definitionChunk 0 ++ (definitionChunk 1 ++ (definitionChunk 2 ++
    (definitionChunk 3 ++ (definitionChunk 4 ++ (definitionChunk 5 ++
    (definitionChunk 6 ++ (definitionChunk 7 ++ (definitionChunk 8 ++
    (definitionChunk 9 ++ (definitionChunk 10 ++ (definitionChunk 11 ++
    (definitionChunk 12 ++ (definitionChunk 13 ++ (definitionChunk 14 ++
    (definitionChunk 15 ++ (definitionChunk 16 ++ (definitionChunk 17 ++
    (definitionChunk 18 ++ (definitionChunk 19 ++ (definitionChunk 20 ++
    (definitionChunk 21 ++ (definitionChunk 22 ++ (definitionChunk 23 ++
    (definitionChunk 24 ++ (definitionChunk 25 ++ (definitionChunk 26 ++
    (definitionChunk 27 ++ (definitionChunk 28 ++ (definitionChunk 29 ++
    (definitionChunk 30 ++ definitionChunk 31))))))))))))))))))))))))))))))

private theorem joinedDefinitionChunks_eq_flatten :
    joinedDefinitionChunks = definitionChunks.flatten := by
  simp [joinedDefinitionChunks, definitionChunks, definitionChunkCount,
    List.ofFn_succ]

theorem joinedDefinitionChunks_exact :
    joinedDefinitionChunks = sourceDefinitions := by
  rw [joinedDefinitionChunks_eq_flatten, definitionChunks_flatten]

/-- All bounded shape certificates compose into strict SSA order and exact
source-registry coverage for the complete oriented definition stream. -/
theorem sourceDefinitionsExecutionValid :
    ExecutionValid (some 0) sourceDefinitions := by
  have valid31 := definitionChunkValidAt 31 (by decide)
  have valid30 := prependDefinitionChunk 30
    (definitionChunkValidAt 30 (by decide)) valid31
  have valid29 := prependDefinitionChunk 29
    (definitionChunkValidAt 29 (by decide)) valid30
  have valid28 := prependDefinitionChunk 28
    (definitionChunkValidAt 28 (by decide)) valid29
  have valid27 := prependDefinitionChunk 27
    (definitionChunkValidAt 27 (by decide)) valid28
  have valid26 := prependDefinitionChunk 26
    (definitionChunkValidAt 26 (by decide)) valid27
  have valid25 := prependDefinitionChunk 25
    (definitionChunkValidAt 25 (by decide)) valid26
  have valid24 := prependDefinitionChunk 24
    (definitionChunkValidAt 24 (by decide)) valid25
  have valid23 := prependDefinitionChunk 23
    (definitionChunkValidAt 23 (by decide)) valid24
  have valid22 := prependDefinitionChunk 22
    (definitionChunkValidAt 22 (by decide)) valid23
  have valid21 := prependDefinitionChunk 21
    (definitionChunkValidAt 21 (by decide)) valid22
  have valid20 := prependDefinitionChunk 20
    (definitionChunkValidAt 20 (by decide)) valid21
  have valid19 := prependDefinitionChunk 19
    (definitionChunkValidAt 19 (by decide)) valid20
  have valid18 := prependDefinitionChunk 18
    (definitionChunkValidAt 18 (by decide)) valid19
  have valid17 := prependDefinitionChunk 17
    (definitionChunkValidAt 17 (by decide)) valid18
  have valid16 := prependDefinitionChunk 16
    (definitionChunkValidAt 16 (by decide)) valid17
  have valid15 := prependDefinitionChunk 15
    (definitionChunkValidAt 15 (by decide)) valid16
  have valid14 := prependDefinitionChunk 14
    (definitionChunkValidAt 14 (by decide)) valid15
  have valid13 := prependDefinitionChunk 13
    (definitionChunkValidAt 13 (by decide)) valid14
  have valid12 := prependDefinitionChunk 12
    (definitionChunkValidAt 12 (by decide)) valid13
  have valid11 := prependDefinitionChunk 11
    (definitionChunkValidAt 11 (by decide)) valid12
  have valid10 := prependDefinitionChunk 10
    (definitionChunkValidAt 10 (by decide)) valid11
  have valid9 := prependDefinitionChunk 9
    (definitionChunkValidAt 9 (by decide)) valid10
  have valid8 := prependDefinitionChunk 8
    (definitionChunkValidAt 8 (by decide)) valid9
  have valid7 := prependDefinitionChunk 7
    (definitionChunkValidAt 7 (by decide)) valid8
  have valid6 := prependDefinitionChunk 6
    (definitionChunkValidAt 6 (by decide)) valid7
  have valid5 := prependDefinitionChunk 5
    (definitionChunkValidAt 5 (by decide)) valid6
  have valid4 := prependDefinitionChunk 4
    (definitionChunkValidAt 4 (by decide)) valid5
  have valid3 := prependDefinitionChunk 3
    (definitionChunkValidAt 3 (by decide)) valid4
  have valid2 := prependDefinitionChunk 2
    (definitionChunkValidAt 2 (by decide)) valid3
  have valid1 := prependDefinitionChunk 1
    (definitionChunkValidAt 1 (by decide)) valid2
  have valid0 := prependDefinitionChunk 0
    (definitionChunkValidAt 0 (by decide)) valid1
  change ExecutionValid (previousForChunk 0) joinedDefinitionChunks at valid0
  rw [joinedDefinitionChunks_exact] at valid0
  simpa only [previousForChunk] using valid0

/-! ## Kernel conversion to `Program.WellFormed` -/

private theorem lower_lt_all_outputs
    {lower : Nat} {values : List Definition}
    (valid : ExecutionValid (some lower) values) :
    ∀ definition ∈ values, lower < definition.output := by
  induction values generalizing lower with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource referencesInSource referencesEarlier rest =>
          intro definition member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · simpa [PreviousOutputLt] using previousLt
          · have lowerHead : lower < head.output := by
              simpa [PreviousOutputLt] using previousLt
            have headDefinition := inductionHypothesis rest definition member
            omega

private theorem wellFormed_of_executionValid
    {previous : Option Nat} {known : List Nat}
    {values : List Definition}
    (valid : ExecutionValid previous values)
    (knownCharacterization : ∀ column,
      column ∈ known ↔
        column ∈ Provenance.sourceColumns ∧
          column ∉ values.map Definition.output) :
    WellFormed known values := by
  induction valid generalizing known with
  | nil => exact .nil known
  | @cons previous head tail previousLt outputInSource referencesInSource
      referencesEarlier rest inductionHypothesis =>
      apply WellFormed.cons
      · intro column reference
        apply (knownCharacterization column).mpr
        refine ⟨referencesInSource column reference, ?_⟩
        intro outputMember
        simp only [List.map_cons, List.mem_cons] at outputMember
        rcases outputMember with outputEq | futureMember
        · have earlier := referencesEarlier column reference
          omega
        · rcases List.mem_map.mp futureMember with
            ⟨future, futureMember, outputEq⟩
          have earlier := referencesEarlier column reference
          have later : head.output < future.output :=
            lower_lt_all_outputs rest future futureMember
          omega
      · intro outputKnown
        have characterized :=
          (knownCharacterization head.output).mp outputKnown
        apply characterized.2
        simp
      · apply inductionHypothesis
        intro column
        constructor
        · intro member
          simp only [List.mem_cons] at member
          rcases member with equal | knownMember
          · subst column
            refine ⟨outputInSource, ?_⟩
            intro futureMember
            rcases List.mem_map.mp futureMember with
              ⟨future, futureMember, outputEq⟩
            have later : head.output < future.output :=
              lower_lt_all_outputs rest future futureMember
            omega
          · have characterized :=
              (knownCharacterization column).mp knownMember
            refine ⟨characterized.1, ?_⟩
            intro futureMember
            apply characterized.2
            simp [futureMember]
        · rintro ⟨sourceMember, notFuture⟩
          by_cases equal : column = head.output
          · exact List.mem_cons.mpr (Or.inl equal)
          · apply List.mem_cons.mpr
            apply Or.inr
            apply (knownCharacterization column).mpr
            refine ⟨sourceMember, ?_⟩
            simp only [List.map_cons, List.mem_cons]
            intro outputMember
            rcases outputMember with headEqual | futureMember
            · exact equal headEqual
            · exact notFuture futureMember

/-- Exact SSA theorem required to execute the materialized production source
program.  No row or retained-check satisfaction premise occurs here. -/
theorem stageProgramWellFormed :
    WellFormed inputColumns (definitions StageProgram.instructions) := by
  rw [← sourceDefinitions_eq_stageProjection]
  apply wellFormed_of_executionValid sourceDefinitionsExecutionValid
  exact mem_inputColumns_iff

/-! The constant-one proof evaluates only the first generated source-column
chunk: exactly 128 proof-free `Nat` records. -/

set_option maxRecDepth 100000 in
private theorem zero_mem_sourceChunk0 :
    0 ∈ Provenance.SourceColumns.Chunk0.values := by
  native_decide

theorem zero_mem_sourceColumns : 0 ∈ Provenance.sourceColumns := by
  unfold Provenance.sourceColumns Provenance.SourceColumns.values
  simp only [List.mem_append, zero_mem_sourceChunk0, true_or]

theorem zero_not_mem_definitionOutputs : 0 ∉ definitionOutputs := by
  intro member
  rcases List.mem_map.mp member with
    ⟨definition, definitionMember, outputEq⟩
  have positive := lower_lt_all_outputs sourceDefinitionsExecutionValid
    definition definitionMember
  omega

theorem constantOne_mem_inputColumns : 0 ∈ inputColumns :=
  (mem_inputColumns_iff 0).mpr
    ⟨zero_mem_sourceColumns, zero_not_mem_definitionOutputs⟩

/-! ## Deterministic execution consequences -/

def reconstruct (seed : Nat → Nat) : Nat → Nat :=
  interpret seed StageProgram.instructions

theorem reconstruct_canonical
    {seed : Nat → Nat}
    (seedCanonical : ∀ column, seed column < goldilocksP) :
    ∀ column, reconstruct seed column < goldilocksP := by
  exact Program.run_canonical seedCanonical

theorem reconstruct_definitionsHold (seed : Nat → Nat) :
    ∀ definition ∈ definitions StageProgram.instructions,
      definition.Holds (reconstruct seed) := by
  exact Program.run_definitions_hold stageProgramWellFormed seed

theorem reconstruct_preserves_inputColumns (seed : Nat → Nat) :
    AgreeOn (reconstruct seed) seed inputColumns := by
  exact Program.run_preserves_known stageProgramWellFormed seed

theorem reconstruct_preserves_known
    {seed : Nat → Nat} {column : Nat}
    (known : column ∈ inputColumns) :
    reconstruct seed column = seed column :=
  reconstruct_preserves_inputColumns seed column known

theorem reconstruct_preserves_constantOne
    {seed : Nat → Nat}
    (constantOne : seed 0 = 1) :
    reconstruct seed 0 = 1 :=
  (reconstruct_preserves_known constantOne_mem_inputColumns).trans constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceExecution
