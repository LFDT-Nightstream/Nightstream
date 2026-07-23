import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceExecution

/-!
Bounded execution certificate for the generated combined-NC compiler-linear
program.

Owns: exact coverage of the 816 generated linear definitions, their concrete
two-phase SSA schedule, and the fact that `SourceAssignment.compilerAssignment`
satisfies every generated compiler definition.

Does not own: selected-row satisfaction, rewrite-chain semantics, source-row
satisfaction, transcript authority, raw-child authority, commitment binding,
costs, or permission to remove rows.

The generated order has one intentional reset.  Its first 68 rowless
definitions are strictly ordered over the retained source-column seed.  The
remaining 748 physical linear definitions are strictly ordered over that seed
plus the 68 rowless outputs.  Treating the complete list as globally
output-increasing would therefore be false.

Every native certificate evaluates a proof-free `(Nat, Bool, Bool, Bool)`
shape derived directly from one generated `LinearDefinitions` shard.  The
first direct 128-record shard is checked as its exact 68/60 phase split; the
other direct shards contain 128, 128, 128, 128, 128, and 48 records.  No
certificate takes or drops the global 816-record concatenation, and no
certificate evaluates `Program.WellFormed` or another proof-carrying value.

Assurance tier: artifact-checked for this fixed generated compiler-linear
program once the focused leaf validates.
-/

/-!
Emits constraints: none; this module proves semantics of compiler-emitted rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.compiler_execution` | Show that the decoded compiler program executes the generated rewrite and retained rows. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.CompilerExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

private def programDefinitions
    (values : List RawSourceDefinition) : List Definition :=
  values.map SourceAssignment.RawSourceDefinition.programDefinition

private theorem programDefinitions_append
    (left right : List RawSourceDefinition) :
    programDefinitions (left ++ right) =
      programDefinitions left ++ programDefinitions right := by
  exact List.map_append

/-! ## Exact generated decomposition -/

/-- The rowless compiler phase is the exact 68-record prefix of direct shard
zero, not a prefix of the global generated list. -/
def rowlessDefinitions : List Definition :=
  programDefinitions (Provenance.LinearDefinitions.Chunk0.values.take 68)

/-- The physical part of direct shard zero is its exact 60-record remainder. -/
def physicalChunk0Definitions : List Definition :=
  programDefinitions (Provenance.LinearDefinitions.Chunk0.values.drop 68)

def physicalChunk1Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk1.values

def physicalChunk2Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk2.values

def physicalChunk3Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk3.values

def physicalChunk4Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk4.values

def physicalChunk5Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk5.values

def physicalChunk6Definitions : List Definition :=
  programDefinitions Provenance.LinearDefinitions.Chunk6.values

def physicalDefinitions : List Definition :=
  physicalChunk0Definitions ++ (physicalChunk1Definitions ++
    (physicalChunk2Definitions ++ (physicalChunk3Definitions ++
      (physicalChunk4Definitions ++ (physicalChunk5Definitions ++
        physicalChunk6Definitions)))))

/-- Kernel composition of the seven direct generated shards.  In particular,
the final shard is the actual 48-record remainder. -/
theorem compilerDefinitionPhases_exact :
    rowlessDefinitions ++ physicalDefinitions =
      SourceAssignment.compilerDefinitions := by
  have shard0Exact :
      rowlessDefinitions ++ physicalChunk0Definitions =
        programDefinitions
          Provenance.LinearDefinitions.Chunk0.values := by
    unfold rowlessDefinitions physicalChunk0Definitions
    rw [← programDefinitions_append]
    exact congrArg programDefinitions
      (List.take_append_drop 68
        Provenance.LinearDefinitions.Chunk0.values)
  rw [physicalDefinitions, ← List.append_assoc, shard0Exact]
  simp only [physicalChunk1Definitions,
    physicalChunk2Definitions, physicalChunk3Definitions,
    physicalChunk4Definitions, physicalChunk5Definitions,
    physicalChunk6Definitions,
    SourceAssignment.compilerDefinitions, Provenance.linearDefinitions,
    Provenance.LinearDefinitions.values, programDefinitions, List.map_append,
    List.append_assoc]

/-! ## Compact independent-phase checker -/

/-- Fixed seed registry used by the rowless compiler phase.  Column zero is
included because a nonzero `RawLinearCombination.constant` is represented as
a term against the constant-one column. -/
def retainedColumns : List Nat :=
  0 :: Provenance.retainedSlots.map fun slot => slot.column

/-- The second phase may read the outputs established by the first phase. -/
def physicalInputColumns : List Nat :=
  knownAfter retainedColumns rowlessDefinitions

/-- One proof-free compiler-step projection: exact output, output membership
in the generated source registry, freshness from the fixed phase input, and
RHS membership in that input. -/
structure IndependentStepShape where
  output : Nat
  outputInSource : Bool
  outputFresh : Bool
  referencesKnown : Bool
deriving DecidableEq, Repr

private instance (previous : Option Nat) (output : Nat) :
    Decidable (SourceExecution.PreviousOutputLt previous output) := by
  cases previous <;>
    unfold SourceExecution.PreviousOutputLt <;>
    infer_instance

def independentStepShape (known : List Nat)
    (definition : Definition) : IndependentStepShape :=
  { output := definition.output
    outputInSource := decide
      (definition.output ∈ Provenance.sourceColumns)
    outputFresh := decide (definition.output ∉ known)
    referencesKnown := decide (ReferencesOnly known definition) }

def independentShapeCheck :
    Option Nat → List IndependentStepShape → Bool
  | _, [] => true
  | previous, shape :: rest =>
      decide (SourceExecution.PreviousOutputLt previous shape.output) &&
        (shape.outputInSource &&
          (shape.outputFresh &&
            (shape.referencesKnown &&
              independentShapeCheck (some shape.output) rest)))

def shapePreviousAfter :
    Option Nat → List IndependentStepShape → Option Nat
  | previous, [] => previous
  | _, head :: tail => shapePreviousAfter (some head.output) tail

/-- Typed relation recovered from the compact phase checker.  Every RHS is
already available in the fixed phase input; strict output order then supplies
single assignment when those outputs are added one at a time. -/
inductive IndependentValid (known : List Nat) :
    Option Nat → List Definition → Prop where
  | nil (previous) : IndependentValid known previous []
  | cons {previous : Option Nat} {head : Definition}
      {tail : List Definition}
      (previousLt :
        SourceExecution.PreviousOutputLt previous head.output)
      (outputInSource : head.output ∈ Provenance.sourceColumns)
      (outputFresh : head.output ∉ known)
      (referencesKnown : ReferencesOnly known head)
      (rest : IndependentValid known (some head.output) tail) :
      IndependentValid known previous (head :: tail)

theorem IndependentValid.referencesOnly
    {known : List Nat} {previous : Option Nat} {values : List Definition}
    (valid : IndependentValid known previous values) :
    ∀ definition ∈ values, ReferencesOnly known definition := by
  induction valid with
  | nil => simp
  | cons previousLt outputInSource outputFresh referencesKnown
      rest inductionHypothesis =>
      intro definition member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact referencesKnown
      · exact inductionHypothesis definition member

/-- Generic kernel bridge from a proof-free shape certificate to the typed
independent-phase relation. -/
theorem independentValid_of_shapeCheck_true :
    ∀ known previous values,
      independentShapeCheck previous
          (values.map (independentStepShape known)) = true →
        IndependentValid known previous values := by
  intro known previous values
  induction values generalizing previous with
  | nil =>
      intro _
      exact .nil previous
  | cons head tail inductionHypothesis =>
      intro checked
      rw [List.map_cons, independentShapeCheck] at checked
      simp only [Bool.and_eq_true] at checked
      refine .cons ?_ ?_ ?_ ?_ ?_
      · exact of_decide_eq_true (by
          simpa only [independentStepShape] using checked.1)
      · exact of_decide_eq_true (by
          simpa only [independentStepShape] using checked.2.1)
      · exact of_decide_eq_true (by
          simpa only [independentStepShape] using checked.2.2.1)
      · exact of_decide_eq_true (by
          simpa only [independentStepShape] using checked.2.2.2.1)
      · exact inductionHypothesis _ checked.2.2.2.2

theorem shapePreviousAfter_map (known : List Nat) :
    ∀ previous values,
      shapePreviousAfter previous
          (values.map (independentStepShape known)) =
        SourceExecution.previousAfter previous values := by
  intro previous values
  induction values generalizing previous with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, shapePreviousAfter,
        independentStepShape, SourceExecution.previousAfter]
      exact inductionHypothesis (some head.output)

theorem independentValid_append
    {known : List Nat} {previous : Option Nat}
    {left right : List Definition}
    (leftValid : IndependentValid known previous left)
    (rightValid : IndependentValid known
      (SourceExecution.previousAfter previous left) right) :
    IndependentValid known previous (left ++ right) := by
  induction leftValid generalizing right with
  | nil =>
      simpa [SourceExecution.previousAfter] using rightValid
  | cons previousLt outputInSource outputFresh referencesKnown
      rest inductionHypothesis =>
      apply IndependentValid.cons previousLt outputInSource outputFresh
        referencesKnown
      apply inductionHypothesis
      simpa [SourceExecution.previousAfter] using rightValid

/-! ## Direct-shard certificates

Each subject below is one direct generated linear-definition shard mapped to
proof-free shapes.  Cardinalities are 68 and 60 for the two exact parts of
direct shard zero, then 128, 128, 128, 128, 128, and 48. -/

def rowlessShapes : List IndependentStepShape :=
  rowlessDefinitions.map (independentStepShape retainedColumns)

def physicalChunk0Shapes : List IndependentStepShape :=
  physicalChunk0Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk1Shapes : List IndependentStepShape :=
  physicalChunk1Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk2Shapes : List IndependentStepShape :=
  physicalChunk2Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk3Shapes : List IndependentStepShape :=
  physicalChunk3Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk4Shapes : List IndependentStepShape :=
  physicalChunk4Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk5Shapes : List IndependentStepShape :=
  physicalChunk5Definitions.map
    (independentStepShape physicalInputColumns)

def physicalChunk6Shapes : List IndependentStepShape :=
  physicalChunk6Definitions.map
    (independentStepShape physicalInputColumns)

set_option maxRecDepth 100000 in
private theorem rowlessCertificate :
    independentShapeCheck (some 0) rowlessShapes = true ∧
      rowlessShapes.length = 68 ∧
      shapePreviousAfter (some 0) rowlessShapes = some 4012155 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk0Certificate :
    independentShapeCheck (some 0) physicalChunk0Shapes = true ∧
      physicalChunk0Shapes.length = 60 ∧
      shapePreviousAfter (some 0) physicalChunk0Shapes = some 63366 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk1Certificate :
    independentShapeCheck (some 63366) physicalChunk1Shapes = true ∧
      physicalChunk1Shapes.length = 128 ∧
      shapePreviousAfter (some 63366) physicalChunk1Shapes = some 80154 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk2Certificate :
    independentShapeCheck (some 80154) physicalChunk2Shapes = true ∧
      physicalChunk2Shapes.length = 128 ∧
      shapePreviousAfter (some 80154) physicalChunk2Shapes = some 3964312 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk3Certificate :
    independentShapeCheck (some 3964312) physicalChunk3Shapes = true ∧
      physicalChunk3Shapes.length = 128 ∧
      shapePreviousAfter (some 3964312) physicalChunk3Shapes = some 3975582 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk4Certificate :
    independentShapeCheck (some 3975582) physicalChunk4Shapes = true ∧
      physicalChunk4Shapes.length = 128 ∧
      shapePreviousAfter (some 3975582) physicalChunk4Shapes = some 4004862 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk5Certificate :
    independentShapeCheck (some 4004862) physicalChunk5Shapes = true ∧
      physicalChunk5Shapes.length = 128 ∧
      shapePreviousAfter (some 4004862) physicalChunk5Shapes = some 4081007 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk6Certificate :
    independentShapeCheck (some 4081007) physicalChunk6Shapes = true ∧
      physicalChunk6Shapes.length = 48 ∧
      shapePreviousAfter (some 4081007) physicalChunk6Shapes = some 4081325 := by
  native_decide

theorem rowlessDefinition_count : rowlessDefinitions.length = 68 := by
  simpa only [rowlessShapes, List.length_map] using rowlessCertificate.2.1

theorem physicalDefinition_count : physicalDefinitions.length = 748 := by
  have length0 : physicalChunk0Definitions.length = 60 := by
    simpa only [physicalChunk0Shapes, List.length_map] using
      physicalChunk0Certificate.2.1
  have length1 : physicalChunk1Definitions.length = 128 := by
    simpa only [physicalChunk1Shapes, List.length_map] using
      physicalChunk1Certificate.2.1
  have length2 : physicalChunk2Definitions.length = 128 := by
    simpa only [physicalChunk2Shapes, List.length_map] using
      physicalChunk2Certificate.2.1
  have length3 : physicalChunk3Definitions.length = 128 := by
    simpa only [physicalChunk3Shapes, List.length_map] using
      physicalChunk3Certificate.2.1
  have length4 : physicalChunk4Definitions.length = 128 := by
    simpa only [physicalChunk4Shapes, List.length_map] using
      physicalChunk4Certificate.2.1
  have length5 : physicalChunk5Definitions.length = 128 := by
    simpa only [physicalChunk5Shapes, List.length_map] using
      physicalChunk5Certificate.2.1
  have length6 : physicalChunk6Definitions.length = 48 := by
    simpa only [physicalChunk6Shapes, List.length_map] using
      physicalChunk6Certificate.2.1
  simp only [physicalDefinitions, List.length_append, length0, length1,
    length2, length3, length4, length5, length6]

theorem compilerDefinition_count :
    SourceAssignment.compilerDefinitions.length = 816 := by
  rw [← compilerDefinitionPhases_exact, List.length_append,
    rowlessDefinition_count, physicalDefinition_count]

/-! ## Kernel conversion and execution -/

private def PreviousOutputLe (previous : Option Nat) (column : Nat) : Prop :=
  match previous with
  | none => False
  | some lower => column ≤ lower

private theorem independentValid_wellFormed_aux
    {base : List Nat} {previous : Option Nat} {values : List Definition}
    (valid : IndependentValid base previous values) :
    ∀ known,
      (∀ column ∈ base, column ∈ known) →
      (∀ column ∈ known, column ∉ base →
        PreviousOutputLe previous column) →
      WellFormed known values := by
  induction valid with
  | nil previous =>
      intro known _ _
      exact .nil known
  | @cons previous head tail previousLt outputInSource outputFresh
      referencesKnown rest inductionHypothesis =>
      intro known baseSubset oldBound
      apply WellFormed.cons
      · intro column reference
        exact baseSubset column (referencesKnown column reference)
      · intro outputKnown
        by_cases outputInBase : head.output ∈ base
        · exact outputFresh outputInBase
        · have bounded := oldBound head.output outputKnown outputInBase
          cases previous with
          | none => simp [PreviousOutputLe] at bounded
          | some lower =>
              simp [SourceExecution.PreviousOutputLt] at previousLt
              simp [PreviousOutputLe] at bounded
              omega
      · apply inductionHypothesis (head.output :: known)
        · intro column inBase
          exact List.mem_cons_of_mem head.output (baseSubset column inBase)
        · intro column inNext notInBase
          simp only [List.mem_cons] at inNext
          rcases inNext with equal | inKnown
          · subst column
            simp [PreviousOutputLe]
          · have bounded := oldBound column inKnown notInBase
            cases previous with
            | none => simp [PreviousOutputLe] at bounded
            | some lower =>
                simp [SourceExecution.PreviousOutputLt] at previousLt
                simp [PreviousOutputLe] at bounded ⊢
                omega

theorem independentValid_wellFormed
    {base : List Nat} {previous : Option Nat} {values : List Definition}
    (valid : IndependentValid base previous values) :
    WellFormed base values := by
  apply independentValid_wellFormed_aux valid base
  · intro column member
    exact member
  · intro column member notMember
    exact False.elim (notMember member)

private theorem previousAfter_eq_of_shapePreviousAfter
    (known : List Nat) (previous : Option Nat) (values : List Definition)
    (next : Option Nat)
    (shapeEnd :
      shapePreviousAfter previous
          (values.map (independentStepShape known)) = next) :
    SourceExecution.previousAfter previous values = next := by
  rw [shapePreviousAfter_map known previous values] at shapeEnd
  exact shapeEnd

theorem rowlessDefinitionsWellFormed :
    WellFormed retainedColumns rowlessDefinitions := by
  apply independentValid_wellFormed
  apply independentValid_of_shapeCheck_true
  simpa only [rowlessShapes] using rowlessCertificate.1

private theorem physicalChunk0Valid :
    IndependentValid physicalInputColumns (some 0)
      physicalChunk0Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk0Shapes] using physicalChunk0Certificate.1

private theorem physicalChunk1Valid :
    IndependentValid physicalInputColumns (some 63366)
      physicalChunk1Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk1Shapes] using physicalChunk1Certificate.1

private theorem physicalChunk2Valid :
    IndependentValid physicalInputColumns (some 80154)
      physicalChunk2Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk2Shapes] using physicalChunk2Certificate.1

private theorem physicalChunk3Valid :
    IndependentValid physicalInputColumns (some 3964312)
      physicalChunk3Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk3Shapes] using physicalChunk3Certificate.1

private theorem physicalChunk4Valid :
    IndependentValid physicalInputColumns (some 3975582)
      physicalChunk4Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk4Shapes] using physicalChunk4Certificate.1

private theorem physicalChunk5Valid :
    IndependentValid physicalInputColumns (some 4004862)
      physicalChunk5Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk5Shapes] using physicalChunk5Certificate.1

private theorem physicalChunk6Valid :
    IndependentValid physicalInputColumns (some 4081007)
      physicalChunk6Definitions := by
  apply independentValid_of_shapeCheck_true
  simpa only [physicalChunk6Shapes] using physicalChunk6Certificate.1

private theorem physicalChunk0End :
    SourceExecution.previousAfter (some 0) physicalChunk0Definitions =
      some 63366 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk0Shapes] using physicalChunk0Certificate.2.2

private theorem physicalChunk1End :
    SourceExecution.previousAfter (some 63366) physicalChunk1Definitions =
      some 80154 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk1Shapes] using physicalChunk1Certificate.2.2

private theorem physicalChunk2End :
    SourceExecution.previousAfter (some 80154) physicalChunk2Definitions =
      some 3964312 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk2Shapes] using physicalChunk2Certificate.2.2

private theorem physicalChunk3End :
    SourceExecution.previousAfter (some 3964312) physicalChunk3Definitions =
      some 3975582 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk3Shapes] using physicalChunk3Certificate.2.2

private theorem physicalChunk4End :
    SourceExecution.previousAfter (some 3975582) physicalChunk4Definitions =
      some 4004862 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk4Shapes] using physicalChunk4Certificate.2.2

private theorem physicalChunk5End :
    SourceExecution.previousAfter (some 4004862) physicalChunk5Definitions =
      some 4081007 := by
  apply previousAfter_eq_of_shapePreviousAfter physicalInputColumns
  simpa only [physicalChunk5Shapes] using physicalChunk5Certificate.2.2

theorem physicalDefinitionsIndependentValid :
    IndependentValid physicalInputColumns (some 0) physicalDefinitions := by
  have valid5And6 :
      IndependentValid physicalInputColumns (some 4004862)
        (physicalChunk5Definitions ++ physicalChunk6Definitions) := by
    apply independentValid_append physicalChunk5Valid
    rw [physicalChunk5End]
    exact physicalChunk6Valid
  have valid4To6 :
      IndependentValid physicalInputColumns (some 3975582)
        (physicalChunk4Definitions ++
          (physicalChunk5Definitions ++ physicalChunk6Definitions)) := by
    apply independentValid_append physicalChunk4Valid
    rw [physicalChunk4End]
    exact valid5And6
  have valid3To6 :
      IndependentValid physicalInputColumns (some 3964312)
        (physicalChunk3Definitions ++ (physicalChunk4Definitions ++
          (physicalChunk5Definitions ++ physicalChunk6Definitions))) := by
    apply independentValid_append physicalChunk3Valid
    rw [physicalChunk3End]
    exact valid4To6
  have valid2To6 :
      IndependentValid physicalInputColumns (some 80154)
        (physicalChunk2Definitions ++ (physicalChunk3Definitions ++
          (physicalChunk4Definitions ++ (physicalChunk5Definitions ++
            physicalChunk6Definitions)))) := by
    apply independentValid_append physicalChunk2Valid
    rw [physicalChunk2End]
    exact valid3To6
  have valid1To6 :
      IndependentValid physicalInputColumns (some 63366)
        (physicalChunk1Definitions ++ (physicalChunk2Definitions ++
          (physicalChunk3Definitions ++ (physicalChunk4Definitions ++
            (physicalChunk5Definitions ++ physicalChunk6Definitions))))) := by
    apply independentValid_append physicalChunk1Valid
    rw [physicalChunk1End]
    exact valid2To6
  unfold physicalDefinitions
  apply independentValid_append physicalChunk0Valid
  rw [physicalChunk0End]
  exact valid1To6

theorem physicalDefinitionsWellFormed :
    WellFormed physicalInputColumns physicalDefinitions :=
  independentValid_wellFormed physicalDefinitionsIndependentValid

private theorem wellFormed_append
    {known : List Nat} {left right : List Definition}
    (leftWellFormed : WellFormed known left)
    (rightWellFormed : WellFormed (knownAfter known left) right) :
    WellFormed known (left ++ right) := by
  induction leftWellFormed generalizing right with
  | nil => simpa [knownAfter] using rightWellFormed
  | cons references fresh rest inductionHypothesis =>
      apply WellFormed.cons references fresh
      apply inductionHypothesis
      simpa [knownAfter] using rightWellFormed

/-- Exact well-formedness of the generated 816-definition compiler program.
The fixed input boundary is the constant-one column plus the generated
retained source-column registry. -/
theorem compilerProgramWellFormed :
    WellFormed retainedColumns SourceAssignment.compilerDefinitions := by
  rw [← compilerDefinitionPhases_exact]
  apply wellFormed_append rowlessDefinitionsWellFormed
  exact physicalDefinitionsWellFormed

/-- Executing the exact generated compiler program establishes every one of
its 816 linear definitions in the final compiler assignment. -/
theorem compilerAssignment_definitionsHold (assignment : Nat → Nat) :
    ∀ definition ∈ SourceAssignment.compilerDefinitions,
      definition.Holds (SourceAssignment.compilerAssignment assignment) := by
  exact run_definitions_hold compilerProgramWellFormed
    (SourceAssignment.retainedSeed assignment)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.CompilerExecution
