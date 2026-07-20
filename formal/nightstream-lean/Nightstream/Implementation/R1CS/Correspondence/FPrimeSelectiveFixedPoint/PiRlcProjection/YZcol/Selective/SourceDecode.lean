import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Schema
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm
import Std.Data.HashMap.Lemmas

/-!
Fail-closed source-column decoding for the bounded selective fixed-point
`y_zcol` projection artifact.

Owns: canonical source linear combinations, retained-slot bounds and radix,
the complete source-column partition, the straight-line compiler program, and
the canonical assignment reconstructed by that program.

Does not own: compact-row decoding or satisfaction, rewrite-step semantics,
trace-eliminated column values, selector truth, projection authority, security
events, or permission to remove rows.

The generated payload remains inert data until the decoders and checked facts
in this module accept it. In particular, stage labels and compiler indices are
not used as semantic evidence.

Emits constraints: no.

| Decoder leaf | Mathematical obligation | Authority class |
|---|---|---|
| source partition | constant, slots, definitions, and eliminated columns are exact | checked |
| compiler program | decoded linear definitions form one SSA program | checked |
| symbolic forms | final-column forms evaluate to compiler values | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

abbrev sourceArmColumnCount : Nat :=
  Materialized.Checked.sourceArtifact.scope.sourceArmColumnCount

abbrev finalRelationColumnCount : Nat :=
  Materialized.Checked.finalRelationColumns

/-- A source-LC term accepted only after its column and coefficient have been
checked against the exact fixture and Goldilocks representation. -/
structure DecodedSourceTerm where
  column : Nat
  coefficient : Nat
  columnBound : column < sourceArmColumnCount
  sourceColumn : column ∈ Materialized.Checked.sourceColumns
  coefficientPositive : 0 < coefficient
  coefficientCanonical : coefficient < goldilocksP

/-- A source LC with its constant represented separately until conversion to
the compiler's constant-one-column convention. -/
structure DecodedSourceLinearCombination where
  constant : Nat
  constantCanonical : constant < goldilocksP
  terms : List DecodedSourceTerm

/-- An exact retained source column and the final-relation field slot from
which it is reconstructed. -/
structure DecodedSourceSlot where
  column : Nat
  start : Nat
  width : Nat
  columnPositive : 0 < column
  columnBound : column < sourceArmColumnCount
  sourceColumn : column ∈ Materialized.Checked.sourceColumns
  startPositive : 0 < start
  supportedWidth : width = 1 ∨ width = 41 ∨ width = 64
  endBound : start + width ≤ finalRelationColumnCount

/-- One compiler linear definition. Its target and every RHS term are decoded
before it can enter `Program.run`. -/
structure DecodedSourceDefinition where
  target : Nat
  targetPositive : 0 < target
  targetBound : target < sourceArmColumnCount
  sourceColumn : target ∈ Materialized.Checked.sourceColumns
  rhs : DecodedSourceLinearCombination

structure DecodedProvenance where
  slots : List DecodedSourceSlot
  definitions : List DecodedSourceDefinition

def decodeSourceTerm (raw : RawSourceTerm) : Option DecodedSourceTerm :=
  if columnBound : raw.column < sourceArmColumnCount then
    if sourceColumn : raw.column ∈ Materialized.Checked.sourceColumns then
      if coefficientPositive : 0 < raw.coefficient then
        if coefficientCanonical : raw.coefficient < goldilocksP then
          some
            { column := raw.column
              coefficient := raw.coefficient
              columnBound
              sourceColumn
              coefficientPositive
              coefficientCanonical }
        else
          none
      else
        none
    else
      none
  else
    none

def decodeSourceLinearCombination (raw : RawSourceLinearCombination) :
    Option DecodedSourceLinearCombination :=
  if constantCanonical : raw.constant < goldilocksP then do
    let terms ← raw.terms.mapM decodeSourceTerm
    pure
      { constant := raw.constant
        constantCanonical
        terms }
  else
    none

def decodeSourceSlot (raw : RawSourceSlot) : Option DecodedSourceSlot :=
  if columnPositive : 0 < raw.column then
    if columnBound : raw.column < sourceArmColumnCount then
      if sourceColumn : raw.column ∈ Materialized.Checked.sourceColumns then
        if startPositive : 0 < raw.start then
          if supportedWidth : raw.width = 1 ∨ raw.width = 41 ∨ raw.width = 64 then
            if endBound : raw.start + raw.width ≤ finalRelationColumnCount then
              some
                { column := raw.column
                  start := raw.start
                  width := raw.width
                  columnPositive
                  columnBound
                  sourceColumn
                  startPositive
                  supportedWidth
                  endBound }
            else
              none
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

def decodeSourceDefinition (raw : RawSourceDefinition) :
    Option DecodedSourceDefinition :=
  if targetPositive : 0 < raw.target then
    if targetBound : raw.target < sourceArmColumnCount then
      if sourceColumn : raw.target ∈ Materialized.Checked.sourceColumns then do
        let rhs ← decodeSourceLinearCombination
          { constant := raw.constant, terms := raw.terms }
        pure
          { target := raw.target
            targetPositive
            targetBound
            sourceColumn
            rhs }
      else
        none
    else
      none
  else
    none

def decodeProvenance : Option DecodedProvenance := do
  let slots ← Materialized.Checked.retainedSlots.mapM decodeSourceSlot
  let definitions ← Materialized.Checked.linearDefinitions.mapM
    decodeSourceDefinition
  pure { slots, definitions }

set_option maxRecDepth 100000 in
theorem provenanceDecodes : decodeProvenance.isSome := by
  native_decide

/-- The only source decoder payload exposed to semantic correspondence. -/
def decoded : DecodedProvenance :=
  decodeProvenance.get provenanceDecodes

def DecodedSourceTerm.asProgramTerm (term : DecodedSourceTerm) : Nat × Nat :=
  (term.column, term.coefficient)

/-- Convert a source constant to the R1CS constant-one column. Zero is omitted
so every emitted sparse coefficient stays nonzero. -/
def DecodedSourceLinearCombination.programTerms
    (linear : DecodedSourceLinearCombination) : List (Nat × Nat) :=
  (if linear.constant = 0 then [] else [(0, linear.constant)]) ++
    linear.terms.map DecodedSourceTerm.asProgramTerm

def DecodedSourceDefinition.programDefinition
    (definition : DecodedSourceDefinition) : Program.Definition :=
  { output := definition.target
    rhs := .linear definition.rhs.programTerms }

def compilerKnownColumns : List Nat :=
  0 :: decoded.slots.map (fun slot => slot.column)

def compilerDefinitions : List Program.Definition :=
  decoded.definitions.map DecodedSourceDefinition.programDefinition

/-- The radix rule used by Rust's `append_slot`: balanced width uses radix
three; the recognized binary widths use radix two. -/
def slotRadix (width : Nat) : Nat :=
  if width = 41 then 3 else 2

def slotExpansionTerms (start width : Nat) : List (Nat × Nat) :=
  (List.range width).map fun offset =>
    (start + offset, slotRadix width ^ offset % goldilocksP)

def DecodedSourceSlot.expansionTerms (slot : DecodedSourceSlot) :
    List (Nat × Nat) :=
  slotExpansionTerms slot.start slot.width

theorem slotRadix_balanced {slot : DecodedSourceSlot}
    (balanced : slot.width = 41) :
    slotRadix slot.width = 3 := by
  simp [slotRadix, balanced]

theorem slotRadix_binary {slot : DecodedSourceSlot}
    (binary : slot.width = 1 ∨ slot.width = 64) :
    slotRadix slot.width = 2 := by
  rcases binary with width | width <;>
    simp [slotRadix, width]

def sourcePartitionColumns : List Nat :=
  [0] ++
    decoded.slots.map (fun slot => slot.column) ++
    decoded.definitions.map (fun definition => definition.target) ++
    Materialized.Checked.traceEliminatedColumns

theorem sourceArmChecked : Materialized.Checked.sourceArm = 2 := by
  native_decide

private def strictlyIncreasingDecidable :
    (columns : List Nat) → Decidable (StrictlyIncreasing columns)
  | [] => isTrue trivial
  | [_] => isTrue trivial
  | first :: second :: rest =>
      if ordered : first < second then
        match strictlyIncreasingDecidable (second :: rest) with
        | isTrue tail => isTrue ⟨ordered, tail⟩
        | isFalse notTail => isFalse (fun valid => notTail valid.2)
      else
        isFalse (fun valid => ordered valid.1)

private instance (columns : List Nat) :
    Decidable (StrictlyIncreasing columns) :=
  strictlyIncreasingDecidable columns

set_option maxRecDepth 100000 in
theorem sourceColumnsStrictlyIncreasing :
    StrictlyIncreasing Materialized.Checked.sourceColumns := by
  native_decide

set_option maxRecDepth 100000 in
theorem sourceColumnsBounded :
    ∀ column ∈ Materialized.Checked.sourceColumns,
      column < sourceArmColumnCount := by
  native_decide

/- Sorting preserves multiplicity, so equality with the strictly increasing
source list checks both complete coverage and pairwise-disjoint ownership by
constant one, retained slots, compiler definitions, and trace elimination. -/
set_option maxRecDepth 100000 in
theorem completeSourcePartition :
    sourcePartitionColumns.mergeSort (fun left right => decide (left ≤ right)) =
      Materialized.Checked.sourceColumns := by
  native_decide

set_option maxRecDepth 100000 in
theorem compilerProgramWellFormed :
    Program.WellFormed compilerKnownColumns compilerDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
theorem compilerDefinitionsCanonical :
    ∀ definition ∈ compilerDefinitions, definition.Canonical := by
  native_decide

def retainedSlot? (column : Nat) : Option DecodedSourceSlot :=
  decoded.slots.find? fun slot => decide (slot.column = column)

/-- Executable retained-slot index for symbolic coefficient reconstruction and
the canonical seed. The theorem below proves that this indexed path implements
the same first-match lookup exactly. -/
def retainedSlotMapOfList : List DecodedSourceSlot →
    Std.HashMap Nat DecodedSourceSlot
  | [] => {}
  | slot :: rest =>
      (retainedSlotMapOfList rest).insert slot.column slot

def retainedSlotMap : Std.HashMap Nat DecodedSourceSlot :=
  retainedSlotMapOfList decoded.slots

def retainedSlotFast? (column : Nat) : Option DecodedSourceSlot :=
  retainedSlotMap[column]?

theorem retainedSlotMapOfList_get? (slots : List DecodedSourceSlot)
    (column : Nat) :
    (retainedSlotMapOfList slots)[column]? =
      slots.find? (fun slot => decide (slot.column = column)) := by
  induction slots with
  | nil => simp [retainedSlotMapOfList]
  | cons slot rest inductionHypothesis =>
      rw [retainedSlotMapOfList, Std.HashMap.getElem?_insert,
        inductionHypothesis]
      by_cases equal : slot.column = column
      · simp [equal]
      · simp [equal]

theorem retainedSlotFast_eq (column : Nat) :
    retainedSlotFast? column = retainedSlot? column := by
  exact retainedSlotMapOfList_get? decoded.slots column

/-- Canonical seed for the retained compiler inputs. Unknown columns are zero;
the complete checked partition prevents an unknown column from being used by
the checked compiler program. -/
def retainedSeed (assignment : Nat → Nat) : Nat → Nat :=
  fun column =>
    if column = 0 then
      assignment 0 % goldilocksP
    else
      match retainedSlotFast? column with
      | some slot => lcEval assignment slot.expansionTerms
      | none => 0

theorem retainedSeedCanonical (assignment : Nat → Nat) :
    ∀ column, retainedSeed assignment column < goldilocksP := by
  intro column
  simp only [retainedSeed]
  split
  · exact Nat.mod_lt _ (by decide)
  · split
    · unfold lcEval
      exact Nat.mod_lt _ (by decide)
    · decide

theorem retainedSeedConstantOne {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    retainedSeed assignment 0 = 1 := by
  simp [retainedSeed, constantOne, goldilocksP]

/-- Deterministic reconstruction of the compiler-defined source columns. -/
def compilerAssignment (assignment : Nat → Nat) : Nat → Nat :=
  Program.run (retainedSeed assignment) compilerDefinitions

theorem compilerAssignmentCanonical (assignment : Nat → Nat) :
    ∀ column, compilerAssignment assignment column < goldilocksP := by
  exact Program.run_canonical (retainedSeedCanonical assignment)

theorem compilerAssignmentDefinitionsHold (assignment : Nat → Nat) :
    ∀ definition ∈ compilerDefinitions,
      definition.Holds (compilerAssignment assignment) := by
  exact Program.run_definitions_hold compilerProgramWellFormed
    (retainedSeed assignment)

theorem compilerAssignmentConstantOne {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    compilerAssignment assignment 0 = 1 := by
  unfold compilerAssignment
  rw [Program.run_preserves_known compilerProgramWellFormed
    (retainedSeed assignment) 0 (by simp [compilerKnownColumns])]
  exact retainedSeedConstantOne constantOne

/-! ## Unnormalized symbolic compiler -/

/-- Sparse final-relation form. The list intentionally preserves append order
and duplicate columns; normalization is a separate coefficient-checking step. -/
abbrev LinearForm :=
  List Materialized.LinearForm.Term

def evalLinearForm (assignment : Nat → Nat) (form : LinearForm) : F :=
  Materialized.LinearForm.eval
    (fun column => Materialized.Semantics.fieldResidue (assignment column)) form

def natTermsLinearForm (terms : List (Nat × Nat)) : LinearForm :=
  terms.map fun term =>
    (term.1, Materialized.Semantics.fieldResidue term.2)

/-- Final-column forms of the retained compiler inputs. -/
def retainedLinearForms (column : Nat) : LinearForm :=
  if column = 0 then
    natTermsLinearForm [(0, 1)]
  else
    match retainedSlotFast? column with
    | some slot => natTermsLinearForm slot.expansionTerms
    | none => []

def scaleLinearForm (coefficient : Nat) (form : LinearForm) : LinearForm :=
  Materialized.LinearForm.scale
    (Materialized.Semantics.fieldResidue coefficient) form

def substituteLinearTerms (forms : Nat → LinearForm)
    (terms : List (Nat × Nat)) : LinearForm :=
  terms.flatMap fun term => scaleLinearForm term.2 (forms term.1)

def setLinearForm (forms : Nat → LinearForm) (column : Nat)
    (form : LinearForm) : Nat → LinearForm :=
  fun candidate => if candidate = column then form else forms candidate

def executeLinearForm (forms : Nat → LinearForm)
    (definition : DecodedSourceDefinition) : Nat → LinearForm :=
  setLinearForm forms definition.target
    (substituteLinearTerms forms definition.rhs.programTerms)

/-- Symbolically execute the same ordered decoded definitions as
`compilerAssignment`, without sorting or coalescing the resulting forms. -/
def runLinearForms :
    (Nat → LinearForm) → List DecodedSourceDefinition → Nat → LinearForm
  | forms, [] => forms
  | forms, definition :: rest =>
      runLinearForms (executeLinearForm forms definition) rest

/-- Eager finite override store for symbolic compiler execution. The base
function owns retained inputs; each decoded definition adds one materialized
output form. -/
structure LinearStore where
  base : Nat → LinearForm
  overrides : Std.HashMap Nat LinearForm

def LinearStore.get (store : LinearStore) (column : Nat) : LinearForm :=
  store.overrides.getD column (store.base column)

def LinearStore.set (store : LinearStore) (column : Nat)
    (form : LinearForm) : LinearStore :=
  { store with overrides := store.overrides.insert column form }

def emptyLinearStore : LinearStore :=
  { base := retainedLinearForms, overrides := {} }

def executeLinearStore (store : LinearStore)
    (definition : DecodedSourceDefinition) : LinearStore :=
  store.set definition.target
    (substituteLinearTerms store.get definition.rhs.programTerms)

def runLinearStore :
    LinearStore → List DecodedSourceDefinition → LinearStore
  | store, [] => store
  | store, definition :: rest =>
      runLinearStore (executeLinearStore store definition) rest

def compilerLinearStore : LinearStore :=
  runLinearStore emptyLinearStore decoded.definitions

def compilerLinearForms : Nat → LinearForm :=
  compilerLinearStore.get

def LinearStore.Represents (store : LinearStore)
    (forms : Nat → LinearForm) : Prop :=
  ∀ column, store.get column = forms column

private theorem emptyLinearStore_represents :
    emptyLinearStore.Represents retainedLinearForms := by
  intro column
  simp [LinearStore.get, emptyLinearStore]

private theorem LinearStore.set_represents {store : LinearStore}
    {forms : Nat → LinearForm} (represents : store.Represents forms)
    (column : Nat) (form : LinearForm) :
    (store.set column form).Represents
      (setLinearForm forms column form) := by
  intro candidate
  simp only [LinearStore.get, LinearStore.set, Std.HashMap.getD_insert,
    setLinearForm]
  by_cases same : column = candidate
  · simp [same]
  · simpa [same, Ne.symm same, LinearStore.get] using
      represents candidate

private theorem substituteLinearTerms_congr
    {left right : Nat → LinearForm}
    (agreement : ∀ column, left column = right column) :
    ∀ terms, substituteLinearTerms left terms =
      substituteLinearTerms right terms := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change
        scaleLinearForm head.2 (left head.1) ++
            substituteLinearTerms left tail =
          scaleLinearForm head.2 (right head.1) ++
            substituteLinearTerms right tail
      rw [agreement head.1, inductionHypothesis]

private theorem executeLinearStore_represents {store : LinearStore}
    {forms : Nat → LinearForm} (represents : store.Represents forms)
    (definition : DecodedSourceDefinition) :
    (executeLinearStore store definition).Represents
      (executeLinearForm forms definition) := by
  unfold executeLinearStore executeLinearForm
  rw [substituteLinearTerms_congr represents]
  exact LinearStore.set_represents represents _ _

private theorem runLinearStore_represents (store : LinearStore)
    (forms : Nat → LinearForm) (represents : store.Represents forms) :
    ∀ definitions,
      (runLinearStore store definitions).Represents
        (runLinearForms forms definitions) := by
  intro definitions
  induction definitions generalizing store forms with
  | nil => exact represents
  | cons definition rest inductionHypothesis =>
      simp only [runLinearStore, runLinearForms]
      exact inductionHypothesis
        (executeLinearStore store definition)
        (executeLinearForm forms definition)
        (executeLinearStore_represents represents definition)

theorem compilerLinearForms_eq_run (column : Nat) :
    compilerLinearForms column =
      runLinearForms retainedLinearForms decoded.definitions column := by
  exact runLinearStore_represents emptyLinearStore retainedLinearForms
    emptyLinearStore_represents decoded.definitions column

private theorem evalNatTermsLinearForm_raw
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    evalLinearForm assignment (natTermsLinearForm terms) =
      Materialized.Semantics.fieldResidue
        (Program.rawLcEval assignment terms) := by
  induction terms with
  | nil =>
      simp [evalLinearForm, natTermsLinearForm,
        Materialized.LinearForm.eval, Program.rawLcEval,
        Materialized.Semantics.fieldResidue]
  | cons head tail inductionHypothesis =>
      simp only [natTermsLinearForm, List.map_cons, evalLinearForm,
        Materialized.LinearForm.eval, Materialized.LinearForm.termValue,
        Program.rawLcEval]
      change
        Materialized.Semantics.fieldResidue head.2 *
              Materialized.Semantics.fieldResidue (assignment head.1) +
            evalLinearForm assignment (natTermsLinearForm tail) =
          Materialized.Semantics.fieldResidue
            (head.2 * assignment head.1 + Program.rawLcEval assignment tail)
      rw [inductionHypothesis,
        ← Materialized.Semantics.fieldResidue_mul,
        ← Materialized.Semantics.fieldResidue_add]

theorem evalNatTermsLinearForm (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    evalLinearForm assignment (natTermsLinearForm terms) =
      Materialized.Semantics.fieldResidue (lcEval assignment terms) := by
  rw [evalNatTermsLinearForm_raw, Program.lcEval_eq_raw_mod]
  apply Fin.ext
  simp [Materialized.Semantics.fieldResidue,
    Materialized.Semantics.modulus_eq]

theorem evalRetainedLinearForms (assignment : Nat → Nat) (column : Nat) :
    evalLinearForm assignment (retainedLinearForms column) =
      Materialized.Semantics.fieldResidue (retainedSeed assignment column) := by
  simp only [retainedLinearForms, retainedSeed]
  split
  · simpa [lcEval, Materialized.Semantics.fieldResidue,
      Materialized.Semantics.modulus_eq, Nat.mod_mod] using
      evalNatTermsLinearForm assignment [(0, 1)]
  · split
    · exact evalNatTermsLinearForm assignment _
    · rfl

private theorem evalSubstituteLinearTerms_eq_natTerms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm assignment (forms column) =
        Materialized.Semantics.fieldResidue (values column)) :
    ∀ terms,
      evalLinearForm assignment (substituteLinearTerms forms terms) =
        evalLinearForm values (natTermsLinearForm terms) := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [substituteLinearTerms, List.flatMap_cons, scaleLinearForm,
        natTermsLinearForm, List.map_cons, evalLinearForm,
        Materialized.LinearForm.eval, Materialized.LinearForm.termValue]
      rw [Materialized.LinearForm.eval_append,
        Materialized.LinearForm.eval_scale]
      change
        Materialized.Semantics.fieldResidue head.2 *
              evalLinearForm assignment (forms head.1) +
            evalLinearForm assignment (substituteLinearTerms forms tail) =
          Materialized.Semantics.fieldResidue head.2 *
              Materialized.Semantics.fieldResidue (values head.1) +
            evalLinearForm values (natTermsLinearForm tail)
      rw [agreement, inductionHypothesis]

theorem evalSubstituteLinearTerms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm assignment (forms column) =
        Materialized.Semantics.fieldResidue (values column))
    (terms : List (Nat × Nat)) :
    evalLinearForm assignment (substituteLinearTerms forms terms) =
      Materialized.Semantics.fieldResidue (lcEval values terms) := by
  rw [evalSubstituteLinearTerms_eq_natTerms assignment forms values agreement]
  exact evalNatTermsLinearForm values terms

private theorem evalExecuteLinearForm
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm assignment (forms column) =
        Materialized.Semantics.fieldResidue (values column))
    (definition : DecodedSourceDefinition) :
    ∀ column,
      evalLinearForm assignment (executeLinearForm forms definition column) =
        Materialized.Semantics.fieldResidue
          (Program.execute values definition.programDefinition column) := by
  intro column
  by_cases isOutput : column = definition.target
  · subst column
    simp only [executeLinearForm, setLinearForm, ↓reduceIte,
      DecodedSourceDefinition.programDefinition, Program.execute,
      Program.setColumn_same, Program.Rhs.eval]
    exact evalSubstituteLinearTerms assignment forms values agreement _
  · simp only [executeLinearForm, setLinearForm, isOutput, ↓reduceIte,
      DecodedSourceDefinition.programDefinition, Program.execute,
      Program.setColumn_other values isOutput]
    exact agreement column

private theorem evalRunLinearForms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm assignment (forms column) =
        Materialized.Semantics.fieldResidue (values column)) :
    ∀ definitions column,
      evalLinearForm assignment (runLinearForms forms definitions column) =
        Materialized.Semantics.fieldResidue
          (Program.run values
            (definitions.map DecodedSourceDefinition.programDefinition) column) := by
  intro definitions
  induction definitions generalizing forms values with
  | nil =>
      intro column
      exact agreement column
  | cons definition rest inductionHypothesis =>
      intro column
      simp only [runLinearForms, List.map_cons, Program.run]
      exact inductionHypothesis
        (executeLinearForm forms definition)
        (Program.execute values definition.programDefinition)
        (evalExecuteLinearForm assignment forms values agreement definition)
        column

/-- Evaluating the unnormalized symbolic form gives exactly the field residue
of the executable compiler assignment for every final assignment and source
column. No source-row satisfaction premise is used. -/
theorem evalCompilerLinearForm (assignment : Nat → Nat) (column : Nat) :
    evalLinearForm assignment (compilerLinearForms column) =
      Materialized.Semantics.fieldResidue
        (compilerAssignment assignment column) := by
  rw [compilerLinearForms_eq_run]
  exact evalRunLinearForms assignment retainedLinearForms
    (retainedSeed assignment) (evalRetainedLinearForms assignment)
    decoded.definitions column

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
