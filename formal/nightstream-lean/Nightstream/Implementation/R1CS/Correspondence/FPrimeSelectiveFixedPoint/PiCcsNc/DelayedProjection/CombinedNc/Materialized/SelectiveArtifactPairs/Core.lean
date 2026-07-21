import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.EmittedRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceAssignment

/-!
Proof-free coefficient surface for the generated production combined-NC
selective rows.

Owns: structural validity predicates for raw emitted/rewrite/retained records,
the symbolic execution of the exact generated linear-definition program, the
thirteen expected physical port forms, and compact normalized coefficient
certificates.

Does not own: the truth of any generated certificate, selected-row
satisfaction, source-program execution, selector truth, transcript order,
parent or raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none.

The executable certificate records below contain only natural numbers,
arrays, lists, and booleans.  Decoded structures and their proof fields are
never certificate inputs.  `Decode.lean` is the kernel bridge from these raw
facts to the typed semantic obligations in `SelectiveCompilerBridge`.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Small symbolic linear-form kernel -/

abbrev LinearTerm := Nat × F
abbrev LinearForm := List LinearTerm
abbrev CompactTerm := Nat × Nat
abbrev CompactForm := List CompactTerm

def termValue (assignment : Nat → F) (term : LinearTerm) : F :=
  term.2 * assignment term.1

def evalLinearForm (assignment : Nat → F) : LinearForm → F
  | [] => 0
  | term :: rest => termValue assignment term + evalLinearForm assignment rest

private theorem fadd_assoc (a b c : F) :
    (a + b) + c = a + (b + c) :=
  Lean.Grind.Fin.add_assoc _ _ _

private theorem fadd_comm (a b : F) : a + b = b + a :=
  Lean.Grind.Fin.add_comm _ _

private theorem fadd_left_comm (a b c : F) :
    a + (b + c) = b + (a + c) := by
  rw [← fadd_assoc, fadd_comm a b, fadd_assoc]

private theorem fadd_mul (a b c : F) :
    (a + b) * c = a * c + b * c := by
  calc
    (a + b) * c = c * (a + b) := Fin.mul_comm _ _
    _ = c * a + c * b := Lean.Grind.Fin.left_distrib _ _ _
    _ = a * c + b * c := by
      rw [Fin.mul_comm c a, Fin.mul_comm c b]

private theorem fmul_add (a b c : F) :
    a * (b + c) = a * b + a * c :=
  Lean.Grind.Fin.left_distrib _ _ _

def insertLinearTerm (term : LinearTerm) : LinearForm → LinearForm
  | [] => if term.2 = 0 then [] else [term]
  | head :: rest =>
      if term.2 = 0 then
        head :: rest
      else if term.1 < head.1 then
        term :: head :: rest
      else if term.1 = head.1 then
        let coefficient := term.2 + head.2
        if coefficient = 0 then rest else (head.1, coefficient) :: rest
      else
        head :: insertLinearTerm term rest

theorem eval_insertLinearTerm (assignment : Nat → F)
    (term : LinearTerm) :
    ∀ terms,
      evalLinearForm assignment (insertLinearTerm term terms) =
        termValue assignment term + evalLinearForm assignment terms := by
  rcases term with ⟨termColumn, termCoefficient⟩
  intro terms
  induction terms with
  | nil =>
      by_cases coefficientZero : termCoefficient = 0
      · simp [insertLinearTerm, coefficientZero, termValue,
          evalLinearForm, Fin.zero_mul]
      · simp [insertLinearTerm, coefficientZero, termValue, evalLinearForm]
  | cons head rest inductionHypothesis =>
      rcases head with ⟨headColumn, headCoefficient⟩
      by_cases coefficientZero : termCoefficient = 0
      · simp [insertLinearTerm, coefficientZero, termValue,
          evalLinearForm, Fin.zero_mul]
      · simp only [insertLinearTerm, coefficientZero, ↓reduceIte]
        by_cases before : termColumn < headColumn
        · simp only [before, ↓reduceIte, evalLinearForm, termValue]
        · simp only [before, ↓reduceIte]
          by_cases same : termColumn = headColumn
          · subst headColumn
            simp only [↓reduceIte, evalLinearForm, termValue]
            by_cases sumZero : termCoefficient + headCoefficient = 0
            · simp only [sumZero, ↓reduceIte]
              have multiplied := congrArg
                (fun value : F => value * assignment termColumn) sumZero
              change
                (termCoefficient + headCoefficient) * assignment termColumn =
                  0 * assignment termColumn at multiplied
              rw [fadd_mul, Fin.zero_mul] at multiplied
              rw [← fadd_assoc, multiplied, Fin.zero_add]
            · simp only [sumZero, ↓reduceIte, evalLinearForm, termValue]
              rw [fadd_mul, fadd_assoc]
          · simp only [same, ↓reduceIte, evalLinearForm,
              inductionHypothesis, termValue]
            exact fadd_left_comm _ _ _

def normalizeLinearForm (terms : LinearForm) : LinearForm :=
  terms.foldr insertLinearTerm []

theorem eval_normalizeLinearForm (assignment : Nat → F)
    (terms : LinearForm) :
    evalLinearForm assignment (normalizeLinearForm terms) =
      evalLinearForm assignment terms := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [normalizeLinearForm, List.foldr_cons,
        eval_insertLinearTerm, evalLinearForm]
      unfold normalizeLinearForm at inductionHypothesis
      rw [inductionHypothesis]

def compactTerm (term : LinearTerm) : CompactTerm :=
  (term.1, term.2.val)

def normalizedShape (terms : LinearForm) : CompactForm :=
  (normalizeLinearForm terms).map compactTerm

private theorem compactTerm_injective : Function.Injective compactTerm := by
  intro left right equal
  apply Prod.ext
  · exact congrArg (fun value : CompactTerm => value.1) equal
  · apply Fin.ext
    exact congrArg (fun value : CompactTerm => value.2) equal

private theorem compactTerms_injective :
    Function.Injective (List.map compactTerm) := by
  intro left right equal
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at equal
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at equal
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          have headEqual := compactTerm_injective equal.1
          have tailEqual := inductionHypothesis equal.2
          subst rightHead
          subst rightTail
          rfl

theorem eval_eq_of_normalizedShape_eq {left right : LinearForm}
    (equal : normalizedShape left = normalizedShape right)
    (assignment : Nat → F) :
    evalLinearForm assignment left = evalLinearForm assignment right := by
  unfold normalizedShape at equal
  have normalizedEqual :
      normalizeLinearForm left = normalizeLinearForm right :=
    compactTerms_injective equal
  rw [← eval_normalizeLinearForm assignment left,
    ← eval_normalizeLinearForm assignment right, normalizedEqual]

def scaleLinearForm (coefficient : F) (terms : LinearForm) : LinearForm :=
  terms.map fun term => (term.1, coefficient * term.2)

theorem eval_append (assignment : Nat → F) (left right : LinearForm) :
    evalLinearForm assignment (left ++ right) =
      evalLinearForm assignment left + evalLinearForm assignment right := by
  induction left with
  | nil => simp [evalLinearForm]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, evalLinearForm, inductionHypothesis]
      exact (fadd_assoc _ _ _).symm

theorem eval_scale (assignment : Nat → F) (coefficient : F)
    (terms : LinearForm) :
    evalLinearForm assignment (scaleLinearForm coefficient terms) =
      coefficient * evalLinearForm assignment terms := by
  induction terms with
  | nil => exact (Fin.mul_zero coefficient).symm
  | cons head tail inductionHypothesis =>
      simp only [scaleLinearForm, List.map_cons, evalLinearForm, termValue]
      unfold scaleLinearForm at inductionHypothesis
      rw [inductionHypothesis, Fin.mul_assoc, fmul_add]

def natTermsLinearForm (terms : List (Nat × Nat)) : LinearForm :=
  terms.map fun term => (term.1, Semantics.fieldResidue term.2)

private theorem fieldResidue_add (left right : Nat) :
    Semantics.fieldResidue (left + right) =
      Semantics.fieldResidue left + Semantics.fieldResidue right := by
  apply Fin.ext
  simp [Semantics.fieldResidue, Fin.val_add, Nat.add_mod]

private theorem fieldResidue_mul (left right : Nat) :
    Semantics.fieldResidue (left * right) =
      Semantics.fieldResidue left * Semantics.fieldResidue right := by
  apply Fin.ext
  simp [Semantics.fieldResidue, Fin.val_mul, Nat.mul_mod]

/-! ## Exact symbolic execution of the generated compiler definitions -/

def retainedLinearForms (column : Nat) : LinearForm :=
  if column = 0 then
    natTermsLinearForm [(0, 1)]
  else
    match SourceAssignment.retainedSlot? column with
    | some slot => natTermsLinearForm
        (SourceAssignment.RawSourceSlot.expansionTerms slot)
    | none => []

def substituteLinearTerms (forms : Nat → LinearForm)
    (terms : List (Nat × Nat)) : LinearForm :=
  terms.flatMap fun term =>
    scaleLinearForm (Semantics.fieldResidue term.2) (forms term.1)

def setLinearForm (forms : Nat → LinearForm) (column : Nat)
    (form : LinearForm) : Nat → LinearForm :=
  fun candidate => if candidate = column then form else forms candidate

def executeLinearForm (forms : Nat → LinearForm)
    (definition : RawSourceDefinition) : Nat → LinearForm :=
  setLinearForm forms definition.target
    (substituteLinearTerms forms
      (SourceAssignment.RawLinearCombination.programTerms definition.value))

def runLinearForms :
    (Nat → LinearForm) → List RawSourceDefinition → Nat → LinearForm
  | forms, [] => forms
  | forms, definition :: rest =>
      runLinearForms (executeLinearForm forms definition) rest

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
    (definition : RawSourceDefinition) : LinearStore :=
  store.set definition.target
    (substituteLinearTerms store.get
      (SourceAssignment.RawLinearCombination.programTerms definition.value))

def runLinearStore :
    LinearStore → List RawSourceDefinition → LinearStore
  | store, [] => store
  | store, definition :: rest =>
      runLinearStore (executeLinearStore store definition) rest

def compilerLinearStore : LinearStore :=
  runLinearStore emptyLinearStore Provenance.linearDefinitions

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
    ∀ terms,
      substituteLinearTerms left terms = substituteLinearTerms right terms := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change
        scaleLinearForm (Semantics.fieldResidue head.2) (left head.1) ++
            substituteLinearTerms left tail =
          scaleLinearForm (Semantics.fieldResidue head.2) (right head.1) ++
            substituteLinearTerms right tail
      rw [agreement head.1, inductionHypothesis]

private theorem executeLinearStore_represents {store : LinearStore}
    {forms : Nat → LinearForm} (represents : store.Represents forms)
    (definition : RawSourceDefinition) :
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
      runLinearForms retainedLinearForms Provenance.linearDefinitions column := by
  exact runLinearStore_represents emptyLinearStore retainedLinearForms
    emptyLinearStore_represents Provenance.linearDefinitions column

private theorem evalNatTermsLinearForm_raw (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    evalLinearForm (fun column => Semantics.fieldResidue (assignment column))
        (natTermsLinearForm terms) =
      Semantics.fieldResidue (Program.rawLcEval assignment terms) := by
  induction terms with
  | nil =>
      simp [evalLinearForm, natTermsLinearForm, Program.rawLcEval,
        Semantics.fieldResidue]
  | cons head tail inductionHypothesis =>
      simp only [natTermsLinearForm, List.map_cons, evalLinearForm, termValue,
        Program.rawLcEval]
      change
        Semantics.fieldResidue head.2 *
              Semantics.fieldResidue (assignment head.1) +
            evalLinearForm
              (fun column => Semantics.fieldResidue (assignment column))
              (natTermsLinearForm tail) =
          Semantics.fieldResidue
            (head.2 * assignment head.1 + Program.rawLcEval assignment tail)
      rw [inductionHypothesis, ← fieldResidue_mul,
        ← fieldResidue_add]

theorem evalNatTermsLinearForm (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    evalLinearForm (fun column => Semantics.fieldResidue (assignment column))
        (natTermsLinearForm terms) =
      Semantics.fieldResidue (lcEval assignment terms) := by
  have modulus_eq : goldilocksP = goldilocksModulus := by rfl
  rw [evalNatTermsLinearForm_raw, Program.lcEval_eq_raw_mod]
  apply Fin.ext
  simp [Semantics.fieldResidue, modulus_eq, Nat.mod_mod]

theorem evalRetainedLinearForms (assignment : Nat → Nat) (column : Nat) :
    evalLinearForm (fun candidate => Semantics.fieldResidue (assignment candidate))
        (retainedLinearForms column) =
      Semantics.fieldResidue (SourceAssignment.retainedSeed assignment column) := by
  by_cases hzero : column = 0
  · simp only [retainedLinearForms, SourceAssignment.retainedSeed, hzero,
      ↓reduceIte]
    simpa [lcEval, Semantics.fieldResidue, Nat.mod_mod] using
      evalNatTermsLinearForm assignment [(0, 1)]
  · simp only [retainedLinearForms, SourceAssignment.retainedSeed, hzero,
      ↓reduceIte]
    cases hslot : SourceAssignment.retainedSlot? column with
    | none =>
        simp [hslot, evalLinearForm, Semantics.fieldResidue]
    | some slot =>
        simp only [hslot]
        exact evalNatTermsLinearForm assignment
          (SourceAssignment.RawSourceSlot.expansionTerms slot)

private theorem evalSubstituteLinearTerms_eq_natTerms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (forms column) =
        Semantics.fieldResidue (values column)) :
    ∀ terms,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (substituteLinearTerms forms terms) =
        evalLinearForm
          (fun candidate => Semantics.fieldResidue (values candidate))
          (natTermsLinearForm terms) := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [substituteLinearTerms, List.flatMap_cons,
        natTermsLinearForm, List.map_cons, evalLinearForm, termValue]
      rw [eval_append, eval_scale]
      change
        Semantics.fieldResidue head.2 *
              evalLinearForm
                (fun candidate => Semantics.fieldResidue (assignment candidate))
                (forms head.1) +
            evalLinearForm
              (fun candidate => Semantics.fieldResidue (assignment candidate))
              (substituteLinearTerms forms tail) =
          Semantics.fieldResidue head.2 *
              Semantics.fieldResidue (values head.1) +
            evalLinearForm
              (fun candidate => Semantics.fieldResidue (values candidate))
              (natTermsLinearForm tail)
      rw [agreement, inductionHypothesis]

private theorem evalSubstituteLinearTerms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (forms column) =
        Semantics.fieldResidue (values column))
    (terms : List (Nat × Nat)) :
    evalLinearForm
        (fun candidate => Semantics.fieldResidue (assignment candidate))
        (substituteLinearTerms forms terms) =
      Semantics.fieldResidue (lcEval values terms) := by
  rw [evalSubstituteLinearTerms_eq_natTerms assignment forms values agreement]
  exact evalNatTermsLinearForm values terms

private theorem evalExecuteLinearForm
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (forms column) =
        Semantics.fieldResidue (values column))
    (definition : RawSourceDefinition) :
    ∀ column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (executeLinearForm forms definition column) =
        Semantics.fieldResidue
          (Program.execute values
            (SourceAssignment.RawSourceDefinition.programDefinition definition)
            column) := by
  intro column
  by_cases isOutput : column = definition.target
  · subst column
    simp only [executeLinearForm, setLinearForm, ↓reduceIte,
      SourceAssignment.RawSourceDefinition.programDefinition, Program.execute,
      Program.setColumn_same, Program.Rhs.eval]
    exact evalSubstituteLinearTerms assignment forms values agreement _
  · simp only [executeLinearForm, setLinearForm, isOutput, ↓reduceIte,
      SourceAssignment.RawSourceDefinition.programDefinition, Program.execute,
      Program.setColumn_other values isOutput]
    exact agreement column

private theorem evalRunLinearForms
    (assignment : Nat → Nat) (forms : Nat → LinearForm)
    (values : Nat → Nat)
    (agreement : ∀ column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (forms column) =
        Semantics.fieldResidue (values column)) :
    ∀ definitions column,
      evalLinearForm
          (fun candidate => Semantics.fieldResidue (assignment candidate))
          (runLinearForms forms definitions column) =
        Semantics.fieldResidue
          (Program.run values
            (definitions.map
              SourceAssignment.RawSourceDefinition.programDefinition)
            column) := by
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
        (Program.execute values
          (SourceAssignment.RawSourceDefinition.programDefinition definition))
        (evalExecuteLinearForm assignment forms values agreement definition)
        column

/- Symbolic source-column forms evaluate to the exact executable compiler
assignment.  This theorem gives coefficient certificates their independent
bridge to `SourceAssignment.compilerAssignment`. -/
set_option maxRecDepth 100000 in
theorem evalCompilerLinearForm (assignment : Nat → Nat) (column : Nat) :
    evalLinearForm (fun candidate => Semantics.fieldResidue (assignment candidate))
        (compilerLinearForms column) =
      Semantics.fieldResidue
        (SourceAssignment.compilerAssignment assignment column) := by
  rw [compilerLinearForms_eq_run]
  exact evalRunLinearForms assignment retainedLinearForms
    (SourceAssignment.retainedSeed assignment)
    (evalRetainedLinearForms assignment)
    Provenance.linearDefinitions column

/-! ## Actual and expected physical port forms -/

def rawTermLinearForm (term : RawTerm) : LinearForm :=
  [(term.column, Semantics.fieldResidue term.coefficient)]

def rawGeometricRunLinearForm (run : RawGeometricRun) : LinearForm :=
  (List.finRange run.length).map fun offset =>
    (run.columnStart + offset.val,
      Semantics.fieldResidue run.initial *
        Semantics.fieldResidue run.ratio ^ offset.val)

def rawPortLinearForm (port : RawPort) : LinearForm :=
  port.explicit.flatMap rawTermLinearForm ++
    port.geometric.flatMap rawGeometricRunLinearForm

def rawEmittedPortLinearForm (row : RawEmittedRow)
    (port : Fin selectivePortCount) : LinearForm :=
  match row.ports[port.val]? with
  | none => []
  | some value => rawPortLinearForm value

def sourceLinearForm (linear : RawLinearCombination) : LinearForm :=
  substituteLinearTerms compilerLinearForms
    (SourceAssignment.RawLinearCombination.programTerms linear)

def derivedLinearForm (compilerIndex : Nat) : LinearForm :=
  match SourceAssignment.derivedSlot? compilerIndex with
  | none => []
  | some slot => natTermsLinearForm
      (SourceAssignment.slotExpansionTerms slot.start slot.width)

def outputLinearForm : RawRewriteOutput → LinearForm
  | .source value => sourceLinearForm value
  | .derivedProductSum compilerIndex => derivedLinearForm compilerIndex

def previousLinearForm : Option Nat → LinearForm
  | none => []
  | some compilerIndex => derivedLinearForm compilerIndex

def negateLinearForm (form : LinearForm) : LinearForm :=
  scaleLinearForm (-1) form

def rewriteCLinearForm (step : RawRewriteStep) : LinearForm :=
  outputLinearForm step.output ++
    negateLinearForm (sourceLinearForm step.base) ++
    negateLinearForm (previousLinearForm step.previous)

def factorLeftLinearForm (factor : RawProductFactor) : LinearForm :=
  scaleLinearForm (Semantics.fieldResidue factor.coefficient)
    (sourceLinearForm factor.left)

def factorRightLinearForm (factor : RawProductFactor) : LinearForm :=
  sourceLinearForm factor.right

def factorLeftLinearFormAt (factors : List RawProductFactor)
    (index : Nat) : LinearForm :=
  match factors[index]? with
  | none => []
  | some factor => factorLeftLinearForm factor

def factorRightLinearFormAt (factors : List RawProductFactor)
    (index : Nat) : LinearForm :=
  match factors[index]? with
  | none => []
  | some factor => factorRightLinearForm factor

def steadySelectorLinearForm : LinearForm :=
  natTermsLinearForm [(Metadata.steadySelectorColumn, 1)]

def rewritePortLinearForm (step : RawRewriteStep) :
    Fin selectivePortCount → LinearForm
  | ⟨0, _⟩ => factorLeftLinearFormAt step.factors 0
  | ⟨1, _⟩ => []
  | ⟨2, _⟩ => factorRightLinearFormAt step.factors 0
  | ⟨3, _⟩ => factorLeftLinearFormAt step.factors 1
  | ⟨4, _⟩ => rewriteCLinearForm step
  | ⟨5, _⟩ => factorRightLinearFormAt step.factors 1
  | ⟨6, _⟩ => factorLeftLinearFormAt step.factors 2
  | ⟨7, _⟩ => steadySelectorLinearForm
  | ⟨8, _⟩ => factorRightLinearFormAt step.factors 2
  | ⟨9, _⟩ => factorLeftLinearFormAt step.factors 3
  | ⟨10, _⟩ => factorRightLinearFormAt step.factors 3
  | ⟨11, _⟩ => factorLeftLinearFormAt step.factors 4
  | ⟨12, _⟩ => factorRightLinearFormAt step.factors 4

def retainedPortLinearForm (step : RawRetainedStep) :
    Fin selectivePortCount → LinearForm
  | ⟨0, _⟩ => []
  | ⟨1, _⟩ => steadySelectorLinearForm
  | ⟨2, _⟩ => sourceLinearForm step.a
  | ⟨3, _⟩ => sourceLinearForm step.b
  | ⟨4, _⟩ => sourceLinearForm step.c
  | ⟨5, _⟩ => []
  | ⟨6, _⟩ => []
  | ⟨7, _⟩ => []
  | ⟨8, _⟩ => []
  | ⟨9, _⟩ => []
  | ⟨10, _⟩ => []
  | ⟨11, _⟩ => []
  | ⟨12, _⟩ => []

structure CoefficientShape where
  actual : Array CompactForm
  expected : Array CompactForm
deriving DecidableEq, Repr

def rewriteCoefficientShape (row : RawEmittedRow)
    (step : RawRewriteStep) : CoefficientShape :=
  { actual := Array.ofFn fun port : Fin selectivePortCount =>
      normalizedShape (rawEmittedPortLinearForm row port)
    expected := Array.ofFn fun port : Fin selectivePortCount =>
      normalizedShape (rewritePortLinearForm step port) }

def retainedCoefficientShape (row : RawEmittedRow)
    (step : RawRetainedStep) : CoefficientShape :=
  { actual := Array.ofFn fun port : Fin selectivePortCount =>
      normalizedShape (rawEmittedPortLinearForm row port)
    expected := Array.ofFn fun port : Fin selectivePortCount =>
      normalizedShape (retainedPortLinearForm step port) }

def coefficientShapeMatches (shape : CoefficientShape) : Prop :=
  shape.actual = shape.expected

instance (shape : CoefficientShape) : Decidable (coefficientShapeMatches shape) := by
  unfold coefficientShapeMatches
  infer_instance

private theorem arrayShape_eq {actual expected : Fin selectivePortCount → LinearForm}
    (equal :
      Array.ofFn (fun port => normalizedShape (actual port)) =
        Array.ofFn (fun port => normalizedShape (expected port)))
    (port : Fin selectivePortCount) :
    normalizedShape (actual port) = normalizedShape (expected port) := by
  have atPort := congrArg
    (fun values : Array CompactForm => values[port.val]?) equal
  simpa using atPort

theorem rewriteCoefficientShape_semantic
    {row : RawEmittedRow} {step : RawRewriteStep}
    (coefficientExact :
      coefficientShapeMatches (rewriteCoefficientShape row step))
    (assignment : Nat → F) (port : Fin selectivePortCount) :
    evalLinearForm assignment (rawEmittedPortLinearForm row port) =
      evalLinearForm assignment (rewritePortLinearForm step port) := by
  apply eval_eq_of_normalizedShape_eq
  apply arrayShape_eq (port := port)
  simpa [coefficientShapeMatches, rewriteCoefficientShape] using
    coefficientExact

theorem retainedCoefficientShape_semantic
    {row : RawEmittedRow} {step : RawRetainedStep}
    (coefficientExact :
      coefficientShapeMatches (retainedCoefficientShape row step))
    (assignment : Nat → F) (port : Fin selectivePortCount) :
    evalLinearForm assignment (rawEmittedPortLinearForm row port) =
      evalLinearForm assignment (retainedPortLinearForm step port) := by
  apply eval_eq_of_normalizedShape_eq
  apply arrayShape_eq (port := port)
  simpa [coefficientShapeMatches, retainedCoefficientShape] using
    coefficientExact

/-! ## Proof-free structural and pair predicates -/

def RawTermValid (columns : Nat) (term : RawTerm) : Prop :=
  term.column < columns ∧
  term.coefficient < goldilocksModulus ∧
  term.coefficient ≠ 0

instance (columns : Nat) (term : RawTerm) :
    Decidable (RawTermValid columns term) := by
  unfold RawTermValid
  infer_instance

def RawGeometricRunValid (columns : Nat) (run : RawGeometricRun) : Prop :=
  0 < run.length ∧
  run.columnStart + run.length ≤ columns ∧
  run.initial < goldilocksModulus ∧ run.initial ≠ 0 ∧
  run.ratio < goldilocksModulus ∧ run.ratio ≠ 0

instance (columns : Nat) (run : RawGeometricRun) :
    Decidable (RawGeometricRunValid columns run) := by
  unfold RawGeometricRunValid
  infer_instance

def RawPortValid (columns : Nat) (port : RawPort) : Prop :=
  (∀ term ∈ port.explicit, RawTermValid columns term) ∧
  (∀ run ∈ port.geometric, RawGeometricRunValid columns run)

instance (columns : Nat) (port : RawPort) :
    Decidable (RawPortValid columns port) := by
  unfold RawPortValid
  infer_instance

def RawEmittedRowValid (row : RawEmittedRow) : Prop :=
  row.schemaVersion = supportedSchemaVersion ∧
  row.rows = Metadata.finalRelationRows ∧
  row.columns = Metadata.finalRelationColumns ∧
  row.emittedRow < row.rows ∧
  row.ports.length = selectivePortCount ∧
  (∀ port ∈ row.ports, RawPortValid row.columns port)

instance (row : RawEmittedRow) : Decidable (RawEmittedRowValid row) := by
  unfold RawEmittedRowValid RawPortValid RawTermValid RawGeometricRunValid
  infer_instance

def RawLinearCombinationValid (columns : Nat)
    (linear : RawLinearCombination) : Prop :=
  linear.constant < goldilocksModulus ∧
  ∀ term ∈ linear.terms, RawTermValid columns term

instance (columns : Nat) (linear : RawLinearCombination) :
    Decidable (RawLinearCombinationValid columns linear) := by
  unfold RawLinearCombinationValid RawTermValid
  infer_instance

def RawProductFactorValid (columns : Nat) (factor : RawProductFactor) : Prop :=
  RawLinearCombinationValid columns factor.left ∧
  RawLinearCombinationValid columns factor.right ∧
  factor.coefficient < goldilocksModulus

instance (columns : Nat) (factor : RawProductFactor) :
    Decidable (RawProductFactorValid columns factor) := by
  unfold RawProductFactorValid RawLinearCombinationValid RawTermValid
  infer_instance

def RawRewriteOutputValid (columns : Nat) : RawRewriteOutput → Prop
  | .source value => RawLinearCombinationValid columns value
  | .derivedProductSum _ => True

instance (columns : Nat) (output : RawRewriteOutput) :
    Decidable (RawRewriteOutputValid columns output) := by
  cases output <;> simp only [RawRewriteOutputValid] <;> infer_instance

def RawRewriteStepValid (step : RawRewriteStep) : Prop :=
  step.emittedRow < Metadata.finalRelationRows ∧
  (∀ range ∈ step.sourceRows,
    Decoder.rowRangeValid Metadata.sourceRelationRows range) ∧
  RawRewriteOutputValid Metadata.sourceRelationColumns step.output ∧
  RawLinearCombinationValid Metadata.sourceRelationColumns step.base ∧
  (∀ factor ∈ step.factors,
    RawProductFactorValid Metadata.sourceRelationColumns factor) ∧
  step.factors.length ≤ 5

instance (step : RawRewriteStep) : Decidable (RawRewriteStepValid step) := by
  unfold RawRewriteStepValid RawProductFactorValid
    RawLinearCombinationValid RawTermValid Decoder.rowRangeValid
  cases step.output <;> simp only [RawRewriteOutputValid] <;> infer_instance

def RawRetainedStepValid (step : RawRetainedStep) : Prop :=
  step.emittedRow < Metadata.finalRelationRows ∧
  step.sourceRow < Metadata.sourceRelationRows ∧
  RawLinearCombinationValid Metadata.sourceRelationColumns step.a ∧
  RawLinearCombinationValid Metadata.sourceRelationColumns step.b ∧
  RawLinearCombinationValid Metadata.sourceRelationColumns step.c

instance (step : RawRetainedStep) : Decidable (RawRetainedStepValid step) := by
  unfold RawRetainedStepValid RawLinearCombinationValid RawTermValid
  infer_instance

structure RawRewritePair where
  emitted : RawEmittedRow
  provenance : RawRewriteStep
deriving DecidableEq, Repr

structure RawRetainedPair where
  emitted : RawEmittedRow
  provenance : RawRetainedStep
deriving DecidableEq, Repr

def RewritePairCertificate (pair : RawRewritePair) : Prop :=
  RawEmittedRowValid pair.emitted ∧
  RawRewriteStepValid pair.provenance ∧
  pair.emitted.emittedRow = pair.provenance.emittedRow ∧
  coefficientShapeMatches
    (rewriteCoefficientShape pair.emitted pair.provenance)

instance (pair : RawRewritePair) : Decidable (RewritePairCertificate pair) := by
  unfold RewritePairCertificate coefficientShapeMatches
  infer_instance

def RetainedPairCertificate (pair : RawRetainedPair) : Prop :=
  RawEmittedRowValid pair.emitted ∧
  RawRetainedStepValid pair.provenance ∧
  pair.emitted.emittedRow = pair.provenance.emittedRow ∧
  coefficientShapeMatches
    (retainedCoefficientShape pair.emitted pair.provenance)

instance (pair : RawRetainedPair) : Decidable (RetainedPairCertificate pair) := by
  unfold RetainedPairCertificate coefficientShapeMatches
  infer_instance

def RewritePairsCertificate (pairs : List RawRewritePair) : Prop :=
  ∀ pair ∈ pairs, RewritePairCertificate pair

instance (pairs : List RawRewritePair) : Decidable (RewritePairsCertificate pairs) := by
  unfold RewritePairsCertificate
  infer_instance

def RetainedPairsCertificate (pairs : List RawRetainedPair) : Prop :=
  ∀ pair ∈ pairs, RetainedPairCertificate pair

instance (pairs : List RawRetainedPair) : Decidable (RetainedPairsCertificate pairs) := by
  unfold RetainedPairsCertificate
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
