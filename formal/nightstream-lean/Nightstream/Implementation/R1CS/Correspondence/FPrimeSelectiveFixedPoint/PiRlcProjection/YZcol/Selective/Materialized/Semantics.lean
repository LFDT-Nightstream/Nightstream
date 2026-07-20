import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics
import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Executable semantics for one successfully decoded compact selective row.

Owns: exact expansion of a geometric run over its half-open column interval,
additive sparse-port action, the independent thirteen-port selective
polynomial residual, and both field- and Nat-assignment satisfaction views.

Does not own: generated-row validity, selector truth, rewrite correctness,
source-column provenance, source-program execution, protocol authority, or
row removal.

Emits constraints: no.

The Nat view exposes the exact `(column, canonical coefficient)` stream used
by `R1CS.lcEval`. This keeps later `Program.run` correspondence from depending
on an implicit representation conversion.

| Semantic leaf | Mathematical obligation | Authority class |
|---|---|---|
| run expansion | geometric encoding expands over its exact interval | computed |
| port action | field and Nat linear-form evaluations agree | derived |
| row residual | thirteen ports evaluate the canonical selective polynomial | direct dataflow |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder

theorem modulus_eq : goldilocksP = goldilocksModulus := by
  rfl

def fieldResidue (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

theorem fieldResidue_add (left right : Nat) :
    fieldResidue (left + right) = fieldResidue left + fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_add, Nat.add_mod]

theorem fieldResidue_mul (left right : Nat) :
    fieldResidue (left * right) = fieldResidue left * fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_mul, Nat.mul_mod]

theorem fieldResidue_val (value : F) : fieldResidue value.val = value := by
  apply Fin.ext
  simp [fieldResidue, Nat.mod_eq_of_lt value.isLt]

def AssignmentCanonical (assignment : Nat → Nat) : Prop :=
  ∀ column, assignment column < goldilocksP

def fieldAssignment {columns : Nat} (assignment : Nat → Nat) :
    Fin columns → F :=
  fun column => fieldResidue (assignment column.val)

theorem fieldAssignment_val_of_canonical {columns : Nat}
    {assignment : Nat → Nat} (canonical : AssignmentCanonical assignment)
    (column : Fin columns) :
    (fieldAssignment assignment column).val = assignment column.val := by
  have bound : assignment column.val < goldilocksModulus := by
    simpa [← modulus_eq] using canonical column.val
  exact Nat.mod_eq_of_lt bound

def termAsFieldTerm {columns : Nat} (term : DecodedTerm columns) :
    Fin columns × F :=
  (term.column, term.coefficient)

def termAsNatTerm {columns : Nat} (term : DecodedTerm columns) :
    Nat × Nat :=
  (term.column.val, term.coefficient.val)

/-- Lossless expansion of one compact run. Offset `i` contributes
`initial * ratio^i` at `columnStart + i`. -/
def expandedRunFieldTerms {columns : Nat}
    (run : DecodedGeometricRun columns) : List (Fin columns × F) :=
  (List.finRange run.length).map fun offset =>
    (run.column offset, run.initial * run.ratio ^ offset.val)

def expandedRunNatTerms {columns : Nat}
    (run : DecodedGeometricRun columns) : List (Nat × Nat) :=
  (expandedRunFieldTerms run).map fun term => (term.1.val, term.2.val)

def expandedFieldTerms {columns : Nat}
    (port : DecodedPort columns) : List (Fin columns × F) :=
  port.explicit.map termAsFieldTerm ++
    port.geometric.flatMap expandedRunFieldTerms

def expandedNatTerms {columns : Nat}
    (port : DecodedPort columns) : List (Nat × Nat) :=
  (expandedFieldTerms port).map fun term => (term.1.val, term.2.val)

/-- Direct field action of the exact expanded compact contribution stream. -/
def action {columns : Nat} (port : DecodedPort columns)
    (assignment : Fin columns → F) : F :=
  (expandedFieldTerms port).foldl
    (fun total term => total + term.2 * assignment term.1) 0

/-- The same exact stream in the repository's executable Nat R1CS carrier. -/
def natAction {columns : Nat} (port : DecodedPort columns)
    (assignment : Nat → Nat) : Nat :=
  lcEval assignment (expandedNatTerms port)

theorem natAction_eq_lcEval {columns : Nat} (port : DecodedPort columns)
    (assignment : Nat → Nat) :
    natAction port assignment = lcEval assignment (expandedNatTerms port) :=
  rfl

private theorem foldl_field_eq_residue {columns : Nat}
    (terms : List (Fin columns × F)) (assignment : Nat → Nat)
    (initial : Nat) :
    terms.foldl
        (fun total term =>
          total + term.2 * fieldAssignment assignment term.1)
        (fieldResidue initial) =
      fieldResidue
        ((terms.map fun term => (term.1.val, term.2.val)).foldl
          (fun total term => total + term.2 * assignment term.1) initial) := by
  induction terms generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl, List.map_cons]
      have accumulated :
          fieldResidue initial +
              head.2 * fieldAssignment assignment head.1 =
            fieldResidue
              (initial + head.2.val * assignment head.1.val) := by
        calc
          fieldResidue initial +
                head.2 * fieldAssignment assignment head.1 =
              fieldResidue initial +
                fieldResidue head.2.val *
                  fieldResidue (assignment head.1.val) := by
            rw [fieldResidue_val]
            rfl
          _ = fieldResidue
                (initial + head.2.val * assignment head.1.val) := by
            rw [← fieldResidue_mul, ← fieldResidue_add]
      rw [accumulated]
      exact inductionHypothesis
        (initial + head.2.val * assignment head.1.val)

/-- The field and executable Nat actions are the same modular linear form. -/
theorem action_fieldAssignment_eq_natAction {columns : Nat}
    (port : DecodedPort columns) (assignment : Nat → Nat) :
    action port (fieldAssignment assignment) =
      fieldResidue (natAction port assignment) := by
  have exactFold := foldl_field_eq_residue
    (expandedFieldTerms port) assignment 0
  simpa [action, natAction, expandedNatTerms, lcEval, fieldResidue,
    modulus_eq, Nat.mod_mod] using exactFold

def rowPoint (row : DecodedRow)
    (assignment : Fin row.columns → F) : Fin 13 → F :=
  fun port => action (row.port port) assignment

def natRowPoint (row : DecodedRow)
    (assignment : Nat → Nat) : Fin 13 → F :=
  fun port => fieldResidue (natAction (row.port port) assignment)

def residual (row : DecodedRow)
    (assignment : Fin row.columns → F) : F :=
  evaluate (rowPoint row assignment)

def natResidual (row : DecodedRow)
    (assignment : Nat → Nat) : F :=
  evaluate (natRowPoint row assignment)

theorem rowPoint_fieldAssignment_eq_natRowPoint (row : DecodedRow)
    (assignment : Nat → Nat) :
    rowPoint row (fieldAssignment assignment) =
      natRowPoint row assignment := by
  funext port
  exact action_fieldAssignment_eq_natAction (row.port port) assignment

theorem residual_fieldAssignment_eq_natResidual (row : DecodedRow)
    (assignment : Nat → Nat) :
    residual row (fieldAssignment assignment) = natResidual row assignment := by
  rw [residual, natResidual, rowPoint_fieldAssignment_eq_natRowPoint]

def RowSatisfied (row : DecodedRow)
    (assignment : Fin row.columns → F) : Prop :=
  residual row assignment = 0

def NatRowSatisfied (row : DecodedRow)
    (assignment : Nat → Nat) : Prop :=
  natResidual row assignment = 0

theorem rowSatisfied_fieldAssignment_iff (row : DecodedRow)
    (assignment : Nat → Nat) :
    RowSatisfied row (fieldAssignment assignment) ↔
      NatRowSatisfied row assignment := by
  rw [RowSatisfied, NatRowSatisfied,
    residual_fieldAssignment_eq_natResidual]

def RowsSatisfied (rows : List DecodedRow)
    (assignment : Nat → Nat) : Prop :=
  ∀ row ∈ rows, NatRowSatisfied row assignment

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
