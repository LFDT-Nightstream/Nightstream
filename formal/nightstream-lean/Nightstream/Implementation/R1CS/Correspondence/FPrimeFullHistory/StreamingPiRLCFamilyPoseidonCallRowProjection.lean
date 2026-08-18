import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallPermutation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallProjection

/-!
Contract: structural same-assignment evaluation bridge for one Rust-emitted
relative PiRLC Poseidon2 row.

Assurance tier: artifact-checked row interpretation.

Owns: equality between the absolute production-column action described by a
relative row and that row's leaf-model action under the exact call projection.
The proof is generic in the row and does not reduce a generated row list.

Does not own: Rust row selection, generated-row identity, row satisfaction,
call-class coverage, family-phase semantics, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallPermutation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection

def absoluteExplicitValue (kind : LeafClass)
    (assignment : Fin productionFinalColumns → F) : ExplicitColumn → F
  | .one => absoluteValue assignment 0
  | .selector => absoluteValue assignment (selectorColumn kind)

def absoluteDigitValue (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (slot : Slot) (digit : Fin 41) : F :=
  match digitColumn kind index slot digit with
  | some column => absoluteValue assignment column
  | none => 0

def absoluteExplicitAction (kind : LeafClass)
    (assignment : Fin productionFinalColumns → F)
    (terms : List ExplicitTerm) : F :=
  sum (terms.map fun term =>
    term.coefficient * absoluteExplicitValue kind assignment term.column)

def absoluteGeometricAction (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (run : GeometricRun) : F :=
  sum (List.ofFn fun digit : Fin 41 =>
    geometricCoefficient run.initial run.ratio digit.val *
      absoluteDigitValue kind index assignment run.slot digit)

def absolutePortAction (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (port : Port) : F :=
  absoluteExplicitAction kind assignment port.explicit +
    sum (port.geometric.map fun run =>
      absoluteGeometricAction kind index assignment run)

def absolutePoint (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (row : Wire.Row) : Fin 13 → F :=
  fun port => absolutePortAction kind index assignment (row.port port)

def absoluteResidual (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (row : Wire.Row) : F :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
    (absolutePoint kind index assignment row)

theorem absoluteExplicitValue_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (column : ExplicitColumn) :
    absoluteExplicitValue kind assignment column =
      (projectFinalAssignment kind index assignment).explicit column := by
  cases column <;> rfl

theorem absoluteDigitValue_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (slot : Slot) (digit : Fin 41) :
    absoluteDigitValue kind index assignment slot digit =
      (projectFinalAssignment kind index assignment).digit slot digit := by
  rfl

theorem absoluteExplicitAction_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (terms : List ExplicitTerm) :
    absoluteExplicitAction kind assignment terms =
      sum (terms.map fun term =>
        term.coefficient *
          (projectFinalAssignment kind index assignment).explicit
            term.column) := by
  unfold absoluteExplicitAction
  induction terms with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [absoluteExplicitValue_eq_projected, inductionHypothesis]

theorem absoluteGeometricAction_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (run : GeometricRun) :
    absoluteGeometricAction kind index assignment run =
      geometricAction run (projectFinalAssignment kind index assignment) := by
  unfold absoluteGeometricAction geometricAction
  congr 1

theorem absolutePortAction_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (port : Port) :
    absolutePortAction kind index assignment port =
      portAction port (projectFinalAssignment kind index assignment) := by
  unfold absolutePortAction portAction
  rw [absoluteExplicitAction_eq_projected]
  congr 1

theorem absolutePoint_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (row : Wire.Row) :
    absolutePoint kind index assignment row =
      point row (projectFinalAssignment kind index assignment) := by
  funext port
  exact absolutePortAction_eq_projected kind index assignment (row.port port)

theorem absoluteResidual_eq_projected (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (row : Wire.Row) :
    absoluteResidual kind index assignment row =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        row (projectFinalAssignment kind index assignment) := by
  unfold absoluteResidual
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
  rw [absolutePoint_eq_projected]

theorem absoluteResidual_eq_of_perm
    (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    {left right : Wire.Row} (permutation : RowPermutes left right) :
    absoluteResidual kind index assignment left =
      absoluteResidual kind index assignment right := by
  rw [absoluteResidual_eq_projected, absoluteResidual_eq_projected]
  exact residual_eq_of_perm permutation
    (projectFinalAssignment kind index assignment)

theorem absolute_rows_imply_projected_rows
    (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (rows : List Wire.Row)
    (holds : ∀ row ∈ rows, absoluteResidual kind index assignment row = 0) :
    ∀ row ∈ rows,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        row (projectFinalAssignment kind index assignment) = 0 := by
  intro row member
  rw [← absoluteResidual_eq_projected]
  exact holds row member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
