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

def absoluteExplicitValue (site : CallSite)
    (assignment : Fin productionFinalColumns → F) : ExplicitColumn → F
  | .one => absoluteValue assignment 0
  | .selector => absoluteValue assignment (selectorColumn site.kind)

def absoluteDigitValue (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (slot : Slot) (digit : Fin 41) : F :=
  match digitColumn site slot digit with
  | some column => absoluteValue assignment column
  | none => 0

def absoluteExplicitAction (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (terms : List ExplicitTerm) : F :=
  sum (terms.map fun term =>
    term.coefficient * absoluteExplicitValue site assignment term.column)

def absoluteGeometricAction (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (run : GeometricRun) : F :=
  sum (List.ofFn fun digit : Fin 41 =>
    geometricCoefficient run.initial run.ratio digit.val *
      absoluteDigitValue site assignment run.slot digit)

def absolutePortAction (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (port : Port) : F :=
  absoluteExplicitAction site assignment port.explicit +
    sum (port.geometric.map fun run =>
      absoluteGeometricAction site assignment run)

def absolutePoint (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (row : Wire.Row) : Fin 13 → F :=
  fun port => absolutePortAction site assignment (row.port port)

def absoluteResidual (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (row : Wire.Row) : F :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
    (absolutePoint site assignment row)

theorem absoluteExplicitValue_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (column : ExplicitColumn) :
    absoluteExplicitValue site assignment column =
      (projectFinalAssignment site assignment).explicit column := by
  cases column <;> rfl

theorem absoluteDigitValue_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (slot : Slot) (digit : Fin 41) :
    absoluteDigitValue site assignment slot digit =
      (projectFinalAssignment site assignment).digit slot digit := by
  rfl

theorem absoluteExplicitAction_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (terms : List ExplicitTerm) :
    absoluteExplicitAction site assignment terms =
      sum (terms.map fun term =>
        term.coefficient *
          (projectFinalAssignment site assignment).explicit
            term.column) := by
  unfold absoluteExplicitAction
  induction terms with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [absoluteExplicitValue_eq_projected, inductionHypothesis]

theorem absoluteGeometricAction_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    (run : GeometricRun) :
    absoluteGeometricAction site assignment run =
      geometricAction run (projectFinalAssignment site assignment) := by
  unfold absoluteGeometricAction geometricAction
  congr 1

theorem absolutePortAction_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (port : Port) :
    absolutePortAction site assignment port =
      portAction port (projectFinalAssignment site assignment) := by
  unfold absolutePortAction portAction
  rw [absoluteExplicitAction_eq_projected]
  congr 1

theorem absolutePoint_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (row : Wire.Row) :
    absolutePoint site assignment row =
      point row (projectFinalAssignment site assignment) := by
  funext port
  exact absolutePortAction_eq_projected site assignment (row.port port)

theorem absoluteResidual_eq_projected (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (row : Wire.Row) :
    absoluteResidual site assignment row =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        row (projectFinalAssignment site assignment) := by
  unfold absoluteResidual
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
  rw [absolutePoint_eq_projected]

theorem absoluteResidual_eq_of_perm
    (site : CallSite)
    (assignment : Fin productionFinalColumns → F)
    {left right : Wire.Row} (permutation : RowPermutes left right) :
    absoluteResidual site assignment left =
      absoluteResidual site assignment right := by
  rw [absoluteResidual_eq_projected, absoluteResidual_eq_projected]
  exact residual_eq_of_perm permutation
    (projectFinalAssignment site assignment)

theorem absolute_rows_imply_projected_rows
    (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (rows : List Wire.Row)
    (holds : ∀ row ∈ rows, absoluteResidual site assignment row = 0) :
    ∀ row ∈ rows,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        row (projectFinalAssignment site assignment) = 0 := by
  intro row member
  rw [← absoluteResidual_eq_projected]
  exact holds row member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
