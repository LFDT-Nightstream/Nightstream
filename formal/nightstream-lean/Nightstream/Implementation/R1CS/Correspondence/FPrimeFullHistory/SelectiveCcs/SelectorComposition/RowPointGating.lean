import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.PolynomialGating

/-!
Contract: connect selector-port matrix actions to factorization of one
interpreted selective-CCS row residual.

Owns: the exact composition from the two physical selector matrix images,
through the relation-owned thirteen-port row point, to the independent
74-term selector-factorization theorem.

Does not own: a concrete matrix artifact, proof that an emitted row has the
required selector images, arithmetic-family classification, branch
semantics, constraint necessity, or row removal.

Emits constraints: no.

Authority boundary: both selector image equations are explicit premises over
`matrixImageAt`. A generated gate label or coverage interval cannot discharge
them without a separate matrix-action refinement theorem.

| Stage path | Physical premises | Exact result |
|---|---|---|
| `f_prime.selective_ccs.row.gating.general` | `G(row)=weight`, `E(row)=0` | residual is `weight` times the general-ungated residual |
| `f_prime.selective_ccs.row.gating.evaluation` | `G(row)=0`, `E(row)=weight` | residual is `weight` times the evaluation-ungated residual |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating

/-- A row whose general selector matrix evaluates to `weight` and whose
evaluation selector matrix evaluates to zero factors exactly through the
general gate. -/
theorem residualAt_general_gated
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows) (weight : F)
    (general :
      matrixImageAt relation assignment row Role.generalSelector.index =
        weight)
    (evaluation :
      matrixImageAt relation assignment row Role.evalSelector.index = 0) :
    residualAt relation assignment row =
      weight * evaluate
        (ungate .general (rowPoint relation assignment row)) := by
  rw [residualAt_eq_evaluate]
  apply evaluate_general_gated
  · exact general
  · exact evaluation

/-- A row whose general selector matrix evaluates to zero and whose
evaluation selector matrix evaluates to `weight` factors exactly through the
evaluation gate. -/
theorem residualAt_evaluation_gated
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows) (weight : F)
    (general :
      matrixImageAt relation assignment row Role.generalSelector.index = 0)
    (evaluation :
      matrixImageAt relation assignment row Role.evalSelector.index =
        weight) :
    residualAt relation assignment row =
      weight * evaluate
        (ungate .evaluation (rowPoint relation assignment row)) := by
  rw [residualAt_eq_evaluate]
  apply evaluate_evaluation_gated
  · exact general
  · exact evaluation

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating
