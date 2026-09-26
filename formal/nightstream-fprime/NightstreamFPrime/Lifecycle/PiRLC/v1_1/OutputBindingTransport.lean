import NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding

/-! Transport the zero-allocation PiRLC output across environments that agree
on its input expressions. No child operation or row list is unfolded. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding

open NightstreamFPrime.Circuit NightstreamFPrime.Circuit.Quadratic NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

structure InputsBelow (interface : Interface logicalWidth publicFits) (offset bound : Nat) : Prop where
  point : ∀ index, (interface.point offset index).VarsBelow bound
  commitment : ∀ row lane, (interface.commitment offset row lane).VarsBelow bound
  publicInput : ∀ column, (interface.publicInput offset column).VarsBelow bound
  eval_K : ∀ coefficient, (interface.eval_K offset coefficient).VarsBelow bound
  eval_A : ∀ matrix coefficient, (interface.eval_A offset matrix coefficient).VarsBelow bound

private theorem point_ext (left right : PaperAlgebra.Point)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

theorem evalOutput_eq_of_agree (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset bound : Nat) (before after : Env)
    (below : InputsBelow interface offset bound)
    (agrees : ∀ index, index < bound → before index = after index) :
    evalOutput relation interface offset before = evalOutput relation interface offset after := by
  unfold evalOutput
  congr 1
  · funext row lane
    exact Expr.eval_eq_of_agree_below _ bound before after (below.commitment row lane) agrees
  · funext column
    exact Expr.eval_eq_of_agree_below _ bound before after (below.publicInput column) agrees
  · apply point_ext
    change List.ofFn (fun index => (interface.point offset index).eval before) =
      List.ofFn (fun index => (interface.point offset index).eval after)
    apply congrArg List.ofFn
    funext index
    exact KExpr.eval_eq_of_agree_below _ bound before after (below.point index) agrees
  · apply congrArg (fun value : PaperAlgebra.Evaluation => #[value])
    unfold evalEvaluation
    congr 1
    · funext coefficient
      exact KExpr.eval_eq_of_agree_below _ bound before after (below.eval_K coefficient) agrees
    · funext matrix coefficient
      exact KExpr.eval_eq_of_agree_below _ bound before after (below.eval_A matrix coefficient) agrees

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding
