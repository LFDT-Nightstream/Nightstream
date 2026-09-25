import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Owns the generic source-support contract for caller-owned PiCCS expressions.

This mirrors `Formal.ExternalInputsBelow` but preserves a caller-selected
source predicate instead of only a numeric upper bound. It does not select a
physical layout, retained slots, or a production application.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Exact source-support premises for every caller-owned PiCCS family. -/
structure ExternalInputsSupported
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (allowed : Nat → Prop) : Prop where
  priorStateFixed : ∀ word ∈ StateBinding.fixedWords,
    (interface.priorState offset word.index).VarsSatisfy allowed
  outputStateFixed : ∀ word ∈ StateBinding.fixedWords,
    (interface.outputState offset word.index).VarsSatisfy allowed
  priorStateContext : ∀ lane : Fin 4,
    (interface.priorState offset
      (StateBinding.contextWordStart + lane.val)).VarsSatisfy allowed
  outputStateContext : ∀ lane : Fin 4,
    (interface.outputState offset
      (StateBinding.contextWordStart + lane.val)).VarsSatisfy allowed
  expectedContext : ∀ lane : Fin 4,
    (interface.expectedContext offset lane).VarsSatisfy allowed
  runningPoint : ∀ coordinate,
    Expr.VarsSatisfy allowed
        ((interface.running offset).point coordinate).c0 ∧
      Expr.VarsSatisfy allowed
        ((interface.running offset).point coordinate).c1
  runningCommitment : ∀ source row coefficient,
    ((interface.running offset).commitment source row coefficient).VarsSatisfy
      allowed
  runningPublicInput : ∀ source column,
    ((interface.running offset).publicInput source column).VarsSatisfy allowed
  runningEval_K : ∀ source coefficient,
    Expr.VarsSatisfy allowed
        (((interface.running offset).evaluation source).eval_K coefficient).c0 ∧
      Expr.VarsSatisfy allowed
        (((interface.running offset).evaluation source).eval_K coefficient).c1
  runningEval_A : ∀ source matrix coefficient,
    Expr.VarsSatisfy allowed
        (((interface.running offset).evaluation source).eval_A matrix
          coefficient).c0 ∧
      Expr.VarsSatisfy allowed
        (((interface.running offset).evaluation source).eval_A matrix
          coefficient).c1
  freshCommitment : ∀ source row coefficient,
    ((interface.fresh offset).commitment source row coefficient).VarsSatisfy
      allowed
  freshPublicInput : ∀ source column,
    ((interface.fresh offset).publicInput source column).VarsSatisfy allowed
  roundCoefficient : ∀ roundIndex coefficient,
    Expr.VarsSatisfy allowed
        ((interface.round offset roundIndex).coefficient coefficient).c0 ∧
      Expr.VarsSatisfy allowed
        ((interface.round offset roundIndex).coefficient coefficient).c1
  outputEval_K : ∀ source coefficient,
    Expr.VarsSatisfy allowed
        ((interface.output offset).padCoordinate source coefficient).c0 ∧
      Expr.VarsSatisfy allowed
        ((interface.output offset).padCoordinate source coefficient).c1
  outputEval_A : ∀ source matrix coefficient,
    Expr.VarsSatisfy allowed
        ((interface.output offset).matrixCoordinate source matrix coefficient).c0 ∧
      Expr.VarsSatisfy allowed
        ((interface.output offset).matrixCoordinate source matrix coefficient).c1

/-- Caller-input support remains valid when the allowed source predicate is
enlarged. -/
theorem ExternalInputsSupported.mono
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {interface : Interface logicalWidth degreeBound publicFits}
    {offset : Nat} {allowed larger : Nat → Prop}
    (support : ExternalInputsSupported interface offset allowed)
    (includes : ∀ index, allowed index → larger index) :
    ExternalInputsSupported interface offset larger := by
  refine {
    priorStateFixed := fun word member => Expr.VarsSatisfy.mono _
      (support.priorStateFixed word member) includes
    outputStateFixed := fun word member => Expr.VarsSatisfy.mono _
      (support.outputStateFixed word member) includes
    priorStateContext := fun lane => Expr.VarsSatisfy.mono _
      (support.priorStateContext lane) includes
    outputStateContext := fun lane => Expr.VarsSatisfy.mono _
      (support.outputStateContext lane) includes
    expectedContext := fun lane => Expr.VarsSatisfy.mono _
      (support.expectedContext lane) includes
    runningPoint := fun coordinate => ⟨
      Expr.VarsSatisfy.mono _ (support.runningPoint coordinate).1 includes,
      Expr.VarsSatisfy.mono _ (support.runningPoint coordinate).2 includes⟩
    runningCommitment := fun source row coefficient => Expr.VarsSatisfy.mono _
      (support.runningCommitment source row coefficient) includes
    runningPublicInput := fun source column => Expr.VarsSatisfy.mono _
      (support.runningPublicInput source column) includes
    runningEval_K := fun source coefficient => ⟨
      Expr.VarsSatisfy.mono _ (support.runningEval_K source coefficient).1
        includes,
      Expr.VarsSatisfy.mono _ (support.runningEval_K source coefficient).2
        includes⟩
    runningEval_A := fun source matrix coefficient => ⟨
      Expr.VarsSatisfy.mono _
        (support.runningEval_A source matrix coefficient).1 includes,
      Expr.VarsSatisfy.mono _
        (support.runningEval_A source matrix coefficient).2 includes⟩
    freshCommitment := fun source row coefficient => Expr.VarsSatisfy.mono _
      (support.freshCommitment source row coefficient) includes
    freshPublicInput := fun source column => Expr.VarsSatisfy.mono _
      (support.freshPublicInput source column) includes
    roundCoefficient := fun roundIndex coefficient => ⟨
      Expr.VarsSatisfy.mono _
        (support.roundCoefficient roundIndex coefficient).1 includes,
      Expr.VarsSatisfy.mono _
        (support.roundCoefficient roundIndex coefficient).2 includes⟩
    outputEval_K := fun source coefficient => ⟨
      Expr.VarsSatisfy.mono _ (support.outputEval_K source coefficient).1
        includes,
      Expr.VarsSatisfy.mono _ (support.outputEval_K source coefficient).2
        includes⟩
    outputEval_A := fun source matrix coefficient => ⟨
      Expr.VarsSatisfy.mono _
        (support.outputEval_A source matrix coefficient).1 includes,
      Expr.VarsSatisfy.mono _
        (support.outputEval_A source matrix coefficient).2 includes⟩ }

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
