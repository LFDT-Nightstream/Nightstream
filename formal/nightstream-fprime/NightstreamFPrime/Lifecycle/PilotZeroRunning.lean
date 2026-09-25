import NightstreamFPrime.Lifecycle.Relation

/-!
Owns the zero-opening proof for the existing pilot running state. SuperNeo
v1.1 Definition 20 permits this bounded CE input for every verifier-owned
matrix family and Ajtai key. No circuit, relation, or key is defined here.
-/

namespace NightstreamFPrime.Lifecycle.PilotZeroRunning

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- Every one of the 16 default running claims has the zero `CE(b)` opening
under the same relation and key used by the production verifier. -/
theorem defaultRunning_holds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ∀ index : Fin productionShape.runningCount,
      CE.Holds (semantics ajtai) productionGlobalParams
        (runningStatement relation defaultRunning index)
        BaseLinear.assignmentZero := by
  intro index
  refine ⟨⟨?_, rfl, ?_⟩, trivial, ?_⟩
  · exact Phi81Relation.PiRLCAlgebra.Commitment.commit_zero ajtai
  · change ∀ column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth),
      centeredMagnitude (0 : F) < 2
    intro column
    simp
  · let source := (runningStatement relation defaultRunning index).constraintSystem
    change #[evaluationFamily (publicFits := publicFits) source
        BaseLinear.assignmentZero zeroPoint] = #[evaluationZero]
    apply congrArg (fun value : PaperAlgebra.Evaluation => #[value])
    have padZero :
        padEvaluation (publicFits := publicFits) source
          BaseLinear.assignmentZero zeroPoint = BaseLinear.evaluationZero :=
      PiRLC.ExplicitMatrix.evaluate_zero
        (canonicalStructure (publicFits := publicFits) source)
        (padMatrix source) zeroPoint
    have matrixZero :
        (fun matrix => Phi81Relation.matrixEvaluation
          (canonicalStructure (publicFits := publicFits) source)
          BaseLinear.assignmentZero zeroPoint matrix) =
        (fun _ => BaseLinear.evaluationZero) := by
      funext matrix
      exact BaseLinear.matrixEvaluation_zero
        (canonicalStructure (publicFits := publicFits) source)
        zeroPoint matrix
    unfold evaluationFamily evaluationZero
    rw [padZero]
    exact congrArg
      (fun matrix => ({ pad := BaseLinear.evaluationZero, matrix := matrix } :
        PaperAlgebra.Evaluation)) matrixZero

end NightstreamFPrime.Lifecycle.PilotZeroRunning
