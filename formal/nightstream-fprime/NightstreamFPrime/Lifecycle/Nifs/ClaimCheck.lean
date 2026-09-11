import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck

/-!
Constructive CE membership for the NIFS extractor's supplied assignments.
The existing commitment, public prefix, strict stage norm, Pad and all matrix
evaluations are checked. Row evaluation reuses the streaming witness-check
operations; no Boolean table or column-index list is materialized. This is
an extractor check, with no runtime or cryptographic security bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.ClaimCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra StrongReduction ConcreteCarrier
open PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stream the same normalized matrix family used by the CE semantics. -/
def evaluation (source : Structure logicalWidth)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) : PaperAlgebra.Evaluation :=
  let stored := Vector.ofFn assignment
  let system := canonicalStructure (publicFits := publicFits) source
  { pad := fun coefficient =>
      StoredWitnessCheck.evaluateRows (fun vertex => K.embed
        (StoredWitnessCheck.matrixRow
          (system.matrixSource.coefficientMatrixOf baseOps (padMatrix source) coefficient)
          stored vertex)) point.coordinates
    matrix := fun matrix coefficient =>
      StoredWitnessCheck.evaluateRows (fun vertex => K.embed
        (StoredWitnessCheck.matrixRow
          (system.matrixSource.coefficientMatrix baseOps matrix coefficient)
          stored vertex)) point.coordinates }

/-- The streamed value is exactly the selected semantics' full evaluation. -/
theorem evaluation_eq (source : Structure logicalWidth)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) :
    evaluation source assignment point = evaluationFamily source assignment point := by
  have stored : (Vector.ofFn assignment).get = assignment := by
    funext column
    simp [Vector.get]
  apply congrArg₂ EvaluationFamily.mk
  · funext coefficient
    simp only [evaluation, evaluationFamily, padEvaluation,
      Phi81Relation.EvaluationHomomorphism.PiRLC.ExplicitMatrix.evaluate,
      Phi81Relation.matrixEvaluation, Phi81Evaluation.evaluate, Phi81Evaluation.table,
      BooleanTable.evaluate, StoredWitnessCheck.evaluateRows_eq,
      StoredWitnessCheck.matrixRow_eq, stored]
  · funext matrix coefficient
    simp only [evaluation, evaluationFamily, padEvaluation,
      Phi81Relation.EvaluationHomomorphism.PiRLC.ExplicitMatrix.evaluate,
      Phi81Relation.matrixEvaluation, Phi81Evaluation.evaluate, Phi81Evaluation.table,
      BooleanTable.evaluate, StoredWitnessCheck.evaluateRows_eq,
      StoredWitnessCheck.matrixRow_eq, stored]
    rfl

/-- Check the complete CE opening at its own stage bound and exact point. -/
def check
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (claim : CE.Instance (Structure logicalWidth)
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) : Bool :=
  letI : DecidableEq RingF := Fintype.decidablePiFintype
  letI : DecidableEq PaperAlgebra.Commitment := Fintype.decidablePiFintype
  letI : DecidableEq PaperAlgebra.Evaluation := evaluationDecidableEq
  decide ((semantics ajtai).commit assignment = claim.commitment) &&
    (decide (Phi81Relation.projectPublicInput assignment = claim.publicInput) &&
      (StoredWitnessCheck.allFin (fun column =>
        decide (centeredMagnitude (assignment column) < claim.stage.bound productionGlobalParams)) &&
        decide (#[evaluation claim.constraintSystem assignment claim.point] = claim.evaluations)))

/-- Acceptance checks every CE field; no opening-validity premise is used. -/
theorem check_eq_true_iff
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (claim : CE.Instance (Structure logicalWidth)
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    check ajtai claim assignment = true ↔
      CE.Holds (semantics ajtai) productionGlobalParams claim assignment := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq,
    StoredWitnessCheck.allFin_eq_true, evaluation_eq,
    CE.Holds, Opening.Holds, semantics, Phi81Relation.assignmentNormBounded,
    true_and, and_assoc]

end NightstreamFPrime.Lifecycle.Nifs.ClaimCheck
