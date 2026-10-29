import NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedPrefix
import NightstreamFPrime.Export.Stage1.PiDECEvaluationWeights
import NightstreamFPrime.Export.Stage1.PiDECStoredSplitHonestWitness
import NightstreamFPrime.Export.Stage1.PiDECInputCheck

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment view)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev selectedPlan := PerApplicationFixedPoint.structuralPlan
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

def family
    (assignments : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (point : PaperAlgebra.Point) (child : Fin productionGlobalParams.k) :
    PaperAlgebra.Evaluation :=
  let pad := PiDECEvaluationWeights.accumulate selectedShape.carrierWidth point
    (PiDECEvaluationSelectedPrefix.padRow assignments child)
  let matrices := PiRLCPartialTrace.FixedArray.ofFn
    (fun matrix : Fin productionShape.matrixCount =>
      PiDECEvaluationWeights.accumulate selectedPlan.rowCount point
        (PiDECEvaluationSelectedPrefix.matrixRow assignments child matrix))
  { pad := pad.toRing
    matrix := fun matrix => (matrices.get matrix).toRing }

private theorem evaluationFamily_rows {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (assignment : Phi81Relation.Assignment
      (PaperAlgebra.FullShape logicalWidth publicFits))
    (point : PaperAlgebra.Point) :
    PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation) assignment point =
      ({ pad := fun lane =>
          (BooleanTable.tabulate (fun vertex => K.embed
            (PiRLC.ExplicitMatrix.rowRing relation.system
              (PaperAlgebra.padMatrix
                (Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation))
              assignment vertex lane))).evaluate extensionOps point
         matrix := fun matrix lane =>
          (BooleanTable.tabulate (fun vertex => K.embed
            (PiRLC.rowRing relation.system assignment matrix vertex lane))).evaluate
            extensionOps point } : PaperAlgebra.Evaluation) := by
  exact PaperAlgebra.evaluationFamily_eq_paper
    (Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation).cubeLayout
    relation.system assignment point

theorem family_eq_evaluationFamily
    (assignments : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (point : PaperAlgebra.Point) (child : Fin productionGlobalParams.k) :
    family assignments point child =
      PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_2.InputBinding.relationSource PiDECInputCheck.relation)
        (view (assignments.get child)) point := by
  rw [PiDECInputCheck.relation_eq_selected,
    evaluationFamily_rows selectedRelation (view (assignments.get child)) point]
  simp only [family, PiRLCPartialTrace.FixedArray.get_ofFn]
  apply congrArg₂ (@StrongReduction.EvaluationFamily.mk K productionShape)
  · funext lane
    rw [PiDECEvaluationWeights.accumulate_prefix_eq_evaluate
      selectedShape.carrierWidth point (PiDECEvaluationSelectedPrefix.padRow assignments child)
      Poseidon2HashChainV1Package.fits.carrier
      (fun index outside _ =>
        PiDECEvaluationSelectedPrefix.padRow_outside assignments child index outside) lane]
    apply congrArg (fun values : BooleanVertex cubeVariables → K =>
      (BooleanTable.tabulate values).evaluate extensionOps point)
    funext vertex
    exact congrArg (fun value : RingF => K.embed (value lane))
      (PiDECEvaluationSelectedPrefix.padRow_value assignments child vertex)
  · funext matrix lane
    rw [PiDECEvaluationWeights.accumulate_prefix_eq_evaluate
      selectedPlan.rowCount point (PiDECEvaluationSelectedPrefix.matrixRow assignments child matrix)
      selectedPlan.rowCount_le
      (fun index outside _ =>
        PiDECEvaluationSelectedPrefix.matrixRow_outside assignments child matrix index outside) lane]
    apply congrArg (fun values : BooleanVertex cubeVariables → K =>
      (BooleanTable.tabulate values).evaluate extensionOps point)
    funext vertex
    exact congrArg (fun value : RingF => K.embed (value lane))
      (PiDECEvaluationSelectedPrefix.matrixRow_value assignments child matrix vertex)

theorem family_honestMessages
    (values : PiDECInputCheck.ParentValues)
    (parentWitness : StoredAssignment selectedShape.carrierWidth)
    (childWitnesses : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (success : StoredSplit.splitChecked parentWitness = some childWitnesses)
    (child : Fin productionGlobalParams.k) :
    #[family childWitnesses values.point child] =
      (PiDEC.PaperVerifier.honestMessages
        (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
        (PiDECInputCheck.parent values) (view parentWitness) child).evaluations := by
  have honest := (PiDECStoredSplitHonestWitness.splitChecked_honestWitness
    (PiDECInputCheck.parent values) parentWitness childWitnesses success child).2.1
  rw [family_eq_evaluationFamily childWitnesses values.point child]
  simpa only [PaperAlgebra.semantics, PiDECInputCheck.parent] using
    (congrArg (fun message : PiDEC.PaperVerifier.ChildMessage
      PaperAlgebra.Evaluation PaperAlgebra.Commitment => message.evaluations) honest).symm

end NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages
