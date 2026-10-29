import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixPreservation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (view)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedRelationSource :=
  Lifecycle.PiRLC.v1_2.InputBinding.relationSource selectedRelation

private def stored {columns : Nat} (assignment : Fin columns → F) :
    Vector (Vector F columns) productionGlobalParams.k :=
  Vector.replicate productionGlobalParams.k (Vector.ofFn assignment)

private theorem stored_get {columns : Nat} (assignment : Fin columns → F)
    (child : Fin productionGlobalParams.k) :
    ((stored assignment).get child).get = assignment := by
  funext column
  change ((Vector.replicate productionGlobalParams.k (Vector.ofFn assignment))[child.val])[column.val] = _
  rw [Vector.getElem_replicate, Vector.getElem_ofFn]

def matrix (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) : MaterializedRingK :=
  MaterializedRingK.ofRing
    ((PiDECEvaluationHonestMessages.family
      (stored (PiCCSOriginalReads.assignment masks source)) point ⟨0, by decide⟩).matrix port)

theorem matrix_eq_evaluationFamily (masks : Array (Array (Nat × Nat)))
    (point : PaperAlgebra.Point) (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    (matrix masks point source port).toRing =
      (PaperAlgebra.evaluationFamily selectedRelationSource
        (PiCCSOriginalReads.assignment masks source) point).matrix port := by
  rw [matrix, MaterializedRingK.toRing_ofRing]
  have familyEq := PiDECEvaluationHonestMessages.family_eq_evaluationFamily
    (stored (PiCCSOriginalReads.assignment masks source)) point ⟨0, by decide⟩
  rw [view, stored_get, PiDECInputCheck.relation_eq_selected] at familyEq
  exact congrArg (fun value : PaperAlgebra.Evaluation => value.matrix port) familyEq

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixPreservation
