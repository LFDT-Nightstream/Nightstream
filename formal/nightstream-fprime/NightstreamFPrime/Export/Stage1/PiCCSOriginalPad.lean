import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Export.Stage1.PiCCSTensorWeights

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalPad

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

def blockValues (masks : Array (Array (Nat × Nat))) (block : Nat) :
    Vector StoredRing productionShape.sourceCount :=
  Vector.ofFn fun source => Vector.ofFn (PiCCSOriginalReads.blockAt masks source block)

def weights (point : PaperAlgebra.Point) (block : Nat) : Vector K ringDegree :=
  Vector.ofFn fun basis =>
    PiDECEvaluationWeights.weight point (block * ringDegree + basis.val)

def products (point : PaperAlgebra.Point) (masks : Array (Array (Nat × Nat)))
    (block : Nat) : Vector MaterializedRingK productionShape.sourceCount :=
  if (masks[block]?.getD #[]).isEmpty then PiDECEvaluationBatch.zero productionShape.sourceCount
  else PiDECPadWeightedProduct.products (weights point block) (blockValues masks block)

def range (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) :
    Vector MaterializedRingK productionShape.sourceCount :=
  let tables := PiCCSTensorWeights.prepare extensionOps point.coordinates
  PiDECEvaluationBatch.sum count fun offset =>
    let block := firstBlock + offset
    if (masks[block]?.getD #[]).isEmpty then
      PiDECEvaluationBatch.zero productionShape.sourceCount
    else
      let blockWeights := Vector.ofFn fun basis : Fin ringDegree =>
        PiCCSTensorWeights.lookup extensionOps point.coordinates tables
          (block * ringDegree + basis.val)
      PiDECPadWeightedProduct.products blockWeights (blockValues masks block)

theorem range_eq_sum (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) :
    range firstBlock count point masks =
      PiDECEvaluationBatch.sum count (fun offset => products point masks (firstBlock + offset)) := by
  simp only [range, PiCCSTensorWeights.lookup_prepare extensionOps
    (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws extensionLaws)]
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSOriginalPad
