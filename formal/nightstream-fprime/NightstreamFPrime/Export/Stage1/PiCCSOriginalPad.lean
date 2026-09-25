import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch

/-! Complete original-source Pad block ranges. Each pair of weighted bar
keys is shared by all 17 original blocks. No split or source omission is used.
Reference assignments and full-family transport belong to the proof module. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalPad

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- All original source blocks at one complete-carrier block index. -/
def blockValues (masks : Array (Array (Nat × Nat))) (block : Nat) :
    Vector StoredRing productionShape.sourceCount :=
  Vector.ofFn fun source => Vector.ofFn (PiCCSOriginalReads.blockAt masks source block)

/-- Pad basis weights retain the absolute scalar row index. -/
def weights (point : PaperAlgebra.Point) (block : Nat) : Vector K ringDegree :=
  Vector.ofFn fun basis =>
    PiDECEvaluationWeights.weight point (block * ringDegree + basis.val)

/-- An omitted whole block contributes zero before weight and key preparation.
Otherwise one pair of weighted products contributes all source coefficients. -/
def products (point : PaperAlgebra.Point) (masks : Array (Array (Nat × Nat)))
    (block : Nat) : Vector MaterializedRingK productionShape.sourceCount :=
  if (masks[block]?.getD #[]).isEmpty then PiDECEvaluationBatch.zero productionShape.sourceCount
  else PiDECPadWeightedProduct.products (weights point block) (blockValues masks block)

/-- Sum complete blocks in [firstBlock, firstBlock + count), in their original
order. The complete-block reader defines positions beyond the carrier as zero. -/
def range (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) :
    Vector MaterializedRingK productionShape.sourceCount :=
  PiDECEvaluationBatch.sum count fun offset => products point masks (firstBlock + offset)

end NightstreamFPrime.Export.Stage1.PiCCSOriginalPad
