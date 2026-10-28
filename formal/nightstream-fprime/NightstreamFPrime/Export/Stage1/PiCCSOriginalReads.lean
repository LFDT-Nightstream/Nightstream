import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation

/-!
Original signed-mask assignment, complete blocks and prepared coefficient reads.
The caller loads and validates masks and shares prepared basis tables. No digit
split, source support shortcut or preservation/reference-plan import belongs
here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalReads

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

def assignment (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) :
    Phi81Relation.Assignment PiCCSSourceImages.shape :=
  fun column => SignedUnitSourceInput.scalar
    (masks[column.val / ringDegree]?.getD #[])
    ⟨source.val, source.isLt⟩ ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

def blockAt (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Nat) : RingF :=
  PiCCSSourceImages.blockAt (assignment masks source) block

def maskEval (form : SparseForm ringDegree) (positive negative : Nat) : F :=
  PiDECNativeSparseEvaluation.nativeEvalSparse form fun lane =>
    if positive.testBit lane.val then 1
    else if negative.testBit lane.val then -1 else 0

def read {columns : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (output : Fin ringDegree) (column : Fin columns) : F :=
  let block := column.val / ringDegree
  if block < PiCCSSourceImages.blockCount then
    let basis : Fin ringDegree := ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
    let blockMasks := masks[block]?.getD #[]
    let pair := blockMasks[source.val]?.getD (0, 0)
    maskEval ((tables.get basis).get output) pair.1 pair.2
  else 0

end NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
