import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation

/-!
Original signed-mask assignment, complete blocks and prepared coefficient reads.
Source indices remain unchanged: fresh 0 and running 1 through 16. The caller
loads and validates masks and shares prepared basis tables. No digit split,
source support shortcut or preservation/reference-plan import belongs here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalReads

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Every complete-carrier column reads its original block and lane. -/
def assignment (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) :
    Phi81Relation.Assignment PiCCSSourceImages.shape :=
  fun column => SignedUnitSourceInput.scalar
    (masks[column.val / ringDegree]?.getD #[])
    ⟨source.val, source.isLt⟩ ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

/-- The existing complete-block reader retains every original carrier tail. -/
def blockAt (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Nat) : RingF :=
  PiCCSSourceImages.blockAt (assignment masks source) block

/-- Evaluate signed masks with the existing native-word sparse accumulator.
Positive bits retain priority even when the supplied masks overlap. -/
def maskEval (form : SparseForm ringDegree) (positive negative : Nat) : F :=
  PiDECNativeSparseEvaluation.nativeEvalSparse form fun lane =>
    if positive.testBit lane.val then 1
    else if negative.testBit lane.val then -1 else 0

/-- Select one original mask pair and evaluate its prepared basis form.
The complete-carrier guard preserves zero reads for arbitrary outside columns. -/
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
