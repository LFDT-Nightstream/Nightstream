import NightstreamFPrime.Export.Stage1.PiCCSNormBuckets
import NightstreamFPrime.Export.Stage1.PiCCSNormSource

/-! Accumulate the first-round norm from the complete original signed masks.
Only scalar equality weights enter the existing norm buckets. Matrix images,
prover messages and claimed norm results are not inputs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormScan

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle (productionShape)
open PiCCSNormSource

/-- Visit every adjacent pair and every source of a full carrier block.
An omitted block decodes to zero in every source. -/
def block (weight : Nat → K) (index : Nat) (masks : Array (Nat × Nat))
    (initial : PiCCSNormBuckets.Buckets) : PiCCSNormBuckets.Buckets :=
  if masks.isEmpty then initial else
    Nat.fold PairCount (fun pair inside accumulated =>
      let pair : Fin PairCount := ⟨pair, inside⟩
      let pairWeight := weight (pairIndex index pair)
      Nat.fold productionShape.sourceCount (fun source inside current =>
        let source : Fin productionShape.sourceCount := ⟨source, inside⟩
        PiCCSNormBuckets.add current source
          (sourceCode masks source (lowLane pair))
          (sourceCode masks source (highLane pair)) pairWeight) accumulated) initial

/-- Accumulate an ascending block range. Missing source records retain their
exact zero meaning. No carrier tail is removed. -/
def range (weight : Nat → K) (masks : Array (Array (Nat × Nat)))
    (start count : Nat) : PiCCSNormBuckets.Buckets :=
  Nat.fold count (fun offset _ accumulated =>
    block weight (start + offset) (masks[start + offset]?.getD #[]) accumulated)
    PiCCSNormBuckets.empty

end NightstreamFPrime.Export.Stage1.PiCCSNormScan
