import NightstreamFPrime.Gadgets.Range.CanonicalU64

/-!
Child-owned witness IR contract for `CanonicalU64`.

This companion module can inspect the child implementation. Semantic and
layout parents import only the core circuit and do not depend on this export.
-/

namespace NightstreamFPrime.Gadgets.Range.CanonicalU64

open NightstreamFPrime.Circuit

def witnessBatches (interface : Interface) (offset : Nat) : List WitnessBatch :=
  [ WitnessBatch.hinted offset (bitHints interface offset),
    WitnessBatch.hinted (offset + bitCount) [inverseHint offset],
    WitnessBatch.arithmetic (offset + bitCount + 1) [flagRecipe offset] ]

@[simp] theorem witnesses_main (interface : Interface) (offset : Nat) :
    witnesses (Circuit.ops (main interface) offset) =
      witnessBatches interface offset := by
  change witnesses (operations interface offset) = _
  simp [operations, witnessBatches, witnesses, Op.witnesses, booleanOps]

end NightstreamFPrime.Gadgets.Range.CanonicalU64
