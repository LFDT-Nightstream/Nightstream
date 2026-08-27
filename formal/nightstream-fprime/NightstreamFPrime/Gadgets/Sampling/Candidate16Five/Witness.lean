import NightstreamFPrime.Gadgets.Sampling.Candidate16Five

/-!
Child-owned witness IR contract for `Candidate16Five`.

This companion module can inspect the child implementation. Semantic and
layout parents import only the core circuit and do not depend on this export.
-/

namespace NightstreamFPrime.Gadgets.Sampling.Candidate16Five

open NightstreamFPrime.Circuit

def witnessBatches (interface : Interface) (offset : Nat) : List WitnessBatch :=
  [ WitnessBatch.hinted offset (quotientRemainderHints interface offset),
    WitnessBatch.hinted (offset + 2) (quotientBitHints offset),
    WitnessBatch.arithmetic (offset + 16) [rejectRecipe interface offset] ]

@[simp] theorem witnesses_main (interface : Interface) (offset : Nat) :
    witnesses (Circuit.ops (main interface) offset) =
      witnessBatches interface offset := by
  change witnesses (operations interface offset) = _
  simp [operations, witnessBatches, witnesses, Op.witnesses,
    quotientBooleanOps]

end NightstreamFPrime.Gadgets.Sampling.Candidate16Five
