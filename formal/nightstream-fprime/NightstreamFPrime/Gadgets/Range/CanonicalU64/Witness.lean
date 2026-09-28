import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Circuit.WitnessSupport

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

/-- Arithmetic recipes and hints use the same declared source and local cells. -/
theorem witnessBatches_readsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (sourceSupported : (interface.source offset).VarsSatisfy allowed)
    (localSupported : ∀ index, index < auxiliaryCount → allowed (offset + index)) :
    ∀ batch ∈ witnessBatches interface offset, batch.ReadsSatisfy allowed := by
  intro batch member
  simp only [witnessBatches, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · rw [WitnessBatch.readsSatisfy_hinted]
    intro hint member
    rcases List.mem_map.mp member with ⟨index, _, rfl⟩
    exact sourceSupported
  · simp only [WitnessBatch.readsSatisfy_hinted, List.mem_singleton,
      forall_eq, inverseHint, Hint.source]
    exact highDifference_varsSatisfy offset allowed localSupported
  · simp only [WitnessBatch.readsSatisfy_arithmetic, List.mem_singleton, forall_eq]
    exact flagRecipe_varsSatisfy offset allowed localSupported

end NightstreamFPrime.Gadgets.Range.CanonicalU64
