import NightstreamFPrime.Gadgets.Sampling.Candidate16Five
import NightstreamFPrime.Circuit.WitnessSupport

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

/-- Hint sources and the reject recipe use only the caller's candidate bits
and the decoder's own quotient slot. -/
theorem witnessBatches_readsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (candidateSupported : (interface.candidate offset).VarsSatisfy allowed)
    (bitsSupported : ∀ index, index < candidateBitCount →
      (interface.candidateBit offset index).VarsSatisfy allowed)
    (localSupported : ∀ index, index < auxiliaryCount → allowed (offset + index)) :
    ∀ batch ∈ witnessBatches interface offset, batch.ReadsSatisfy allowed := by
  intro batch member
  simp only [witnessBatches, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · rw [WitnessBatch.readsSatisfy_hinted]
    intro hint member
    simp only [quotientRemainderHints, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> exact candidateSupported
  · rw [WitnessBatch.readsSatisfy_hinted]
    intro hint member
    rcases List.mem_map.mp member with ⟨index, _, rfl⟩
    exact localSupported 0 (by decide)
  · simp only [WitnessBatch.readsSatisfy_arithmetic, List.mem_singleton, forall_eq]
    exact productExpr_varsSatisfy (interface.candidateBit offset) candidateBitCount
      allowed bitsSupported

end NightstreamFPrime.Gadgets.Sampling.Candidate16Five
