import NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar
import NightstreamFPrime.Circuit.WitnessSupport

/-! Child-owned read support for the single signed-split hint. -/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar

open NightstreamFPrime.Circuit

@[simp] theorem witnesses_main (interface : Interface) (offset : Nat) :
    witnesses (Circuit.ops (main interface) offset) =
      [WitnessBatch.hinted offset [signHint interface offset]] := by
  change witnesses (operations interface offset) = _
  simp [operations, witnesses, Op.witnesses]

theorem witnesses_main_readsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (parentSupported : (interface.parent offset).VarsSatisfy allowed) :
    ∀ batch ∈ witnesses (Circuit.ops (main interface) offset),
      batch.ReadsSatisfy allowed := by
  rw [witnesses_main]
  intro batch member
  rw [List.mem_singleton] at member
  subst batch
  simp only [WitnessBatch.readsSatisfy_hinted, List.mem_singleton,
    forall_eq, signHint, Hint.source]
  exact parentSupported

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar
