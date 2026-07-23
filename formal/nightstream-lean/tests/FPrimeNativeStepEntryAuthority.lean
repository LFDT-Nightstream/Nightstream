import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Composition
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Generated.Receipts

namespace Nightstream.Tests.FPrimeNativeStepEntryAuthority

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

/-- Lifecycle data needed to authenticate the prior active state is separate
from the native call receipt, which contains only the newly folded running
digest. -/
def recursiveEntryBoundary : BoundaryReceipt where
  initialNebula := none
  calls := [
    .runningDigest ⟨1⟩ ⟨1⟩,
    .hash (.initialBoundary {
      structureDigest := ⟨1⟩
      publicInputLength := some 1
    }) ⟨2⟩
  ]

/-- The active entry obligation is inhabited once lifecycle replay supplies
the authoritative prior running digest and initial-boundary hash. -/
theorem honestRecursive_entryAuthority :
    EntryAuthority
      (boundaryHashSemantics Generated.honestRecursive recursiveEntryBoundary)
      (boundaryStepSemantics Generated.honestRecursive recursiveEntryBoundary)
      .stateless Generated.context Generated.state2 := by
  unfold EntryAuthority
  refine ⟨rfl, by decide, by decide, rfl, rfl, ?_⟩
  constructor
  · rfl
  · rfl
  · rfl
  · intro _
    rfl

end Nightstream.Tests.FPrimeNativeStepEntryAuthority
