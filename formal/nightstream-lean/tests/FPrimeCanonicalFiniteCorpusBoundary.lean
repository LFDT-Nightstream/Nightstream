import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary

/-!
Focused regression for the exact logical boundary of the generated one-slot
Rust differential corpora.
-/

namespace Nightstream.Tests.FPrimeCanonicalFiniteCorpusBoundary

open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary

#check Step.flippedOutside_conformsOnGenerated
#check Step.flippedOutside_disagrees
#check Step.not_attemptedUniversalBridge
#check Terminal.flippedOutside_conformsOnGenerated
#check Terminal.flippedOutside_disagrees
#check Terminal.not_attemptedUniversalBridge

example : ¬ Step.AttemptedUniversalBridge :=
  Step.not_attemptedUniversalBridge

example : ¬ Terminal.AttemptedUniversalBridge :=
  Terminal.not_attemptedUniversalBridge

end Nightstream.Tests.FPrimeCanonicalFiniteCorpusBoundary
