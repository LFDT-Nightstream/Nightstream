import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.StepCases
import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.TerminalCases

/-!
Kernel-checked replay of the Rust-generated linked one-slot differential
corpus.

Tier: bounded Rust-conformant differential. The results cover the exact
generated step and terminal profiles and acceptance mappings only. Primitive
receipt correctness, general Rust refinement, and R1CS refinement remain out
of scope.
-/

namespace Nightstream.Tests.FPrimeCanonicalSharedInputDifferential

open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot
open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated

/-- On all nine shared inputs, the independently evaluated frozen checker
agrees with the acceptance bit recorded by the production Rust verifier. -/
theorem generated_all_agree :
    all.all stepAgrees = true := by
  decide

example : all.length = 9 := by
  decide

example : honest.all stepAccepted = true := by
  decide

example : mutations.all (fun case => !stepAccepted case) = true := by
  decide

end Nightstream.Tests.FPrimeCanonicalSharedInputDifferential

namespace Nightstream.Tests.FPrimeCanonicalSharedTerminalDifferential

open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot

/-- On all seven shared terminal inputs, the independently evaluated frozen
checker agrees with the acceptance bit recorded by `verify_uncompressed`. -/
theorem generated_all_agree :
    Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.all.all
      terminalAgrees = true := by
  decide

example :
    Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.all.length =
      7 := by
  decide

example :
    Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.honest.all
      terminalAccepted = true := by
  decide

example :
    Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.mutations.all
      (fun case => !terminalAccepted case) = true := by
  decide

end Nightstream.Tests.FPrimeCanonicalSharedTerminalDifferential
