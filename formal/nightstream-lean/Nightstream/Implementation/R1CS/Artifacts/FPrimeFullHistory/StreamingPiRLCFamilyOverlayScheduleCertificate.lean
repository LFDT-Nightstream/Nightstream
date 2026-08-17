import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionIdentity
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained

/-!
Contract: exact verifier-owned seed schedule for the production PiRLC family
overlay receipt.

Assurance tier: structural Rust-to-Lean artifact certificate.

Owns the equality between the literal Rust schedule fields and the schedule
derived from the fixed production seed, dimensions, and rejection fuel.

Does not own sampler liveness, sampled coefficients, row placement, or
lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayScheduleCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup
open Nightstream.Implementation.R1CS

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit

abbrev expectedSchedule : SeededPhi81Sampler.Schedule :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup.expectedSchedule

/-- The generated schedule is the verifier-owned schedule. This leaf uses
kernel reduction of the six fixed seed chunks. It does not evaluate sampler
coefficients or the complete overlay artifact. -/
theorem schedule_exact :
    audit.chunkSize = expectedSchedule.chunkSize /\
      audit.chunkSeedsByRow = expectedSchedule.seedsByOutput /\
      expectedSchedule.rejectionFuel = 16 /\
      audit.chunkSeedsByRow.length = 2 /\
      (audit.chunkSeedsByRow.all fun row => decide (row.length = 3)) = true /\
      (audit.chunkSeedsByRow.all fun row =>
        row.all fun seed =>
          decide (seed.length = 32) &&
            seed.all fun byte => decide (byte < 256)) = true := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayScheduleCertificate
