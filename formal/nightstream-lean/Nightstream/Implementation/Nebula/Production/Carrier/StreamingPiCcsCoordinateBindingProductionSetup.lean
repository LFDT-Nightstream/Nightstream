import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingSetup

/-!
Contract: exact verifier-owned production setup for the streaming PiCCS
coordinate commitment.

Owns the fixed Rust seed, rejection fuel 16, and the checked successful
bounded sampler execution for the rank-two, 16,112-column verifier key.

Does not own Rust ChaCha8 conformance, generated coordinate rows, phase
selection, accumulator updates, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingProductionSetup

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS

/-- Typed verifier seed for the fixed Rust production identity. -/
def rustSeed : SeededAjtai.Seed where
  bytes := rustSeedBytes
  length_eq := exact_rust_identity.1
  canonical := exact_rust_identity.2.1

/-- The bounded sampler succeeds for the exact production seed and fuel.
This definition makes sampler liveness verifier-owned setup evidence. -/
def productionSetup : ProductionSetup where
  setup := {
    seed := rustSeed
    rejectionFuel := 16
    samplingSuccess := by
      let execution :=
        (SeededAjtai.schedule rustSeed.bytes verifierRows
          messageColumnCount 16).baseRotations
            SeededPhi81Sampler.pureStream messageColumnCount
      have success : execution ≠ none := by native_decide
      cases result : execution with
      | none => exact False.elim (success result)
      | some outputs => exact ⟨outputs, result⟩ }
  seed_eq := rfl

theorem exact_identity :
    productionSetup.setup.seed.bytes = rustSeedBytes ∧
      productionSetup.setup.rejectionFuel = 16 := by
  exact ⟨rfl, rfl⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingProductionSetup
