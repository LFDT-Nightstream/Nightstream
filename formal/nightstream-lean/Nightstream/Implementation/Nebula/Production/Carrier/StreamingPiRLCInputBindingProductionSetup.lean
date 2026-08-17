import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionIdentity

/-!
Contract: exact verifier-owned production setup for the streaming PiRLC input
commitment.

Owns checked successful bounded sampler execution for the proof-free
production identity and its rank-two, 76,670-column verifier key.

Does not own Rust ChaCha8 conformance, generated family rows, phase selection,
residual telescoping, or Module-SIS hardness. The map is outside the pinned
estimator ceiling, so this file does not claim an estimator result.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.R1CS

/-- The fixed production identity includes successful bounded-sampler
evidence. The rejection fuel is verifier data, not prover advice. -/
structure ProductionSetup where
  setup : SeededAjtai.Setup verifierRows messageColumnCount
  seed_eq : setup.seed.bytes = rustSeedBytes

theorem exact_chunk_geometry :
    SeededAjtai.chunkSize messageColumnCount = 32768 /\
      SeededAjtai.chunkCount messageColumnCount = 3 := by
  decide

def productionSetup : ProductionSetup where
  setup := {
    seed := rustSeed
    rejectionFuel := rejectionFuel
    samplingSuccess := by
      let execution :=
        (SeededAjtai.schedule rustSeed.bytes verifierRows
          messageColumnCount rejectionFuel).baseRotations
            SeededPhi81Sampler.pureStream messageColumnCount
      have success : execution ≠ none := by native_decide
      cases result : execution with
      | none => exact False.elim (success result)
      | some outputs => exact ⟨outputs, result⟩ }
  seed_eq := rfl

theorem exact_identity :
    productionSetup.setup.seed.bytes = rustSeedBytes /\
      productionSetup.setup.rejectionFuel = 16 := by
  exact ⟨rfl, rfl⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup
