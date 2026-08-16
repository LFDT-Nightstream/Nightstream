import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup

/-!
Contract: exact verifier-owned production setup for the streaming PiRLC input
commitment.

Owns the fixed Rust seed, rejection fuel 16, and checked successful bounded
sampler execution for the rank-two, 67,650-column verifier key.

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

/-- Master seed used by Rust's
`PI_RLC_INPUT_COORDINATE_SIS_CONFIG`. -/
def rustSeedBytes : List Nat := List.replicate 32 201

def rustDomain : Nat := 0x5049_524C_4349_4E50

theorem exact_rust_identity :
    rustSeedBytes.length = 32 /\
      (∀ byte ∈ rustSeedBytes, byte < 256) /\
      rustDomain = 5785245683833982544 := by
  constructor
  · simp [rustSeedBytes]
  constructor
  · intro byte member
    have : byte = 201 := by
      simpa [rustSeedBytes] using member
    omega
  · rfl

/-- Typed verifier seed for the fixed Rust production identity. -/
def rustSeed : SeededAjtai.Seed where
  bytes := rustSeedBytes
  length_eq := exact_rust_identity.1
  canonical := exact_rust_identity.2.1

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
    productionSetup.setup.seed.bytes = rustSeedBytes /\
      productionSetup.setup.rejectionFuel = 16 := by
  exact ⟨rfl, rfl⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup
