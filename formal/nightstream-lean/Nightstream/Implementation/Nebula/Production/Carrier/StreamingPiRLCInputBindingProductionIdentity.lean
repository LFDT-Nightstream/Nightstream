import Nightstream.Implementation.R1CS.Core.SeededAjtai

/-!
Contract: proof-free verifier identity for the production PiRLC input
commitment schedule.

Owns the fixed Rust master seed, domain, rejection fuel, and schedule geometry
for the supported `b = 2`, `k_rho = 16` profile.

Does not own sampler liveness, sampled coefficients, generated rows,
placement, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup

open Nightstream.Implementation.R1CS

/-- Master seed used by Rust's
`PI_RLC_INPUT_COORDINATE_SIS_CONFIG`. -/
def rustSeedBytes : List Nat := List.replicate 32 201

def rustDomain : Nat := 0x5049_524C_4349_4E50

def rejectionFuel : Nat := 16

def scheduleVerifierRows : Nat := 2

def scheduleMessageColumnCount : Nat := 76670

theorem exact_rust_identity :
    rustSeedBytes.length = 32 /\
      (∀ byte ∈ rustSeedBytes, byte < 256) /\
      rustDomain = 5785245683833982544 /\
      rejectionFuel = 16 /\
      scheduleVerifierRows = 2 /\
      scheduleMessageColumnCount = 76670 := by
  constructor
  · simp [rustSeedBytes]
  constructor
  · intro byte member
    have : byte = 201 := by
      simpa [rustSeedBytes] using member
    omega
  · exact ⟨rfl, rfl, rfl, rfl⟩

/-- Typed verifier seed for the fixed Rust production identity. -/
def rustSeed : SeededAjtai.Seed where
  bytes := rustSeedBytes
  length_eq := exact_rust_identity.1
  canonical := exact_rust_identity.2.1

def expectedSchedule : SeededPhi81Sampler.Schedule :=
  SeededAjtai.schedule rustSeed.bytes scheduleVerifierRows
    scheduleMessageColumnCount rejectionFuel

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup
