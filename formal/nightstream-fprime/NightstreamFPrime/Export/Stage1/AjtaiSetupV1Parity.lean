import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority
import NightstreamFPrime.Spec.AjtaiSetupV1

namespace NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec (ringDegree)
open NightstreamFPrime.Spec.AjtaiSetupV1

def schema : Nat := 4

def testSeedBytes : List Nat := List.range 32

def testSeed : Seed where
  bytes := testSeedBytes
  length_eq := by simp [testSeedBytes]
  canonical := by
    intro byte member
    simp [testSeedBytes] at member
    omega

def natListValue (values : List Nat) : Value :=
  .array (values.map Value.atom)

def coordinateValue (seed : Seed) (row block lane : Nat) : Value :=
  .array [.atom row, .atom block, .atom lane,
    .atom (wideCoefficientNat seed.bytes row block lane)]

/-- Entry 3 is the FIPS 202 SHAKE128 example for the empty message. -/
def parityValue : Value :=
  .array [
    .atom schema,
    natListValue setupIdBytes,
    natListValue testSeed.bytes,
    natListValue (Shake128.bytes [] 32),
    natListValue Poseidon2HashChainV1SetupAuthority.productionSeed.bytes,
    .array [
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 0,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 53,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        1 32768 17,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        21 (Poseidon2HashChainV1SetupAuthority.messageColumns - 1) 53],
    natListValue Poseidon2HashChainV1SetupAuthority.authorityNats,
    natListValue (Shake128.bytes
      (elementInput Poseidon2HashChainV1SetupAuthority.productionSeed.bytes 1 32768)
      (32 * ringDegree))]

end NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity
