import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority
import NightstreamFPrime.Spec.AjtaiSetupV1

/-!
Owns compact executable conformance vectors for the SHAKE128 wide-reduction
Ajtai setup. It includes the FIPS 202 SHAKE128 example, selected indexed
coefficients, the complete raw authority descriptor, and one complete key
element.
-/

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

/-- Canonical setup fixture. Entry 3 is the first 32 bytes of SHAKE128 of
the empty message (FIPS 202 example). The last entry is all 1,728 output
bytes of production element (1, 32768); coefficients 5, 10, 15, 26, 31, 36,
47 and 52 cross a 168-byte rate boundary. -/
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
