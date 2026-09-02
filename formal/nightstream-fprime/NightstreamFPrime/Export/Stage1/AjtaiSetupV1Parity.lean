import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority
import NightstreamFPrime.Spec.AjtaiSetupV1

/-!
Owns compact executable conformance vectors for the approved ChaCha20
wide-reduction Ajtai setup. It includes the RFC-8439 block-function vector,
selected indexed coefficients, and the complete raw authority descriptor.
-/

namespace NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec.AjtaiSetupV1

def schema : Nat := 3

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

/-- Canonical setup fixture. The first block is RFC 8439 Section 2.3.2:
counter 1 and nonce `000000090000004a00000000`. -/
def parityValue : Value :=
  .array [
    .atom schema,
    natListValue setupIdBytes,
    natListValue testSeed.bytes,
    natListValue (ChaCha20.blockWords testSeed.bytes
      0x09000000 0x4a000000 1),
    natListValue Poseidon2HashChainV1SetupAuthority.productionSeed.bytes,
    .array [
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 0,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 53,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        1 32768 17,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        21 4900508 53],
    natListValue Poseidon2HashChainV1SetupAuthority.authorityNats]

end NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity
